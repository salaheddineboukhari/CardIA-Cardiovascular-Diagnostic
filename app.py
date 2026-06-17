from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for, flash
from twin.twin_logic import CardiacTwin
from twin.explainable_twin import ExplainableTwin
from twin.temporal_predictor import TemporalPredictor
from twin.recommendation_engine import RecommendationEngine
from twin.report_generator import ReportGenerator
from database import PatientDatabase
import secrets
import os
import numpy as np
import pandas as pd
import io
from datetime import datetime
import json
import uuid
import re
import unicodedata

# AUTHENTIFICATION - IMPORTS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# CONFIGURATION AUTHENTIFICATION
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page'

# Fichiers de données
USERS_FILE = 'users.json'
HISTORY_FILE = 'patient_history.json'
PATIENTS_FILE = 'patients.json'

# FONCTION UTILITAIRE POUR GÉNÉRER UN EMAIL

def generate_email_from_name(full_name):
    """Génère un email à partir du nom complet"""
    # Normaliser et supprimer les accents
    name_normalized = unicodedata.normalize('NFKD', full_name).encode('ASCII', 'ignore').decode('ASCII')
    # Remplacer les espaces et caractères spéciaux par des points
    email_base = re.sub(r'[^a-zA-Z0-9.]', '.', name_normalized.lower())
    # Supprimer les points multiples
    email_base = re.sub(r'\.+', '.', email_base)
    # Supprimer les points au début et à la fin
    email_base = email_base.strip('.')
    return f"{email_base}@test.com"


# CLASSE UTILISATEUR

class User:
    def __init__(self, id, email, password_hash, role, name):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.name = name
        self.birth_date = None
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        return str(self.id)
    
    @staticmethod
    def get_by_email(email):
        if not os.path.exists(USERS_FILE):
            return None
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for user in data.get('users', []):
                if user['email'] == email:
                    u = User(
                        id=user['id'],
                        email=user['email'],
                        password_hash=user['password_hash'],
                        role=user['role'],
                        name=user['name']
                    )
                    if user.get('role') == 'patient' and os.path.exists(PATIENTS_FILE):
                        try:
                            with open(PATIENTS_FILE, 'r', encoding='utf-8') as pf:
                                pts = json.load(pf)
                            for pt in pts:
                                if pt.get('full_name', '').strip().lower() == user['name'].strip().lower():
                                    u.birth_date = pt.get('birth_date')
                                    break
                        except Exception:
                            pass
                    return u
        return None

    @staticmethod
    def get_by_id(user_id):
        if not os.path.exists(USERS_FILE):
            return None
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for user in data.get('users', []):
                if str(user['id']) == str(user_id):
                    u = User(
                        id=user['id'],
                        email=user['email'],
                        password_hash=user['password_hash'],
                        role=user['role'],
                        name=user['name']
                    )
                    if user.get('role') == 'patient' and os.path.exists(PATIENTS_FILE):
                        try:
                            with open(PATIENTS_FILE, 'r', encoding='utf-8') as pf:
                                pts = json.load(pf)
                            for pt in pts:
                                if pt.get('full_name', '').strip().lower() == user['name'].strip().lower():
                                    u.birth_date = pt.get('birth_date')
                                    break
                            else:
                                u.birth_date = None
                        except Exception:
                            u.birth_date = None
                    else:
                        u.birth_date = None
                    return u
        return None

    def save(self):
        users_data = []
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f).get('users', [])
        
        user_dict = {
            'id': self.id,
            'email': self.email,
            'password_hash': self.password_hash,
            'role': self.role,
            'name': self.name,
            'birth_date': getattr(self, 'birth_date', None),
            'created_at': datetime.now().isoformat()
        }
        
        found = False
        for i, u in enumerate(users_data):
            if u['id'] == self.id:
                users_data[i] = user_dict
                found = True
                break
        if not found:
            users_data.append(user_dict)
        
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'users': users_data}, f, indent=2, ensure_ascii=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_doctor(self):
        return self.role == 'doctor'
    
    def is_patient(self):
        return self.role == 'patient'

# FONCTIONS DE GESTION DE L'HISTORIQUE

def get_patient_history(user_id):
    if not os.path.exists(HISTORY_FILE):
        return []

    patient_user = User.get_by_id(user_id)
    if not patient_user:
        return []

    patient_name = patient_user.name.strip().lower()

    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return sorted([
        c for c in data.get('consultations', [])
        if c.get('patient_name', '').strip().lower() == patient_name
        and c.get('patient_name', '').strip() not in ['', 'Patient inconnu']
    ], key=lambda x: x['date'])

def add_consultation(user_id, patient_name, patient_data, prediction, created_by_role='doctor'):
    if not patient_name or patient_name.strip() == "" or patient_name == "Patient inconnu":
        print(" Consultation ignorée - patient invalide")
        return None

    try:
        consultations = []

        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                consultations = data.get('consultations', [])

        new_id = 1
        if consultations:
            new_id = max(c.get('id', 0) for c in consultations) + 1

        new_consultation = {
            'id': new_id,
            'user_id': user_id,
            'patient_name': patient_name,
            'type': 'risk',
            'date': datetime.now().isoformat(),
            'patient': patient_data,
            'prediction': prediction,
            'created_by_role': created_by_role,
            'shared_with_doctor': created_by_role == 'doctor',
            'shared_with_doctors': [user_id] if created_by_role == 'doctor' else []
        }

        consultations.append(new_consultation)

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        print(f" Consultation RISQUE sauvegardée - Patient: {patient_name}")
        return new_consultation

    except Exception as e:
        print("❌ Erreur sauvegarde consultation:", e)
        return None

def get_all_doctors():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [
        {'id': u['id'], 'name': u['name']}
        for u in data.get('users', [])
        if u.get('role') == 'doctor'
    ]

def get_all_consultations(shared_only=True):
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    consultations = data.get('consultations', [])

    if not shared_only:
        return consultations

    doctor_id = getattr(current_user, 'id', None)
    result = []
    for c in consultations:
        # Cas 1 : Consultation créée par le médecin lui-même → toujours visible
        if c.get('created_by_role') == 'doctor' and c.get('user_id') == doctor_id:
            result.append(c)
        # Cas 2 : Consultation créée par un patient → visible SEULEMENT si partagée
        elif c.get('created_by_role') == 'patient':
            shared_doctors = c.get('shared_with_doctors', [])
            if doctor_id in shared_doctors:
                result.append(c)
    return result

def add_ecg_consultation(user_id, patient_name, ecg_result, created_by_role='doctor'):
    if not patient_name or patient_name.strip() == "":
        print(" ECG ignoré - patient invalide")
        return None

    try:
        consultations = []

        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                consultations = data.get('consultations', [])

        new_id = 1
        if consultations:
            new_id = max(c.get('id', 0) for c in consultations) + 1

        new_consultation = {
            'id': new_id,
            'user_id': user_id,
            'patient_name': patient_name,
            'type': 'ecg',
            'date': datetime.now().isoformat(),
            'result': ecg_result,
            'created_by_role': created_by_role,
            'shared_with_doctor': created_by_role == 'doctor',
            'shared_with_doctors': [user_id] if created_by_role == 'doctor' else []
        }

        consultations.append(new_consultation)

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        print(f" Consultation ECG sauvegardée - Patient: {patient_name}")
        return new_consultation

    except Exception as e:
        print(" Erreur sauvegarde ECG:", e)
        return None

def add_xray_consultation(user_id, patient_name, xray_result, created_by_role='doctor'):
    """Sauvegarde une analyse radio dans l'historique."""
    if not patient_name or patient_name.strip() == "":
        print(" Radio ignorée - patient invalide")
        return None

    try:
        consultations = []

        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                consultations = data.get('consultations', [])

        new_id = 1
        if consultations:
            new_id = max(c.get('id', 0) for c in consultations) + 1

        new_consultation = {
            'id': new_id,
            'user_id': user_id,
            'patient_name': patient_name,
            'type': 'xray',
            'date': datetime.now().isoformat(),
            'result': xray_result,
            'created_by_role': created_by_role,
            'shared_with_doctor': created_by_role == 'doctor',
            'shared_with_doctors': [user_id] if created_by_role == 'doctor' else []
        }

        consultations.append(new_consultation)

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        print(f" Consultation RADIO sauvegardée - Patient: {patient_name}")
        return new_consultation

    except Exception as e:
        print(" Erreur sauvegarde radio:", e)
        return None

# FONCTIONS DE GESTION DES PATIENTS

def load_patients():
    if os.path.exists(PATIENTS_FILE):
        try:
            with open(PATIENTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("⚠️ Fichier patients.json corrompu, recréation...")
            return []
    return []

def get_doctor_patients(doctor_id):
    all_patients = load_patients()

    own_patients = {
        p['full_name'].strip().lower(): p
        for p in all_patients
        if p.get('created_by') == doctor_id or p.get('doctor_id') == doctor_id
    }

    shared_patients = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                hist = json.load(f)
            for c in hist.get('consultations', []):
                if doctor_id in c.get('shared_with_doctors', []):
                    pname = c.get('patient_name', '').strip().lower()
                    if pname and pname not in own_patients:
                        for p in all_patients:
                            if p.get('full_name', '').strip().lower() == pname:
                                shared_patients[pname] = p
                                break
                        else:
                            if pname not in shared_patients:
                                shared_patients[pname] = {
                                    'full_name': c.get('patient_name'),
                                    'birth_date': '',
                                    'shared_only': True
                                }
        except Exception as e:
            print(f"⚠️ Erreur lecture historique: {e}")

    result = list(own_patients.values()) + list(shared_patients.values())
    return result

def save_patients(patients):
    with open(PATIENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)

# DÉCORATEURS POUR LES RÔLES

def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if not current_user.is_doctor():
            return "Accès réservé aux médecins", 403
        return f(*args, **kwargs)
    return decorated

def patient_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if not current_user.is_patient():
            return "Accès réservé aux patients", 403
        return f(*args, **kwargs)
    return decorated

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

# INITIALISATION DES MODULES

try:
    twin = CardiacTwin('models/')
    print("✅ Modèle de risque chargé avec succès")
    explainer = ExplainableTwin(twin.model, twin.all_features)
    temporal = TemporalPredictor()
    recommender = RecommendationEngine()
    reporter = ReportGenerator()
    db = PatientDatabase()
except Exception as e:
    print(f"❌ Erreur chargement modèle risque: {e}")
    twin = None
    explainer = None
    temporal = None
    recommender = None
    reporter = None
    db = PatientDatabase()

# CHARGEMENT DU MODÈLE ECG

try:
    from ecg_analysis.ecg_classifier import ECGClassifier
    ecg_classifier = ECGClassifier(
        model_path='models/cnn_heartbeat_model.h5',
        scaler_path='models/scaler_ecg.pkl'
    )
    print("✅ Modèle ECG chargé avec succès")
except Exception as e:
    print(f" Erreur chargement modèle ECG: {e}")
    ecg_classifier = None

# CHARGEMENT MODÈLE RADIO - TorchXRayVision DenseNet-121

import base64 as b64

try:
    import torchxrayvision as xrv
    import torchvision.transforms as xrv_transforms
    import torch

    xray_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    xray_model.eval()
    print(" Modèle TorchXRayVision (DenseNet-121) chargé - AUC Cardiomégalie: 0.90")
except Exception as e:
    print(f" Erreur chargement TorchXRayVision: {e}")
    xray_model = None


# ROUTES D'AUTHENTIFICATION


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        user = User.get_by_email(email)
        
        if user and user.check_password(password) and user.role == role:
            login_user(user)
            if user.is_doctor():
                return redirect(url_for('index'))
            else:
                return redirect(url_for('patient_dashboard'))
        else:
            return render_template('login.html', error='Email ou mot de passe incorrect')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', 'patient')
        birth_date = request.form.get('birth_date', '').strip()

        if not name or not email or not password:
            return render_template('register.html', error='Tous les champs obligatoires doivent être remplis')

        if role == 'patient' and not birth_date:
            return render_template('register.html', error='La date de naissance est obligatoire pour un patient')

        if User.get_by_email(email):
            return render_template('register.html', error='Cet email est déjà utilisé')

        new_user = User(
            id=str(uuid.uuid4())[:8],
            email=email,
            password_hash='',
            role=role,
            name=name
        )
        new_user.set_password(password)
        new_user.birth_date = birth_date if role == 'patient' else None
        new_user.save()

        if role == 'patient':
            patients = load_patients()
            already_exists = any(
                p.get('full_name', '').strip().lower() == name.lower()
                for p in patients
            )
            if not already_exists:
                new_patient_record = {
                    'id': max((p['id'] for p in patients), default=0) + 1,
                    'full_name': name,
                    'birth_date': birth_date,
                    'created_at': datetime.now().isoformat(),
                    'created_by': 'self',
                    'doctor_id': None
                }
                patients.append(new_patient_record)
                save_patients(patients)
                print(f"✅ Patient auto-créé dans patients.json: {name}")

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/patient-dashboard')
@login_required
@patient_required
def patient_dashboard():
    history = get_patient_history(current_user.id)

    risk_history = [c for c in history if c.get('type') == 'risk' and 'prediction' in c]
    last_risk = risk_history[-1]['prediction']['probability'] * 100 if risk_history else None
    last_date = history[-1]['date'][:10] if history else None

    doctors = get_all_doctors()
    
    existing_risk = any(c.get('type') == 'risk' for c in history)
    
    return render_template('patient_dashboard.html',
                         user=current_user,
                         history=history,
                         total_consultations=len(history),
                         last_risk=last_risk,
                         last_date=last_date,
                         doctors=doctors,
                         existing_risk=existing_risk)


# ROUTES DE GESTION DES PATIENTS


@app.route('/register-patient', methods=['GET', 'POST'])
@login_required
@doctor_required
def register_patient():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        birth_date = request.form.get('birth_date', '')
        phone = request.form.get('phone', '')
        email = request.form.get('email', '')
        address = request.form.get('address', '')
        blood_type = request.form.get('blood_type', '')
        allergies = request.form.get('allergies', '')
        ssn = request.form.get('ssn', '')
        
        # Si l'email n'est pas fourni, générer un email par défaut
        if not email:
            email = generate_email_from_name(full_name)
        
        patients = load_patients()
        new_patient = {
            'id': max((p.get('id', 0) for p in patients), default=0) + 1,
            'full_name': full_name,
            'birth_date': birth_date,
            'ssn': ssn,
            'phone': phone,
            'email': email,
            'address': address,
            'doctor': current_user.name,
            'doctor_id': current_user.id,
            'blood_type': blood_type,
            'allergies': allergies,
            'created_at': datetime.now().isoformat(),
            'created_by': current_user.id
        }
        patients.append(new_patient)
        save_patients(patients)
        
        # --- CRÉATION AUTOMATIQUE DU COMPTE PATIENT ---
        existing_user = User.get_by_email(email)
        default_password = "1234"
        
        if not existing_user:
            user_id = str(uuid.uuid4())[:8]
            
            new_user = User(
                id=user_id,
                email=email,
                password_hash='',
                role='patient',
                name=full_name
            )
            new_user.set_password(default_password)
            new_user.birth_date = birth_date
            new_user.save()
            
            flash(f'Patient {full_name} créé avec succès. Compte patient: {email} / Mot de passe: {default_password}', 'success')
        else:
            flash(f'Patient {full_name} créé avec succès (compte existant). Email: {email}', 'success')
        
        return redirect(url_for('index'))
    
    return render_template('patient_register.html')

@app.route('/quick-add-patient', methods=['POST'])
@login_required
@doctor_required
def quick_add_patient():
    try:
        data = request.get_json()
        name = data.get('full_name', '').strip()
        birth = data.get('birth_date', '')
        
        if not name:
            return jsonify({'success': False, 'error': 'Nom requis'})
        
        # Générer l'email par défaut
        email = generate_email_from_name(name)
        
        patients = load_patients()
        
        for p in patients:
            if p.get('full_name', '').lower() == name.lower():
                return jsonify({'success': False, 'error': 'Patient déjà existant'})
        
        new_patient = {
            'id': max((p.get('id', 0) for p in patients), default=0) + 1,
            'full_name': name,
            'birth_date': birth or '',
            'email': email,
            'created_at': datetime.now().isoformat(),
            'created_by': current_user.id,
            'doctor_id': current_user.id
        }
        patients.append(new_patient)
        save_patients(patients)
        
        # --- CRÉATION AUTOMATIQUE DU COMPTE PATIENT ---
        existing_user = User.get_by_email(email)
        default_password = "1234"
        
        if not existing_user:
            user_id = str(uuid.uuid4())[:8]
            
            new_user = User(
                id=user_id,
                email=email,
                password_hash='',
                role='patient',
                name=name
            )
            new_user.set_password(default_password)
            new_user.birth_date = birth
            new_user.save()
        
        return jsonify({
            'success': True,
            'email': email,
            'password': default_password,
            'message': f'Compte patient créé: {email} / {default_password}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/patients-list')
@login_required
@doctor_required
def patients_list():
    patients = load_patients()
    return render_template('patient_list.html', patients=patients)


# ROUTES PRINCIPALES


@app.route('/')
@login_required
def index():
    if current_user.is_patient():
        return redirect(url_for('patient_dashboard'))
    patients = get_doctor_patients(current_user.id)
    return render_template('index.html', patients_list=patients)


@app.route('/patient-risk', methods=['GET', 'POST'])
@login_required
@patient_required
def patient_risk():
    patient_name = current_user.name.strip()

    # VÉRIFICATION : le patient a-t-il déjà une analyse de risque ?
    existing_risk_analysis = None
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            hist_data = json.load(f)
        for c in hist_data.get('consultations', []):
            if (c.get('patient_name', '').strip().lower() == patient_name.lower() 
                and c.get('type') == 'risk'
                and 'prediction' in c):
                existing_risk_analysis = c
                break

    if existing_risk_analysis:
        flash("Vous avez déjà effectué votre analyse de risque. Une seule analyse est autorisée par patient.", "info")
        return redirect(url_for('patient_dashboard'))

    if request.method == 'POST':
        try:
            if twin is None:
                return render_template('error.html', error="Modèle de prédiction non disponible.")

            patient_birth = getattr(current_user, 'birth_date', None)

            if patient_birth:
                try:
                    birth = datetime.strptime(patient_birth, '%Y-%m-%d')
                    today = datetime.now()
                    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                except ValueError:
                    age = int(request.form.get('age', 50))
            else:
                age = int(request.form.get('age', 50))

            patient_data = {
                'age':      age,
                'sex':      int(request.form.get('sex', 1)),
                'cp':       int(request.form.get('cp', 4)),
                'trestbps': float(request.form.get('trestbps', 120)),
                'chol':     float(request.form.get('chol', 200)),
                'fbs':      int(request.form.get('fbs', 0)),
                'restecg':  int(request.form.get('restecg', 0)),
                'thalach':  float(request.form.get('thalach', max(60, 220 - age))),
                'exang':    int(request.form.get('exang', 0)),
                'oldpeak':  float(request.form.get('oldpeak', 0)),
                'slope':    int(request.form.get('slope', 1)),
                'ca':       int(request.form.get('ca', 0)),
                'thal':     int(request.form.get('thal', 3)),
            }

            session['patient_data'] = patient_data
            result = twin.predict(patient_data)
            session['last_result'] = result
            session['current_patient_name'] = patient_name

            add_consultation(current_user.id, patient_name, patient_data, result, created_by_role='patient')

            return render_template('result.html',
                                 patient=patient_data,
                                 result=result,
                                 patient_name=patient_name)

        except Exception as e:
            print(f"❌ Erreur patient-risk: {e}")
            import traceback; traceback.print_exc()
            return render_template('error.html', error=str(e))

    patient_age = None
    patient_birth = getattr(current_user, 'birth_date', None)
    if patient_birth:
        try:
            birth = datetime.strptime(patient_birth, '%Y-%m-%d')
            today = datetime.now()
            patient_age = today.year - birth.year - (
                (today.month, today.day) < (birth.month, birth.day)
            )
        except (ValueError, TypeError):
            patient_age = None

    return render_template('index.html', patients_list=[], patient_age=patient_age)

@app.route('/predict', methods=['POST'])
@login_required
@doctor_required
def predict():
    try:
        if twin is None:
            return render_template('error.html', error="Modèle de prédiction non disponible.")

        patient_name = request.form.get('patient_name', '').strip()

        if not patient_name or patient_name == 'Patient inconnu':
            return render_template('error.html', error="Veuillez sélectionner un patient.")

        patients = load_patients()
        patient_birth = None
        patient_found = False
        for p in patients:
            if p.get('full_name', '').strip() == patient_name:
                patient_found = True
                patient_birth = p.get('birth_date')
                break

        if not patient_found:
            return render_template('error.html', error=f"Patient '{patient_name}' introuvable.")

        if patient_birth:
            try:
                birth = datetime.strptime(patient_birth, '%Y-%m-%d')
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            except ValueError:
                age = int(request.form.get('age', 50))
        else:
            age = int(request.form.get('age', 50))

        patient_data = {
            'age':      age,
            'sex':      int(request.form.get('sex', 1)),
            'cp':       int(request.form.get('cp', 4)),
            'trestbps': float(request.form.get('trestbps', 120)),
            'chol':     float(request.form.get('chol', 200)),
            'fbs':      int(request.form.get('fbs', 0)),
            'restecg':  int(request.form.get('restecg', 0)),
            'thalach':  float(request.form.get('thalach', max(60, 220 - age))),
            'exang':    int(request.form.get('exang', 0)),
            'oldpeak':  float(request.form.get('oldpeak', 0)),
            'slope':    int(request.form.get('slope', 1)),
            'ca':       int(request.form.get('ca', 0)),
            'thal':     int(request.form.get('thal', 3)),
        }

        session['patient_data'] = patient_data
        result = twin.predict(patient_data)
        session['last_result'] = result
        session['current_patient_name'] = patient_name

        add_consultation(current_user.id, patient_name, patient_data, result, created_by_role='doctor')

        return render_template('result.html',
                             patient=patient_data,
                             result=result,
                             patient_name=patient_name)

    except Exception as e:
        print(f"❌ Erreur predict: {e}")
        import traceback; traceback.print_exc()
        return render_template('error.html', error=str(e))


@app.route('/simulate', methods=['GET', 'POST'])
@login_required
@doctor_required
def simulate():
    if request.method == 'POST':
        try:
            base_patient = session.get('patient_data', None)
            if not base_patient:
                return render_template('error.html', error="Veuillez d'abord entrer un patient")

            modifications = {}
            for field, cast in [
                ('trestbps', float), ('chol', float), ('thalach', float),
                ('oldpeak', float), ('exang', int), ('cp', int),
                ('fbs', int), ('slope', int),
            ]:
                if field in request.form and request.form[field] != '':
                    modifications[field] = cast(request.form[field])

            result      = twin.simulate_parameter_change(base_patient, modifications)
            base_result = twin.predict(base_patient)
            improvement = base_result['probability'] - result['probability']

            return render_template('simulation.html',
                                 base_patient=base_patient,
                                 modifications=modifications,
                                 result=result,
                                 improvement=improvement,
                                 base_result=base_result,
                                 twin=twin)

        except Exception as e:
            return render_template('error.html', error=str(e))

    base_patient = session.get('patient_data', None)
    if not base_patient:
        return render_template('error.html', error="Veuillez d'abord entrer un patient")

    patients = get_doctor_patients(current_user.id)
    return render_template('simulation_form.html',
                         patient=base_patient,
                         patients_list=patients)


# ROUTES ECG


@app.route('/ecg')
@login_required
def ecg_page():
    patients = get_doctor_patients(current_user.id) if current_user.is_doctor() else []

    consultations = get_all_consultations()
    last_risk_data = {}
    for c in consultations:
        if c.get('type') == 'risk':
            patient_name = c.get('patient_name')
            if patient_name and patient_name not in last_risk_data:
                last_risk_data[patient_name] = {
                    'age': c.get('patient', {}).get('age'),
                    'ap_hi': c.get('patient', {}).get('ap_hi'),
                    'ap_lo': c.get('patient', {}).get('ap_lo'),
                    'cholesterol': c.get('patient', {}).get('cholesterol'),
                    'gluc': c.get('patient', {}).get('gluc'),
                    'date': c.get('date')
                }
    
    existing_risk = False
    if current_user.is_patient():
        history = get_patient_history(current_user.id)
        existing_risk = any(c.get('type') == 'risk' for c in history)
    
    return render_template('ecg.html', 
                         patients_list=patients,
                         last_risk_data=last_risk_data,
                         existing_risk=existing_risk)

@app.route('/analyze-ecg', methods=['POST'])
@login_required
def analyze_ecg():
    try:
        if ecg_classifier is None:
            return jsonify({'error': 'Modèle ECG non disponible'}), 503
        
        data = request.get_json()
        patient_name = data.get('patient_name')
        
        if not patient_name or patient_name == '':
            return jsonify({'error': 'Nom du patient requis'}), 400
        
        signal = np.array(data.get('signal', []))
        
        if len(signal) == 0:
            return jsonify({'error': 'Signal ECG vide'}), 400
        
        # ── NOUVEAU : accepter signal brut ou 187 points ──
        # Le classifieur gère automatiquement les deux formats
        result = ecg_classifier.predict(signal)
        
        if 'error' in result:
            return jsonify(result), 400
        
        add_ecg_consultation(
            current_user.id, 
            patient_name, 
            result, 
            created_by_role=current_user.role
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erreur analyze_ecg: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/ecg-sample')
@login_required
def ecg_sample():
    try:
        t = np.linspace(0, 2, 187)
        normal_signal = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(187)
        return jsonify({'signal': normal_signal.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ROUTES RADIO CARDIAQUE


@app.route('/xray')
@login_required
def xray_page():
    patients = get_doctor_patients(current_user.id) if current_user.is_doctor() else []
    
    existing_risk = False
    if current_user.is_patient():
        history = get_patient_history(current_user.id)
        existing_risk = any(c.get('type') == 'risk' for c in history)
    
    return render_template('xray.html', 
                         patients_list=patients,
                         existing_risk=existing_risk)

@app.route('/analyze-xray', methods=['POST'])
@login_required
def analyze_xray():
    try:
        if xray_model is None:
            return jsonify({'error': 'Modèle radio non disponible.'}), 503

        data = request.get_json()
        patient_name = data.get('patient_name', '').strip()

        if not patient_name:
            return jsonify({'error': 'Nom du patient requis'}), 400
        if 'image' not in data or not data['image']:
            return jsonify({'error': 'Image requise'}), 400

        img_bytes = b64.b64decode(data['image'])
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert('L')

        import numpy as np
        img_array = np.array(img).astype(np.float32)
        img_array = xrv.datasets.normalize(img_array, 255)
        img_array = img_array[None, ...]

        transform = xrv_transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        img_array = transform(img_array)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        with torch.no_grad():
            preds = xray_model(img_tensor)[0]

        scores = {}
        for i, pathology in enumerate(xray_model.pathologies):
            if pathology:
                scores[pathology] = round(float(preds[i]) * 100, 1)

        cardiac_score   = scores.get('Cardiomegaly', 0)
        effusion_score  = scores.get('Effusion', 0)
        edema_score     = scores.get('Edema', 0)

        normal = cardiac_score < 30 and effusion_score < 30 and edema_score < 30
        anomalies = {k: v for k, v in scores.items() if v > 30}

        main_scores = {
            'Cardiomegaly': scores.get('Cardiomegaly', 0),
            'Effusion':     scores.get('Effusion', 0),
            'Edema':        scores.get('Edema', 0),
            'Pneumonia':    scores.get('Pneumonia', 0),
            'Atelectasis':  scores.get('Atelectasis', 0),
            'No Finding':   round(100 - max(scores.values(), default=0), 1)
        }

        result = {
            'status': 'normal' if normal else 'anomalie_detectee',
            'scores': main_scores,
            'all_scores': scores,
            'anomalies': anomalies,
            'patient_name': patient_name,
            'date': datetime.now().isoformat()
        }

        add_xray_consultation(current_user.id, patient_name, result, created_by_role=current_user.role)
        return jsonify(result)

    except Exception as e:
        print(f" Erreur analyze_xray: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ROUTES DASHBOARD ET HISTORIQUE

@app.route('/doctor-dashboard')
@login_required
@doctor_required
def doctor_dashboard():
    consultations = get_all_consultations()

    consultations = [
        c for c in consultations
        if c.get('patient_name', '').strip() not in ['', 'Patient inconnu']
    ]

    consultations_by_patient = {}
    for c in consultations:
        patient_name = c.get('patient_name')
        if patient_name not in consultations_by_patient:
            consultations_by_patient[patient_name] = []
        consultations_by_patient[patient_name].append(c)

    patients_chart_data = []
    patients_list = []
    for patient_name, consults in consultations_by_patient.items():
        patients_list.append(patient_name)
        risk_consults = [c for c in consults if 'prediction' in c]
        patients_chart_data.append({
            'name': patient_name,
            'dates': [c['date'][:10] for c in risk_consults],
            'risks': [c['prediction']['probability'] * 100 for c in risk_consults]
        })

    risk_consultations = [c for c in consultations if 'prediction' in c]
    stats = {
        'total_consultations': len(consultations),
        'high_risk': len([c for c in risk_consultations if c.get('prediction', {}).get('probability', 0) > 0.6]),
        'total_ecg': len([c for c in consultations if c.get('type') == 'ecg']),
        'total_xray': len([c for c in consultations if c.get('type') == 'xray']),
        'unique_patients': len(consultations_by_patient)
    }

    return render_template('doctor_dashboard.html',
                         stats=stats,
                         consultations_by_patient=consultations_by_patient,
                         patients_list=patients_list,
                         patients_chart_data=patients_chart_data)

@app.route('/share-consultation/<int:consultation_id>', methods=['POST'])
@login_required
@patient_required
def share_consultation(consultation_id):
    try:
        data_req = request.get_json()
        doctor_id = data_req.get('doctor_id', '').strip()
        share = bool(data_req.get('share', True))

        if not doctor_id:
            return jsonify({'success': False, 'error': 'doctor_id requis'}), 400

        doctors = get_all_doctors()
        doctor = next((d for d in doctors if d['id'] == doctor_id), None)
        if not doctor:
            return jsonify({'success': False, 'error': 'Médecin introuvable'}), 404

        if not os.path.exists(HISTORY_FILE):
            return jsonify({'success': False, 'error': 'Historique introuvable'}), 404

        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            hist = json.load(f)

        consultations = hist.get('consultations', [])
        target = None
        for c in consultations:
            if c.get('id') == consultation_id:
                if c.get('patient_name', '').strip().lower() != current_user.name.strip().lower():
                    return jsonify({'success': False, 'error': 'Accès refusé'}), 403
                shared_list = c.get('shared_with_doctors', [])
                if share and doctor_id not in shared_list:
                    shared_list.append(doctor_id)
                elif not share and doctor_id in shared_list:
                    shared_list.remove(doctor_id)
                c['shared_with_doctors'] = shared_list
                c['shared_with_doctor'] = len(shared_list) > 0
                target = c
                break

        if not target:
            return jsonify({'success': False, 'error': 'Consultation introuvable'}), 404

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        return jsonify({
            'success': True,
            'shared_with_doctors': target['shared_with_doctors'],
            'doctor_name': doctor['name'],
            'share': share
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete-consultation/<int:consultation_id>', methods=['DELETE', 'POST'])
@login_required
@doctor_required
def delete_consultation(consultation_id):
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify({'success': False, 'error': 'Historique introuvable'}), 404

        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        consultations = data.get('consultations', [])
        original_count = len(consultations)
        consultations = [c for c in consultations if c.get('id') != consultation_id]

        if len(consultations) == original_count:
            return jsonify({'success': False, 'error': f'Consultation {consultation_id} introuvable'}), 404

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'deleted_id': consultation_id})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete-patient-consultations/<patient_name>', methods=['DELETE', 'POST'])
@login_required
@doctor_required
def delete_patient_consultations(patient_name):
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify({'success': False, 'error': 'Historique introuvable'}), 404

        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        consultations = data.get('consultations', [])
        original_count = len(consultations)
        consultations = [c for c in consultations if c.get('patient_name') != patient_name]
        deleted = original_count - len(consultations)

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'deleted': deleted})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/cleanup-invalid-consultations', methods=['POST'])
@login_required
@doctor_required
def cleanup_invalid_consultations():
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify({'success': True, 'deleted': 0})

        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        consultations = data.get('consultations', [])
        original_count = len(consultations)
        consultations = [
            c for c in consultations
            if c.get('patient_name', '').strip() not in ['', 'Patient inconnu']
        ]

        for i, c in enumerate(consultations, 1):
            c['id'] = i

        deleted = original_count - len(consultations)

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({'consultations': consultations}, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True, 'deleted': deleted})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/patient-history/<patient_name>')
@login_required
@doctor_required
def patient_history(patient_name):
    consultations = get_all_consultations()
    history = [c for c in consultations if c.get('patient_name') == patient_name and 'prediction' in c]
    history = sorted(history, key=lambda x: x['date'])

    if not history:
        return render_template('error.html', error=f"Aucun historique trouvé pour {patient_name}")

    risks = [c['prediction']['probability'] for c in history]
    avg_risk = sum(risks) / len(risks)
    trend = risks[-1] - risks[0] if len(risks) > 1 else 0
    high_risk_episodes = sum(1 for r in risks if r > 0.6)
    first_visit = history[0]['date'][:10]

    last = history[-1]
    patient = last.get('patient', {})
    patient['id'] = patient_name
    patient['gender'] = patient.get('gender', 2)

    return render_template('patient_history.html',
                         patient=patient,
                         patient_name=patient_name,
                         history=history,
                         first_visit=first_visit,
                         avg_risk=avg_risk,
                         trend=trend,
                         high_risk_episodes=high_risk_episodes)


# ROUTES EXPLICATION ET PRÉDICTION

@app.route('/explain-prediction', methods=['POST'])
@login_required
def explain_prediction():
    patient_data = session.get('patient_data')
    if not patient_data:
        return jsonify({'error': 'No patient data'})
    
    try:
        df = twin.prepare_patient_data(patient_data)
        explanation = explainer.explain_prediction(df.values)
        return jsonify(explanation)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-evolution', methods=['POST'])
@login_required
def predict_evolution():
    history = session.get('patient_history', [])
    prediction = temporal.predict_risk_trajectory(history)
    return jsonify(prediction)

@app.route('/get-recommendations', methods=['POST'])
@login_required
def get_recommendations():
    patient_data = session.get('patient_data')
    result = session.get('last_result')
    
    if not patient_data or not result:
        return jsonify({'error': 'No data'})
    
    try:
        recommendations = recommender.get_recommendations(patient_data, result)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ROUTES RAPPORTS PDF

@app.route('/generate-report', methods=['POST'])
@login_required
def generate_report():
    patient_data = session.get('patient_data')
    result = session.get('last_result')
    
    if not patient_data or not result:
        return jsonify({'error': 'No data'}), 400
    
    try:
        recommendations = recommender.get_recommendations(patient_data, result)
        filename = f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = reporter.generate_pdf(patient_data, result, recommendations, filename)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate-simulation-report', methods=['POST'])
@login_required
def generate_simulation_report():
    try:
        data = request.get_json()
        
        reports_dir = os.path.join('static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(reports_dir, filename)
        
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4
        
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Rapport de Simulation - Digital Twin")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 120, "Résultats de la simulation")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 150, f"Risque initial: {data['base_result']['probability']*100:.1f}%")
        c.drawString(50, height - 170, f"Risque après simulation: {data['result']['probability']*100:.1f}%")
        
        if data['improvement'] > 0:
            c.drawString(50, height - 190, f"Réduction du risque: {data['improvement']*100:.1f}%")
        else:
            c.drawString(50, height - 190, "Pas de changement significatif")
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 230, "Modifications appliquées:")
        
        c.setFont("Helvetica", 12)
        y = height - 250
        modifications = data.get('modifications', {})
        base_patient = data.get('base_patient', {})
        
        for key, value in modifications.items():
            if key == 'ap_hi':
                param_name = "Pression systolique"
                old_value = base_patient.get(key, 'N/A')
            elif key == 'weight':
                param_name = "Poids"
                old_value = base_patient.get(key, 'N/A')
            elif key == 'cholesterol':
                param_name = "Cholestérol"
                old_value = base_patient.get(key, 'N/A')
                if value == 1: value = "Normal"
                elif value == 2: value = "Élevé"
                else: value = "Très élevé"
            elif key == 'smoke':
                param_name = "Tabagisme"
                old_value = "Oui" if base_patient.get(key) == 1 else "Non"
                value = "Oui" if value == 1 else "Non"
            elif key == 'active':
                param_name = "Activité physique"
                old_value = "Oui" if base_patient.get(key) == 1 else "Non"
                value = "Oui" if value == 1 else "Non"
            else:
                param_name = key
                old_value = base_patient.get(key, 'N/A')
            
            c.drawString(70, y, f"• {param_name}: {old_value} → {value}")
            y -= 20
        
        c.setFont("Helvetica", 9)
        c.drawString(50, 50, "Document généré par Digital Twin - Système d'aide à la décision cardiovasculaire")
        c.drawString(50, 35, "Cet outil ne remplace pas un avis médical professionnel.")
        
        c.save()
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# API REST

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        data = request.get_json()
        result = twin.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ecg', methods=['POST'])
@login_required
def api_ecg():
    try:
        if ecg_classifier is None:
            return jsonify({'error': 'Modèle ECG non disponible'}), 503
        
        data = request.get_json()
        signal = np.array(data.get('signal', []))
        
        if len(signal) == 0:
            return jsonify({'error': 'Signal ECG vide'}), 400
        
        # Accepte signal brut ou 187 points
        result = ecg_classifier.predict(signal)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/api/xray', methods=['POST'])
@login_required
def api_xray():
    try:
        if xray_model is None:
            return jsonify({'error': 'Modèle radio non disponible'}), 503

        data = request.get_json()
        img_bytes = b64.b64decode(data['image'])
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img_array = np.array(img).astype(np.float32)
        img_array = xrv.datasets.normalize(img_array, 255)
        img_array = img_array[None, ...]
        transform = xrv_transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        img_array = transform(img_array)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        with torch.no_grad():
            preds = xray_model(img_tensor)[0]

        scores = {xray_model.pathologies[i]: round(float(preds[i]) * 100, 1)
                  for i in range(len(xray_model.pathologies)) if xray_model.pathologies[i]}

        cardiac_score = scores.get('Cardiomegaly', 0)
        normal = cardiac_score < 30
        anomalies = {k: v for k, v in scores.items() if v > 30}

        return jsonify({
            'status': 'normal' if normal else 'anomalie_detectee',
            'scores': scores,
            'anomalies': anomalies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/generate-full-report/<patient_name>')
@login_required
def generate_full_report(patient_name):
    try:
        if current_user.is_doctor():
            all_consults = get_all_consultations()
        else:
            all_consults = get_patient_history(current_user.id)

        consultations = [
            c for c in all_consults
            if c.get('patient_name', '').strip().lower() == patient_name.strip().lower()
        ]

        if not consultations:
            return render_template('error.html', error=f"Aucune donnée trouvée pour {patient_name}")

        filename = f"rapport_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = reporter.generate_full_report(patient_name, consultations, filename)

        return send_file(filepath, as_attachment=True, download_name=filename)

    except Exception as e:
        print(f"❌ Erreur rapport complet: {e}")
        import traceback; traceback.print_exc()
        return render_template('error.html', error=str(e))

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    success = None
    error = None

    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not current_user.check_password(current_password):
            error = 'Mot de passe actuel incorrect.'
        elif len(new_password) < 6:
            error = 'Le nouveau mot de passe doit contenir au moins 6 caractères.'
        elif new_password != confirm_password:
            error = 'Les mots de passe ne correspondent pas.'
        else:
            current_user.set_password(new_password)
            current_user.save()
            success = 'Mot de passe changé avec succès !'

    return render_template('change_password.html', success=success, error=error)

# LANCEMENT DE L'APPLICATION

if __name__ == '__main__':
    os.makedirs('static/reports', exist_ok=True)
    
    if not os.path.exists(PATIENTS_FILE):
        save_patients([])
    
    if not os.path.exists(USERS_FILE):
        default_doctor = User(
            id="doc_001",
            email="doctor@test.com",
            password_hash="",
            role="doctor",
            name="Dr. Martin"
        )
        default_doctor.set_password("password")
        default_doctor.save()
        print("✅ Compte médecin par défaut créé: doctor@test.com / password")
    
    app.run(debug=True, host='0.0.0.0', port=5000)