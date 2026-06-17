"""
Cardio Digital Twin — CardiacTwin
Modèle : LR Optuna (CTGAN + SMOTE + Optuna)
Dataset : 4 bases UCI Heart Disease
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class CardiacTwin:
    """
    Jumeau numérique cardiaque basé sur
    Logistic Regression optimisée (Optuna) + seuil clinique.
    """

    ALL_FEATURES = [
        # Features originales UCI
        'age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak','slope','ca','thal',
        # Features engineerées
        'hr_reserve','hr_age_ratio','st_hr_index',
        'age_thalach','age_oldpeak','chol_age','bp_chol',
        'ecg_risk','global_risk','age_group',
        'typical_angina','low_thalach','high_chol','hypertension',
    ]

    CONTINUOUS_COLS = [
        'age','trestbps','chol','thalach','oldpeak',
        'hr_reserve','hr_age_ratio','st_hr_index',
        'age_thalach','age_oldpeak','chol_age','bp_chol'
    ]

    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        self.model     = None
        self.scaler    = None
        self.threshold = 0.5
        self.all_features = self.ALL_FEATURES

        self._load()

    def _load(self):
        """Charge le modèle LR Optuna et ses artefacts."""
        model_path     = os.path.join(self.model_dir, 'lr_model.pkl')
        scaler_path    = os.path.join(self.model_dir, 'scaler_lr.pkl')
        threshold_path = os.path.join(self.model_dir, 'threshold_lr.pkl')
        features_path  = os.path.join(self.model_dir, 'feature_names_lr.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                f"Lancez d'abord : python train_lr_optuna.py"
            )

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(threshold_path, 'rb') as f:
            self.threshold = pickle.load(f)
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                self.all_features = pickle.load(f)

        print(f"✅ LR Optuna chargé — seuil clinique : {self.threshold:.4f}")

    def _feature_engineering(self, patient_data: dict) -> pd.DataFrame:
        """Calcule toutes les features engineerées depuis les données brutes."""
        p = patient_data

        age      = float(p.get('age', 50))
        thalach  = float(p.get('thalach', 150))
        oldpeak  = float(p.get('oldpeak', 1.0))
        trestbps = float(p.get('trestbps', p.get('ap_hi', 120)))
        chol     = float(p.get('chol', 200))
        restecg  = float(p.get('restecg', 0))
        exang    = float(p.get('exang', 0))
        fbs      = float(p.get('fbs', 0))
        cp       = float(p.get('cp', 0))

        row = {
            # Originales UCI
            'age':      age,
            'sex':      float(p.get('sex', p.get('gender', 1))),
            'cp':       cp,
            'trestbps': trestbps,
            'chol':     chol,
            'fbs':      fbs,
            'restecg':  restecg,
            'thalach':  thalach,
            'exang':    exang,
            'oldpeak':  oldpeak,
            'slope':    float(p.get('slope', 1)),
            'ca':       float(p.get('ca', 0)),
            'thal':     float(p.get('thal', 2)),

            # Feature Engineering (identique au notebook)
            'hr_reserve':   thalach - 60,
            'hr_age_ratio': thalach / age if age > 0 else 0,
            'st_hr_index':  oldpeak / (thalach + 1),
            'age_thalach':  age * thalach,
            'age_oldpeak':  age * oldpeak,
            'chol_age':     chol * age,
            'bp_chol':      trestbps * chol,

            'ecg_risk': int(oldpeak > 1.0) + int(restecg > 0) + int(exang),
            'global_risk': (
                int(age > 55) + int(trestbps > 130) + int(chol > 240) +
                int(fbs == 1) + int(exang) + int(oldpeak > 1)
            ),

            'age_group':      int(pd.cut([age], bins=[0,40,50,55,60,100], labels=[0,1,2,3,4])[0]),
            'typical_angina': int(cp == 4),
            'low_thalach':    int(thalach < 120),
            'high_chol':      int(chol > 240),
            'hypertension':   int(trestbps > 140),
        }

        df = pd.DataFrame([row])

        # Garder uniquement les features du modèle, dans le bon ordre
        cols_present = [c for c in self.all_features if c in df.columns]
        df = df[cols_present]
        return df

    def _adapt_input(self, patient_data: dict) -> dict:
        """
        Adapte les données de l'interface Flask (qui utilise ap_hi/ap_lo)
        vers le format UCI (trestbps) + valeurs par défaut raisonnables.
        """
        p = dict(patient_data)

        # Tension : ap_hi → trestbps
        if 'ap_hi' in p and 'trestbps' not in p:
            p['trestbps'] = p['ap_hi']

        # Cholestérol : code 1/2/3 → valeur mg/dL approximative
        if 'cholesterol' in p and 'chol' not in p:
            chol_map = {1: 180, 2: 220, 3: 280}
            p['chol'] = chol_map.get(int(p['cholesterol']), 200)

        # Genre : gender 2=homme/1=femme → sex 1=homme/0=femme (UCI)
        if 'gender' in p and 'sex' not in p:
            p['sex'] = 1 if int(p['gender']) == 2 else 0

        # Valeurs par défaut UCI si absentes
        defaults = {
            'cp': 0, 'fbs': 0, 'restecg': 0,
            'thalach': max(60, 220 - int(p.get('age', 50))),
            'exang': int(p.get('smoke', 0)),
            'oldpeak': 1.0 if int(p.get('ap_hi', 120)) > 140 else 0.5,
            'slope': 1, 'ca': 0, 'thal': 2,
        }
        for k, v in defaults.items():
            if k not in p:
                p[k] = v

        return p

    def predict(self, patient_data: dict) -> dict:
        """
        Prédit le risque cardiovasculaire.
        Retourne : probability, prediction, risk_level
        """
        p = self._adapt_input(patient_data)
        df = self._feature_engineering(p)

        # Scaling
        cont_present = [c for c in self.CONTINUOUS_COLS if c in df.columns]
        df_scaled = df.copy()
        df_scaled[cont_present] = self.scaler.transform(df[cont_present])

        proba = float(self.model.predict_proba(df_scaled)[0, 1])
        pred  = int(proba >= self.threshold)

        if proba < 0.3:
            risk_level = 'Faible'
        elif proba < 0.6:
            risk_level = 'Modéré'
        else:
            risk_level = 'Élevé'

        return {
            'probability': proba,
            'prediction':  pred,
            'risk_level':  risk_level,
            'threshold':   self.threshold,
        }

    def prepare_patient_data(self, patient_data: dict) -> pd.DataFrame:
        """Pour ExplainableTwin (SHAP)."""
        p = self._adapt_input(patient_data)
        df = self._feature_engineering(p)
        cont_present = [c for c in self.CONTINUOUS_COLS if c in df.columns]
        df_scaled = df.copy()
        df_scaled[cont_present] = self.scaler.transform(df[cont_present])
        return df_scaled

    def simulate_parameter_change(self, base_patient: dict, modifications: dict) -> dict:
        """Simulation what-if."""
        modified = dict(base_patient)
        modified.update(modifications)
        return self.predict(modified)
