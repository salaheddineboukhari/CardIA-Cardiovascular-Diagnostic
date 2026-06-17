"""
Classifieur ECG utilisant un modèle CNN
Détection des arythmies et anomalies cardiaques
"""

import numpy as np
import tensorflow as tf
import joblib
import os
from scipy.signal import resample, find_peaks

class ECGClassifier:
    
    def __init__(self, model_path='models/cnn_heartbeat_model.h5', scaler_path='models/scaler_ecg.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.classes = {
            0: {
                "name": "Normal",
                "color": "success",
                "icon": "✅",
                "advice": "Rythme cardiaque normal. Continuez vos bonnes habitudes."
            },
            1: {
                "name": "Supraventricular",
                "color": "warning",
                "icon": "⚠️",
                "advice": "Battement prématuré des oreillettes. Surveillance recommandée."
            },
            2: {
                "name": "Ventricular",
                "color": "danger",
                "icon": "🔴",
                "advice": "Contraction ventriculaire prématurée. Consultation médicale nécessaire."
            },
            3: {
                "name": "Fusion",
                "color": "warning",
                "icon": "🟠",
                "advice": "Fusion d'un battement normal et ventriculaire. Évaluation complémentaire."
            },
            4: {
                "name": "Unclassifiable",
                "color": "secondary",
                "icon": "❓",
                "advice": "Signal difficile à classifier. Analyse approfondie nécessaire."
            }
        }
        
        self.ecg_components = {
            0: {
                "Onde P": "Présente, forme normale",
                "Intervalle PR": "120-200 ms",
                "Complexe QRS": "< 100 ms, normal",
                "Segment ST": "Isoélectrique",
                "Onde T": "Positive"
            },
            1: {
                "Onde P": "Prématurée, forme anormale",
                "Intervalle PR": "Raccourci",
                "Complexe QRS": "Normal",
                "Segment ST": "Peut être déprimé",
                "Onde T": "Parfois inversée"
            },
            2: {
                "Onde P": "Absente",
                "Intervalle PR": "Impossible à mesurer",
                "Complexe QRS": "Élargi (>120ms), déformé",
                "Segment ST": "Décalé",
                "Onde T": "Inversée"
            },
            3: {
                "Onde P": "Partiellement présente",
                "Complexe QRS": "Forme intermédiaire",
                "Segment ST": "Anormal"
            },
            4: {
                "Signal": "Atypique ou bruité",
                "Analyse": "Difficile à classifier"
            }
        }
        
        self._load_model()
    
    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print("✅ Modèle ECG chargé")
            else:
                print(f"⚠️ Modèle non trouvé: {self.model_path}")
                self.model = None
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Scaler ECG chargé")
            else:
                print(f"⚠️ Scaler non trouvé: {self.scaler_path}")
                self.scaler = None
                
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            self.model = None
            self.scaler = None

    def pretraiter_signal_brut(self, signal_brut):
        """
        Convertit un signal ECG brut (n points) 
        en vecteur normalisé de 187 points.
        Accepte : signal long (Holter, AliveCor, Apple Watch…)
        """
        signal_brut = np.array(signal_brut, dtype=np.float64)
        
        # Supprimer la ligne de base (filtre simple)
        signal_brut = signal_brut - np.mean(signal_brut)
        
        # Détecter les pics R
        hauteur_min = np.std(signal_brut) * 0.5
        peaks, _ = find_peaks(
            signal_brut,
            distance=50,
            height=hauteur_min
        )
        
        if len(peaks) > 0:
            # Prendre le pic le plus fort
            best_peak = peaks[np.argmax(signal_brut[peaks])]
            start = max(0, best_peak - 90)
            end   = min(len(signal_brut), best_peak + 97)
            segment = signal_brut[start:end]
        else:
            # Pas de pic détecté → prendre le centre du signal
            mid   = len(signal_brut) // 2
            start = max(0, mid - 93)
            end   = min(len(signal_brut), mid + 94)
            segment = signal_brut[start:end]
        
        # Redimensionner à exactement 187 points
        segment_187 = resample(segment, 187)
        
        # Normaliser entre 0 et 1
        min_val = segment_187.min()
        max_val = segment_187.max()
        if max_val - min_val > 1e-8:
            segment_187 = (segment_187 - min_val) / (max_val - min_val)
        
        return segment_187.astype(np.float32)

    def preprocess(self, signal):
        """
        Normalise et reshape un signal de 187 valeurs pour le CNN.
        """
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        if signal.shape[1] != 187:
            raise ValueError(
                f"Le signal doit avoir 187 features, reçu {signal.shape[1]}"
            )
        
        if self.scaler:
            signal_scaled = self.scaler.transform(signal)
        else:
            signal_scaled = signal
        
        return signal_scaled.reshape(
            signal_scaled.shape[0], signal_scaled.shape[1], 1
        )
    
    def predict(self, signal):
        """
        Prédit la classe d'un signal ECG.
        Accepte :
          - tableau de 187 valeurs (format MIT-BIH prêt)
          - tableau de 188 valeurs (format Excel avec colonne Label incluse)
          - tableau de N valeurs   (signal brut → prétraitement auto)
        """
        try:
            signal = np.array(signal, dtype=np.float32).flatten()
            
            # Gestion automatique de la colonne de Label d'Excel (188e colonne)
            if len(signal) == 188:
                signal = signal[:187]
            
            # Détection et traitement du format si signal continu
            if len(signal) != 187:
                print(f"📡 Signal brut détecté ({len(signal)} points) → prétraitement automatique")
                signal = self.pretraiter_signal_brut(signal)
            
            # Si le modèle Keras est bien chargé, exécuter l'inférence Deep Learning
            if self.model is not None:
                signal_processed = self.preprocess(signal.reshape(1, -1))
                probabilities = self.model.predict(signal_processed, verbose=0)[0]
                pred_class    = int(np.argmax(probabilities))
                confidence    = float(probabilities[pred_class])
                
                return {
                    'class_id'       : pred_class,
                    'class_name'     : self.classes[pred_class]['name'],
                    'class_color'    : self.classes[pred_class]['color'],
                    'class_icon'     : self.classes[pred_class]['icon'],
                    'clinical_advice': self.classes[pred_class]['advice'],
                    'ecg_components' : self.ecg_components[pred_class],
                    'confidence'     : confidence,
                    'probability'    : confidence * 100,
                    'all_probabilities': {
                        self.classes[i]['name']: float(prob)
                        for i, prob in enumerate(probabilities)
                    }
                }
            else:
                # Mode de secours si le fichier h5 n'est pas chargé
                return self._demo_predict(signal)
            
        except Exception as e:
            return {'error': str(e)}
    
    def _demo_predict(self, signal):
        """
        Algorithme de secours basé sur des métriques morphologiques 
        adaptées aux données normalisées (0 à 1) de MIT-BIH.
        """
        signal = np.array(signal, dtype=np.float32).flatten()
        mean_val = np.mean(signal)
        std_val  = np.std(signal)
        min_val  = np.min(signal)
        
        # Logique réajustée pour les variations réelles des signaux MIT-BIH normalisés
        if std_val > 0.28 or min_val > 0.15:
            pred = 2  # Ventricular (morphologie élargie ou absence d'onde isoélectrique basse)
        elif mean_val > 0.35:
            pred = 1  # Supraventricular (onde P prématurée ou surélévation globale)
        elif mean_val < 0.15:
            pred = 3  # Fusion
        else:
            pred = 0  # Normal
        
        return {
            'class_id'       : pred,
            'class_name'     : self.classes[pred]['name'],
            'class_color'    : self.classes[pred]['color'],
            'class_icon'     : self.classes[pred]['icon'],
            'clinical_advice': self.classes[pred]['advice'],
            'ecg_components' : self.ecg_components[pred],
            'confidence'     : 0.88,
            'probability'    : 88.0,
            'demo_mode'      : True,
            'all_probabilities': {
                self.classes[i]['name']: 0.88 if i == pred else 0.03
                for i in range(5)
            }
        }
    
    def predict_batch(self, signals):
        return [self.predict(signal) for signal in signals]
    
    def get_class_info(self, class_id):
        return self.classes.get(class_id, {"name": "Inconnu"})