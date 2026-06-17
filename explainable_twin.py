import shap
import numpy as np
import pandas as pd


class ExplainableTwin:
    """Explicabilité SHAP pour Logistic Regression (variables UCI)"""

    # Labels lisibles pour les variables UCI
    FEATURE_LABELS = {
        'age':          'Âge',
        'sex':          'Sexe',
        'cp':           'Type douleur thoracique',
        'trestbps':     'Tension artérielle au repos',
        'chol':         'Cholestérol',
        'fbs':          'Glycémie à jeun > 120',
        'restecg':      'ECG au repos',
        'thalach':      'Fréquence cardiaque max',
        'exang':        'Angine d\'effort',
        'oldpeak':      'Dépression ST',
        'slope':        'Pente segment ST',
        'ca':           'Vaisseaux colorés',
        'thal':         'Thalassémie',
        'hr_reserve':   'Réserve cardiaque',
        'hr_age_ratio': 'Ratio FC/Âge',
        'st_hr_index':  'Index ST/FC',
        'age_thalach':  'Interaction Âge×FC',
        'age_oldpeak':  'Interaction Âge×ST',
        'chol_age':     'Interaction Chol×Âge',
        'bp_chol':      'Interaction TA×Chol',
        'ecg_risk':     'Score risque ECG',
        'global_risk':  'Score risque global',
        'age_group':    'Groupe d\'âge',
        'typical_angina': 'Angine typique',
        'low_thalach':  'FC max faible',
        'high_chol':    'Cholestérol élevé',
        'hypertension': 'Hypertension',
    }

    def __init__(self, model, feature_names, background_data=None):
        self.model        = model
        self.feature_names = list(feature_names)
        self.explainer    = None
        self._init_explainer(background_data)

    def _init_explainer(self, background_data=None):
        """Initialise LinearExplainer pour LR."""
        try:
            if background_data is not None:
                masker = shap.maskers.Independent(background_data)
            else:
                # Masker par défaut avec données nulles
                n_features = len(self.feature_names)
                bg = np.zeros((1, n_features))
                masker = shap.maskers.Independent(bg)

            self.explainer = shap.LinearExplainer(self.model, masker=masker)
            print("✅ SHAP LinearExplainer initialisé")
        except Exception as e:
            print(f"⚠️ LinearExplainer failed, trying Explainer: {e}")
            try:
                self.explainer = shap.Explainer(self.model)
                print("✅ SHAP Explainer générique initialisé")
            except Exception as e2:
                print(f"⚠️ SHAP non disponible: {e2}")
                self.explainer = None

    def explain_prediction(self, patient_data):
        """
        Explique la prédiction pour un patient.
        patient_data : array numpy (1, n_features) déjà scalé
        """
        if self.explainer is None:
            return self._fallback_explanation(patient_data)

        try:
            shap_values = self.explainer.shap_values(patient_data)

            # LR binaire → shap_values peut être [neg, pos] ou directement pos
            if isinstance(shap_values, list):
                sv = np.array(shap_values[1][0])  # classe positive
            else:
                sv = np.array(shap_values[0]) if shap_values.ndim > 1 else np.array(shap_values)

            # Importance par feature
            feature_importance = {}
            for i, feat in enumerate(self.feature_names):
                if i < len(sv):
                    feature_importance[feat] = float(abs(sv[i]))

            top_features = sorted(feature_importance.items(),
                                  key=lambda x: x[1], reverse=True)[:8]

            # Labels lisibles
            top_features_labeled = [
                (self.FEATURE_LABELS.get(f, f), imp)
                for f, imp in top_features
            ]

            # Extraire les valeurs patient
            if hasattr(patient_data, 'iloc'):
                row = patient_data.iloc[0].to_dict()
            elif hasattr(patient_data, 'tolist'):
                row = {self.feature_names[i]: float(patient_data[0][i])
                       for i in range(min(len(self.feature_names), patient_data.shape[1]))}
            else:
                row = {}

            # Explications en français
            explanations = self._generate_explanations(top_features[:5], row)

            return {
                'top_features':  top_features_labeled,
                'feature_names': [self.FEATURE_LABELS.get(f, f) for f, _ in top_features],
                'shap_values':   [float(v) for _, v in top_features_labeled],
                'explanations':  explanations,
            }

        except Exception as e:
            print(f"⚠️ SHAP explain error: {e}")
            return self._fallback_explanation(patient_data)

    def _generate_explanations(self, top_features, row):
        """Génère des explications cliniques en français."""
        explanations = []
        for feature, importance in top_features:
            val = row.get(feature, None)
            label = self.FEATURE_LABELS.get(feature, feature)

            if feature == 'trestbps' and val is not None:
                if val > 140:
                    explanations.append(f"{label} élevée ({val:.0f} mmHg) — facteur de risque important")
                else:
                    explanations.append(f"{label} normale ({val:.0f} mmHg) — facteur protecteur")

            elif feature == 'chol' and val is not None:
                if val > 240:
                    explanations.append(f"{label} très élevé ({val:.0f} mg/dL) — risque significatif")
                elif val > 200:
                    explanations.append(f"{label} élevé ({val:.0f} mg/dL) — à surveiller")
                else:
                    explanations.append(f"{label} normal ({val:.0f} mg/dL)")

            elif feature == 'age' and val is not None:
                if val > 60:
                    explanations.append(f"Âge ({val:.0f} ans) — facteur de risque non modifiable")
                elif val > 50:
                    explanations.append(f"Âge ({val:.0f} ans) — surveillance recommandée")

            elif feature == 'thalach' and val is not None:
                if val < 120:
                    explanations.append(f"{label} faible ({val:.0f} bpm) — capacité cardiaque réduite")
                else:
                    explanations.append(f"{label} correcte ({val:.0f} bpm)")

            elif feature == 'oldpeak' and val is not None:
                if val > 2:
                    explanations.append(f"{label} élevée ({val:.1f}) — ischémie possible")
                elif val > 0:
                    explanations.append(f"{label} modérée ({val:.1f})")

            elif feature == 'exang' and val is not None:
                if val == 1:
                    explanations.append(f"Angine d'effort présente — signe d'ischémie")
                else:
                    explanations.append(f"Pas d'angine d'effort — facteur favorable")

            elif feature == 'cp' and val is not None:
                cp_labels = {1: 'typique', 2: 'atypique', 3: 'non angineuse', 4: 'asymptomatique'}
                explanations.append(f"Douleur thoracique {cp_labels.get(int(val), '')} (type {int(val)})")

            elif feature in ('ecg_risk', 'global_risk') and val is not None:
                if val > 2:
                    explanations.append(f"{label} élevé ({int(val)}/5) — multiple facteurs de risque")

            elif importance > 0.05:
                explanations.append(f"{label} — contribution significative au risque")

        return explanations[:5]

    def _fallback_explanation(self, patient_data):
        """Explication de secours sans SHAP."""
        return {
            'top_features':  [(label, 0.1) for label in list(self.FEATURE_LABELS.values())[:5]],
            'feature_names': list(self.FEATURE_LABELS.values())[:5],
            'shap_values':   [0.1] * 5,
            'explanations':  ['Analyse SHAP non disponible pour ce modèle.'],
        }