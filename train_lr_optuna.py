"""

Cardio Digital Twin - Réentraînement LR Optuna
Pipeline : 4 bases UCI → Feature Engineering → CTGAN →
           QuantileTransformer → SMOTE → Optuna → LR clinique
Utilisation :
    python train_lr_optuna.py 
        --cleveland  cleveland.data 
        --hungarian  hungarian.data 
        --switzerland switzerland.data 
        --longbeach  long-beach-va.data

Sorties dans models/ :
    lr_model.pkl       - modèle LR Optuna entraîné
    scaler_lr.pkl      - QuantileTransformer
    threshold_lr.pkl   - seuil clinique optimal
    feature_names.pkl  - liste des features dans l'ordre
"""

import os, re, io, warnings, argparse, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score
)
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# 1. CHARGEMENT DES 4 BASES UCI
COL_NAMES = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak',
    'slope','ca','thal','target'
]
COLS_76   = [0,1,2,3,4,5,6,10,11,15,18,31,37,57]
ENCODINGS = ['utf-8','latin-1','iso-8859-1','cp1252']

def parse_uci_file(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()

    text = None
    for enc in ENCODINGS:
        try:
            text = content.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        raise ValueError(f'Impossible de décoder {filepath}')

    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith('--')
             and not l.strip().startswith('#')]
    full_text = ' '.join(lines)

    # Format 76 colonnes
    if re.search(r'\bname\b', full_text, re.IGNORECASE):
        patients_raw = re.split(r'\bname\b', full_text, flags=re.IGNORECASE)
        rows = []
        for pat in patients_raw:
            nums = re.findall(r'-?\d+\.?\d*', pat)
            if not nums: continue
            if len(nums) >= 58:
                try: row = [nums[i] for i in COLS_76]
                except IndexError: row = nums[:14]
            elif len(nums) >= 14:
                row = nums[:14]
            else: continue
            rows.append(row)
        if rows:
            df_tmp = pd.DataFrame(rows, columns=COL_NAMES)
            df_tmp = df_tmp.replace({'-9': np.nan, '-9.0': np.nan})
            for col in COL_NAMES:
                df_tmp[col] = pd.to_numeric(df_tmp[col], errors='coerce')
            df_tmp = df_tmp.replace({-9: np.nan, -9.0: np.nan})
            return df_tmp

    # Format CSV 14 colonnes
    for enc in ENCODINGS:
        try:
            df_tmp = pd.read_csv(
                io.BytesIO(content), header=None, names=COL_NAMES,
                na_values=['?','-9','-9.0'], encoding=enc, on_bad_lines='skip'
            )
            if df_tmp.shape[1] > 14:
                df_tmp = df_tmp.iloc[:,:14]
                df_tmp.columns = COL_NAMES
            return df_tmp
        except Exception:
            continue
    raise ValueError(f'Format non reconnu pour {filepath}')


def load_data(files_dict):
    dfs = []
    for name, path in files_dict.items():
        if path and os.path.exists(path):
            df_tmp = parse_uci_file(path)
            print(f'  ✅ {name}: {len(df_tmp)} patients')
            dfs.append(df_tmp)
        else:
            print(f'  ⚠️  {name}: fichier non trouvé ({path})')
    if not dfs:
        raise ValueError('Aucun fichier chargé !')
    return pd.concat(dfs, ignore_index=True)


# 2. NETTOYAGE & FEATURE ENGINEERING
def clean_and_engineer(df):
    # Cible binaire
    df['target'] = (df['target'] > 0).astype(int)
    if 'source' in df.columns:
        df.drop(columns=['source'], inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace({-9: np.nan, -9.0: np.nan}, inplace=True)
    df = df[df.isnull().sum(axis=1) <= 6]

    # Filtres physiologiques
    df = df[
        (df['age'].isna()      | df['age'].between(10, 90))       &
        (df['trestbps'].isna() | df['trestbps'].between(50, 260)) &
        (df['chol'].isna()     | df['chol'].between(50, 750))     &
        (df['thalach'].isna()  | df['thalach'].between(40, 250))  &
        (df['oldpeak'].isna()  | df['oldpeak'].between(-2, 10))
    ]
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Imputation KNN
    cat_cols  = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    cont_cols = ['trestbps','chol','thalach','oldpeak']
    for col in cat_cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)

    imputer = KNNImputer(n_neighbors=5)
    df[cont_cols] = imputer.fit_transform(df[cont_cols])
    for col in cat_cols:
        df[col] = df[col].round().astype(int)

    # Feature Engineering
    df['hr_reserve']   = df['thalach'] - 60
    df['hr_age_ratio'] = df['thalach'] / df['age']
    df['st_hr_index']  = df['oldpeak'] / (df['thalach'] + 1)
    df['age_thalach']  = df['age'] * df['thalach']
    df['age_oldpeak']  = df['age'] * df['oldpeak']
    df['chol_age']     = df['chol'] * df['age']
    df['bp_chol']      = df['trestbps'] * df['chol']

    df['ecg_risk'] = (
        (df['oldpeak'] > 1.0).astype(int) +
        (df['restecg'] > 0).astype(int)   +
        df['exang']
    )
    df['global_risk'] = (
        (df['age'] > 55).astype(int)       +
        (df['trestbps'] > 130).astype(int) +
        (df['chol'] > 240).astype(int)     +
        (df['fbs'] == 1).astype(int)       +
        df['exang']                         +
        (df['oldpeak'] > 1).astype(int)
    )

    df['age_group']      = pd.cut(df['age'], bins=[0,40,50,55,60,100],
                                   labels=[0,1,2,3,4]).astype(int)
    df['typical_angina'] = (df['cp'] == 4).astype(int)
    df['low_thalach']    = (df['thalach'] < 120).astype(int)
    df['high_chol']      = (df['chol'] > 240).astype(int)
    df['hypertension']   = (df['trestbps'] > 140).astype(int)

    print(f'   Dataset nettoyé : {df.shape} | Balance : {df["target"].value_counts().to_dict()}')
    return df


# 3. CTGAN 
def augment_with_ctgan(df_train_orig, n_synthetic=300):
    try:
        from ctgan import CTGAN
        print('   CTGAN disponible - augmentation...')
        discrete_columns = [
            'sex','cp','fbs','restecg','exang','slope','ca','thal',
            'ecg_risk','global_risk','age_group',
            'typical_angina','low_thalach','high_chol','hypertension','target'
        ]
        df_sains   = df_train_orig[df_train_orig['target'] == 0].reset_index(drop=True)
        df_malades = df_train_orig[df_train_orig['target'] == 1].reset_index(drop=True)

        n_sain   = max(50, n_synthetic // 2)
        n_malade = max(50, n_synthetic // 2)

        ctgan_sain = CTGAN(epochs=200, batch_size=50, verbose=False)
        ctgan_sain.fit(df_sains, discrete_columns)
        synth_sains = ctgan_sain.sample(n_sain)
        synth_sains['target'] = 0

        ctgan_malade = CTGAN(epochs=200, batch_size=50, verbose=False)
        ctgan_malade.fit(df_malades, discrete_columns)
        synth_malades = ctgan_malade.sample(n_malade)
        synth_malades['target'] = 1

        df_aug = pd.concat([df_train_orig, synth_sains, synth_malades], ignore_index=True)
        print(f'  ✅ CTGAN: {len(df_train_orig)} → {len(df_aug)} patients')
        return df_aug
    except ImportError:
        print('    ctgan non installé - skip CTGAN (SMOTE seul)')
        return df_train_orig


# ============================================================
# 4. PIPELINE PRINCIPAL
# ============================================================
def train_lr_optuna(files_dict, n_trials=100, use_ctgan=True, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)

    print('\n== 1. Chargement des données ==')
    df = load_data(files_dict)

    print('\n== 2. Nettoyage & Feature Engineering ==')
    df = clean_and_engineer(df)

    print('\n== 3. Split Train/Test ==')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f'  Train: {X_train.shape} | Test: {X_test.shape}')
    print(f'  Balance train: {y_train.value_counts().to_dict()}')

    feature_names = list(X_train.columns)

    print('\n== 4. Scaling (QuantileTransformer) ==')
    continuous_cols = [
        'age','trestbps','chol','thalach','oldpeak',
        'hr_reserve','hr_age_ratio','st_hr_index',
        'age_thalach','age_oldpeak','chol_age','bp_chol'
    ]
    continuous_cols = [c for c in continuous_cols if c in X_train.columns]

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols]  = scaler.transform(X_test[continuous_cols])
    print('  ✅ Scaling OK')

    print('\n== 5. CTGAN Augmentation ==')
    if use_ctgan:
        df_train_orig = X_train.copy()
        df_train_orig['target'] = y_train.values
        df_aug = augment_with_ctgan(df_train_orig, n_synthetic=400)

        X_aug = df_aug.drop('target', axis=1)
        y_aug = df_aug['target']
        X_aug_scaled = X_aug.copy()
        X_aug_scaled[continuous_cols] = scaler.transform(X_aug[continuous_cols])
        X_test_scaled_aug = X_test_scaled
    else:
        X_aug_scaled = X_train_scaled
        y_aug = y_train
        X_test_scaled_aug = X_test_scaled

    print('\n== 6. SMOTE ==')
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_aug_scaled, y_aug)
    print(f'  ✅ SMOTE: {X_aug_scaled.shape[0]} → {X_train_sm.shape[0]} samples')
    print(f'  Balance: {pd.Series(y_train_sm).value_counts().to_dict()}')

    print(f'\n== 7. Optuna LR ({n_trials} trials) ==')
    def objective_lr(trial):
        model = LogisticRegression(
            C=trial.suggest_float('C', 0.001, 100, log=True),
            penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
            solver='liblinear', class_weight='balanced',
            max_iter=2000, random_state=42
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(
            model, X_train_sm, y_train_sm,
            cv=cv, scoring='roc_auc', n_jobs=-1
        ).mean()

    study_lr = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study_lr.optimize(objective_lr, n_trials=n_trials, show_progress_bar=True)
    print(f'  🏆 Meilleur AUC CV : {study_lr.best_value:.4f}')
    print(f'  Params : {study_lr.best_params}')

    print('\n== 8. Entraînement final LR Optuna ==')
    logreg_opt = LogisticRegression(
        C=study_lr.best_params['C'],
        penalty=study_lr.best_params['penalty'],
        solver='liblinear', class_weight='balanced',
        max_iter=2000, random_state=42
    )
    logreg_opt.fit(X_train_sm, y_train_sm)
    proba = logreg_opt.predict_proba(X_test_scaled_aug)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    print(f'  AUC test : {auc:.4f}')

    print('\n== 9. Seuil clinique optimal min(Recall, Spécificité) ==')
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, proba)

    best_thresh = 0.5
    best_score  = -1
    for t, fp_t, tp_t in zip(thresholds, fpr_arr, tpr_arr):
        n_neg = (y_test == 0).sum()
        n_pos = (y_test == 1).sum()
        tn_t = int(round((1 - fp_t) * n_neg))
        tp_t2 = int(round(tp_t * n_pos))
        rec_t  = tp_t2 / n_pos if n_pos > 0 else 0
        spec_t = tn_t  / n_neg if n_neg > 0 else 0
        score  = min(rec_t, spec_t)
        if score > best_score:
            best_score  = score
            best_thresh = t

    y_pred = (proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall   = tp / (tp + fn)
    spec     = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'  Seuil optimal : {best_thresh:.4f}')
    print(f'  Recall        : {recall:.4f}')
    print(f'  Spécificité   : {spec:.4f}')
    print(f'  min(R,S)      : {min(recall, spec):.4f}')
    print(f'  Accuracy      : {accuracy:.4f}')
    print(f'  AUC test      : {auc:.4f}')

    print(f'\n== 10. Sauvegarde dans {output_dir}/ ==')
    with open(os.path.join(output_dir, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(logreg_opt, f)
    with open(os.path.join(output_dir, 'scaler_lr.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, 'threshold_lr.pkl'), 'wb') as f:
        pickle.dump(float(best_thresh), f)
    with open(os.path.join(output_dir, 'feature_names_lr.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    print('   lr_model.pkl')
    print('   scaler_lr.pkl')
    print('   threshold_lr.pkl')
    print('   feature_names_lr.pkl')
    print(f'\n Pipeline terminé - LR Optuna enregistré dans {output_dir}/')

    return logreg_opt, scaler, best_thresh, feature_names


# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LR Optuna on UCI Heart Disease data')
    parser.add_argument('--cleveland',   default='cleveland.data')
    parser.add_argument('--hungarian',   default='hungarian.data')
    parser.add_argument('--switzerland', default='switzerland.data')
    parser.add_argument('--longbeach',   default='long-beach-va.data')
    parser.add_argument('--trials',      type=int, default=100)
    parser.add_argument('--no-ctgan',    action='store_true')
    parser.add_argument('--output',      default='models')
    args = parser.parse_args()

    files_dict = {
        'Cleveland':   args.cleveland,
        'Hungarian':   args.hungarian,
        'Switzerland': args.switzerland,
        'Long Beach':  args.longbeach,
    }

    train_lr_optuna(
        files_dict=files_dict,
        n_trials=args.trials,
        use_ctgan=not args.no_ctgan,
        output_dir=args.output
    )
