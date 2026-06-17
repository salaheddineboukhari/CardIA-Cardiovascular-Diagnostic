"""
Script d'entraînement du modèle CNN pour classification ECG
Utilise les datasets MIT-BIH et PTB (si disponibles)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("ENTRAÎNEMENT MODÈLE CNN POUR CLASSIFICATION ECG")
print("=" * 60)

# Créer les dossiers
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# =====================================================
# 1. CHARGEMENT DES DONNÉES
# =====================================================

print("\n📂 1. Chargement des données...")

# MIT-BIH
mitbih_train = pd.read_csv('data/mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('data/mitbih_test.csv', header=None)

print(f"   MIT-BIH train: {mitbih_train.shape}")
print(f"   MIT-BIH test: {mitbih_test.shape}")

# Vérifier PTB
ptb_available = os.path.exists('data/ptbdb_abnormal.csv') and os.path.exists('data/ptbdb_normal.csv')

if ptb_available:
    print("\n📂 Fichiers PTB trouvés, chargement...")
    ptb_abnormal = pd.read_csv('data/ptbdb_abnormal.csv', header=None)
    ptb_normal = pd.read_csv('data/ptbdb_normal.csv', header=None)
    
    # Ajouter les labels
    ptb_abnormal[187] = 1  # Anormal
    ptb_normal[187] = 0     # Normal
    
    # Combiner
    ptb_data = pd.concat([ptb_abnormal, ptb_normal], ignore_index=True)
    ptb_data = ptb_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split 80/20
    split_idx = int(0.8 * len(ptb_data))
    ptb_train = ptb_data[:split_idx]
    ptb_test = ptb_data[split_idx:]
    
    print(f"   PTB train: {ptb_train.shape}")
    print(f"   PTB test: {ptb_test.shape}")
    
    use_ptb = True
else:
    print("\n⚠️ Fichiers PTB non trouvés, utilisation uniquement de MIT-BIH")
    use_ptb = False

# =====================================================
# 2. COMBINAISON DES DATASETS
# =====================================================

print("\n🔗 2. Combinaison des datasets...")

if use_ptb:
    train_df = pd.concat([mitbih_train, ptb_train], ignore_index=True)
    test_df = pd.concat([mitbih_test, ptb_test], ignore_index=True)
    
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   Train combiné: {train_df.shape}")
    print(f"   Test combiné: {test_df.shape}")
else:
    train_df = mitbih_train
    test_df = mitbih_test

# =====================================================
# 3. PRÉPARATION DES DONNÉES
# =====================================================

print("\n🔧 3. Préparation des données...")

# Séparer features et labels
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   Distribution classes (train): {np.bincount(y_train.astype(int))}")

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape pour CNN (samples, timesteps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# One-hot encoding
num_classes = 5
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"   X_train shape (CNN): {X_train.shape}")
print(f"   y_train shape: {y_train_cat.shape}")

# =====================================================
# 4. MODÈLE CNN
# =====================================================

print("\n🏗️ 4. Construction du modèle CNN...")

model = tf.keras.Sequential([
    # Bloc 1
    tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(187, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    
    # Bloc 2
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    
    # Bloc 3
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.3),
    
    # Classification
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

model.summary()

# =====================================================
# 5. ENTRAÎNEMENT
# =====================================================

print("\n🏋️ 5. Entraînement...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('models/best_ecg_model.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# =====================================================
# 6. ÉVALUATION
# =====================================================

print("\n📊 6. Évaluation...")

# Charger le meilleur modèle
best_model = tf.keras.models.load_model('models/best_ecg_model.h5')

# Évaluation
test_loss, test_acc, test_auc = best_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n   Test Accuracy: {test_acc:.4f}")
print(f"   Test AUC: {test_auc:.4f}")
print(f"   Test Loss: {test_loss:.4f}")

# Prédictions
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.savefig('models/ecg_confusion_matrix.png')
plt.show()

# =====================================================
# 7. COURBES D'APPRENTISSAGE
# =====================================================

print("\n📈 7. Courbes d'apprentissage...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('models/ecg_training_curves.png')
plt.show()

# =====================================================
# 8. SAUVEGARDE
# =====================================================

print("\n💾 8. Sauvegarde...")

# Sauvegarder le modèle final
best_model.save('models/ecg_model.h5')
print("   ✅ models/ecg_model.h5")

# Sauvegarder le scaler
joblib.dump(scaler, 'models/scaler_ecg.pkl')
print("   ✅ models/scaler_ecg.pkl")

# =====================================================
# 9. STATISTIQUES FINALES
# =====================================================

print("\n" + "=" * 60)
print("📊 RAPPORT FINAL")
print("=" * 60)

print(f"\n📁 Dataset utilisé: {'MIT-BIH + PTB' if use_ptb else 'MIT-BIH'}")
print(f"📊 Taille train: {len(X_train)}")
print(f"📊 Taille test: {len(X_test)}")
print(f"🎯 Classes: {num_classes}")
print(f"🏆 Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"📈 AUC: {test_auc:.4f}")

print("\n✅ Modèle prêt pour l'intégration dans l'application Flask!")