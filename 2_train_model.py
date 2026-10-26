#!/usr/bin/env python3
# ============================================================
#  2_train_model.py — Entraînement du modèle LSTM
# ============================================================
#
#  Usage :
#      python 2_train_model.py
#
#  Charge les données collectées, entraîne le LSTM,
#  sauvegarde le modèle et affiche les courbes d'apprentissage.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.utils import to_categorical

from config import (
    ACTIONS, SEQUENCE_LENGTH, DATA_PATH,
    MODEL_PATH, LABEL_PATH,
    EPOCHS, BATCH_SIZE, TEST_SIZE,
)
from utils import build_lstm_model, get_callbacks


# ── 1. Chargement des données ──────────────────────────────
def load_dataset():
    X, y = [], []
    missing = []

    for action in ACTIONS:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_dir):
            print(f"[WARN] Dossier manquant : {action_dir}")
            continue

        sequences = sorted(os.listdir(action_dir), key=lambda x: int(x))
        for seq in sequences:
            seq_dir = os.path.join(action_dir, seq)
            frames = []
            valid = True

            for frame_num in range(SEQUENCE_LENGTH):
                npy_file = os.path.join(seq_dir, f"{frame_num}.npy")
                if not os.path.exists(npy_file):
                    valid = False
                    missing.append(npy_file)
                    break
                frames.append(np.load(npy_file))

            if valid and len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(action)

    if missing:
        print(f"[WARN] {len(missing)} fichiers manquants ignorés.")

    return np.array(X, dtype=np.float32), np.array(y)


# ── 2. Visualisation des courbes ───────────────────────────
def plot_training(history, save_path="models/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val",   linewidth=2)
    axes[0].set_title("Accuracy", fontsize=14)
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val",   linewidth=2)
    axes[1].set_title("Loss", fontsize=14)
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  → Courbes sauvegardées : {save_path}")


# ── 3. Matrice de confusion ────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test_cat, label_names,
                          save_path="models/confusion_matrix.png"):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names
    )
    plt.title("Matrice de confusion", fontsize=14)
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  → Matrice de confusion sauvegardée : {save_path}")

    print("\n── Rapport de classification ──")
    print(classification_report(y_true, y_pred, target_names=label_names))


# ── Main ───────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ENTRAÎNEMENT DU MODÈLE — RECONNAISSANCE D'ACTIONS")
    print("=" * 55)

    # --- Chargement ---
    print("\n[1/4] Chargement des données...")
    X, y_raw = load_dataset()

    if len(X) == 0:
        print("[ERREUR] Aucune donnée trouvée. Lance d'abord : python 1_collect_data.py")
        return

    print(f"  → {len(X)} séquences chargées | Shape : {X.shape}")

    # --- Encodage des labels ---
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    y_cat = to_categorical(y_enc, num_classes=len(le.classes_))

    label_names = list(le.classes_)
    print(f"  → Classes : {label_names}")

    np.save(LABEL_PATH, label_names)
    print(f"  → Labels sauvegardés : {LABEL_PATH}")

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=42, stratify=y_enc
    )
    print(f"  → Train : {len(X_train)} | Test : {len(X_test)}")

    # --- Modèle ---
    print("\n[2/4] Construction du modèle...")
    n_features = X.shape[2]  # 132
    model = build_lstm_model(SEQUENCE_LENGTH, n_features, len(label_names))
    model.summary()

    # --- Entraînement ---
    print(f"\n[3/4] Entraînement ({EPOCHS} époques max)...")
    os.makedirs("models", exist_ok=True)
    callbacks = get_callbacks(MODEL_PATH)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Évaluation ---
    print("\n[4/4] Évaluation...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  → Loss : {loss:.4f} | Accuracy : {acc * 100:.2f}%")

    plot_training(history)
    plot_confusion_matrix(model, X_test, y_test, label_names)

    print(f"\n✅ Modèle sauvegardé : {MODEL_PATH}")
    print("   Lance maintenant : python 3_realtime_inference.py")


if __name__ == "__main__":
    main()
