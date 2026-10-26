#!/usr/bin/env python3
# ============================================================
#  3_realtime_inference.py — Inférence temps réel via webcam
# ============================================================
#
#  Usage :
#      python 3_realtime_inference.py
#
#  Raccourcis clavier :
#      Q  →  Quitter
#      R  →  Réinitialiser le buffer de séquence
#      S  →  Screenshot (sauvegardé dans captures/)
# ============================================================

import os
import cv2
import numpy as np
from collections import deque, Counter
from datetime import datetime
import tensorflow as tf

from config import (
    SEQUENCE_LENGTH, MODEL_PATH, LABEL_PATH,
    CONFIDENCE_THRESHOLD,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
)
from utils import get_pose_model, extract_keypoints, mediapipe_detection, draw_styled_landmarks


# ── Palette de couleurs par action ────────────────────────
COLORS = [
    (0,   255, 100),
    (0,   200, 255),
    (255, 100, 0  ),
    (255, 0,   150),
    (180, 0,   255),
    (255, 255, 0  ),
    (0,   128, 255),
]


def get_action_color(action_idx: int):
    return COLORS[action_idx % len(COLORS)]


# ── Barre de probabilités ─────────────────────────────────
def draw_probability_bars(frame, predictions, label_names):
    h, w = frame.shape[:2]
    bar_w   = 180
    bar_h   = 18
    padding = 8
    start_x = w - bar_w - 15
    start_y = 80

    cv2.rectangle(frame, (start_x - 10, start_y - 25),
                  (w - 5, start_y + len(label_names) * (bar_h + padding) + 5),
                  (20, 20, 20), -1)

    cv2.putText(frame, "Probabilités", (start_x - 5, start_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for i, (prob, label) in enumerate(zip(predictions, label_names)):
        y = start_y + i * (bar_h + padding)
        color = get_action_color(i)

        # Fond gris
        cv2.rectangle(frame, (start_x, y), (start_x + bar_w, y + bar_h),
                      (60, 60, 60), -1)
        # Barre colorée
        filled = int(prob * bar_w)
        cv2.rectangle(frame, (start_x, y), (start_x + filled, y + bar_h),
                      color, -1)
        # Texte
        cv2.putText(frame, f"{label}  {prob * 100:.1f}%",
                    (start_x + 4, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


# ── Affichage de l'action prédite ─────────────────────────
def draw_prediction(frame, action: str, confidence: float, action_idx: int):
    h, w = frame.shape[:2]
    color = get_action_color(action_idx)

    # Fond semi-transparent en bas
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"{action.upper()}",
                (15, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(frame, f"Confiance : {confidence * 100:.1f}%",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ── Buffer visuel (historique des prédictions) ─────────────
def draw_history(frame, history_deque, label_names):
    x = 10
    for i, action in enumerate(history_deque):
        idx = label_names.index(action) if action in label_names else 0
        color = get_action_color(idx)
        cv2.circle(frame, (x + i * 22, 75), 8, color, -1)


# ── Main ───────────────────────────────────────────────────
def main():
    # --- Chargement du modèle ---
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR] Modèle introuvable : {MODEL_PATH}")
        print("  Lance d'abord : python 2_train_model.py")
        return
    if not os.path.exists(LABEL_PATH):
        print(f"[ERREUR] Labels introuvables : {LABEL_PATH}")
        return

    print("Chargement du modèle...")
    model = tf.keras.models.load_model(MODEL_PATH)
    label_names = list(np.load(LABEL_PATH, allow_pickle=True))
    print(f"  → {len(label_names)} actions : {label_names}")

    # --- Variables d'état ---
    sequence     = deque(maxlen=SEQUENCE_LENGTH)
    predictions  = deque(maxlen=10)  # lissage sur 10 prédictions
    action_history = deque(maxlen=15)

    current_action  = ""
    current_conf    = 0.0
    current_probs   = np.zeros(len(label_names))
    frame_count     = 0

    os.makedirs("captures", exist_ok=True)

    # --- Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n✅ Inférence en cours... (Q=Quitter, R=Reset, S=Screenshot)")

    with get_pose_model(MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            _, results = mediapipe_detection(frame, pose)
            draw_styled_landmarks(frame, results)

            # Extraction des keypoints
            kp = extract_keypoints(results)
            sequence.append(kp)
            frame_count += 1

            # Inférence dès que le buffer est plein
            if len(sequence) == SEQUENCE_LENGTH:
                input_seq = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 132)
                probs = model.predict(input_seq, verbose=0)[0]

                pred_idx  = np.argmax(probs)
                pred_conf = probs[pred_idx]
                predictions.append(pred_idx)

                # Lissage : action la plus fréquente sur les 10 dernières prédictions
                smoothed_idx = Counter(predictions).most_common(1)[0][0]

                current_probs = probs
                if pred_conf >= CONFIDENCE_THRESHOLD:
                    current_action = label_names[smoothed_idx]
                    current_conf   = pred_conf
                    action_history.append(current_action)
                else:
                    current_action = "..."
                    current_conf   = pred_conf

            # ── HUD ──────────────────────────────────────────
            # Header
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (20, 20, 20), -1)
            cv2.putText(frame, "ACTION RECOGNITION — Temps réel",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Buffer de remplissage
            buf_ratio = len(sequence) / SEQUENCE_LENGTH
            buf_w     = 250
            cv2.rectangle(frame, (10, 32), (10 + buf_w, 48), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 32), (10 + int(buf_w * buf_ratio), 48),
                          (0, 200, 255), -1)
            cv2.putText(frame, f"Buffer {int(buf_ratio * 100)}%",
                        (270, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Prédiction courante
            if current_action and current_action != "...":
                idx = label_names.index(current_action) if current_action in label_names else 0
                draw_prediction(frame, current_action, current_conf, idx)

            # Barres de probabilités
            if np.any(current_probs):
                draw_probability_bars(frame, current_probs, label_names)

            # Historique
            if action_history:
                draw_history(frame, action_history, label_names)

            # FPS
            cv2.putText(frame, f"Frame #{frame_count}",
                        (frame.shape[1] - 120, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("Reconnaissance d'actions", frame)

            # ── Gestion des touches ───────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sequence.clear()
                predictions.clear()
                current_action = ""
                print("  → Buffer réinitialisé")
            elif key == ord("s"):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"captures/capture_{ts}.jpg"
                cv2.imwrite(path, frame)
                print(f"  → Screenshot : {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nAu revoir !")


if __name__ == "__main__":
    main()
