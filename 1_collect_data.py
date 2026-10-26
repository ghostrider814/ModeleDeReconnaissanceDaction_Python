#!/usr/bin/env python3
# ============================================================
#  1_collect_data.py — Collecte des séquences via webcam
# ============================================================
#
#  Usage :
#      python 1_collect_data.py
#
#  Pour chaque action, le script :
#    - Affiche un compte à rebours (prépare-toi !)
#    - Enregistre SEQUENCES_PER_ACTION squences de SEQUENCE_LENGTH frames
#    - Sauvegarde les keypoints dans data/<action>/<seq_idx>/<frame_idx>.npy
# ============================================================

import os
import cv2
import numpy as np
import time

from config import (
    ACTIONS, SEQUENCE_LENGTH, SEQUENCES_PER_ACTION, DATA_PATH,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
)
from utils import get_pose_model, extract_keypoints, mediapipe_detection, draw_styled_landmarks


# ── Création de l'arborescence des dossiers ────────────────
def create_dirs():
    for action in ACTIONS:
        for seq in range(SEQUENCES_PER_ACTION):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)


# ── Overlay HUD sur la frame ───────────────────────────────
def draw_hud(frame, action, seq, frame_num, collecting):
    h, w = frame.shape[:2]

    # Barre de progression
    progress = int((frame_num / SEQUENCE_LENGTH) * w)
    cv2.rectangle(frame, (0, h - 12), (progress, h), (0, 255, 100), -1)

    # Infos texte
    status_color = (0, 255, 0) if collecting else (0, 165, 255)
    status_text  = "ENREGISTREMENT" if collecting else "PRET..."

    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, f"Action : {action}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Seq {seq + 1}/{SEQUENCES_PER_ACTION}  |  Frame {frame_num}/{SEQUENCE_LENGTH}  |  {status_text}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)


# ── Compte à rebours ───────────────────────────────────────
def countdown(cap, model, seconds=3):
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        _, results = mediapipe_detection(frame, model)
        draw_styled_landmarks(frame, results)

        remaining = int(seconds - (time.time() - start)) + 1
        cv2.putText(frame, str(remaining),
                    (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 165, 255), 6)
        cv2.imshow("Collecte de données", frame)
        cv2.waitKey(1)


# ── Boucle principale ──────────────────────────────────────
def main():
    create_dirs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with get_pose_model(MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE) as pose:
        for action in ACTIONS:
            print(f"\n{'='*50}")
            print(f"  Action : {action.upper()}")
            print(f"  Prépare-toi ! Appuie sur ESPACE pour commencer.")
            print(f"{'='*50}")

            # Attente de l'appui sur ESPACE
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                _, results = mediapipe_detection(frame, pose)
                draw_styled_landmarks(frame, results)

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (20, 20, 20), -1)
                cv2.putText(frame, f"ACTION : {action.upper()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
                cv2.putText(frame, "Appuie sur ESPACE pour commencer",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("Collecte de données", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            for seq in range(SEQUENCES_PER_ACTION):
                # Compte à rebours entre chaque séquence
                countdown(cap, pose, seconds=2)

                keypoints_sequence = []

                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    _, results = mediapipe_detection(frame, pose)
                    draw_styled_landmarks(frame, results)

                    # Extraction et sauvegarde
                    kp = extract_keypoints(results)
                    keypoints_sequence.append(kp)

                    npy_path = os.path.join(DATA_PATH, action, str(seq), str(frame_num))
                    np.save(npy_path, kp)

                    draw_hud(frame, action, seq, frame_num + 1, collecting=True)
                    cv2.imshow("Collecte de données", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                print(f"  ✓ Séquence {seq + 1}/{SEQUENCES_PER_ACTION} enregistrée")

            print(f"  ✅ '{action}' terminé !")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Collecte terminée ! Lance maintenant : python 2_train_model.py")


if __name__ == "__main__":
    main()
