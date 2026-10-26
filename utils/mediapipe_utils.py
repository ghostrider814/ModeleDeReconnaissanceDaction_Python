# ============================================================
#  utils/mediapipe_utils.py — Extraction des keypoints
# ============================================================

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_pose_model(detection_confidence=0.5, tracking_confidence=0.5):
    """Retourne une instance du modèle MediaPipe Pose."""
    return mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )


def extract_keypoints(results) -> np.ndarray:
    """
    Extrait les 33 keypoints (x, y, z, visibility) du résultat MediaPipe.
    Retourne un vecteur de taille 132. Si aucune pose détectée → zéros.
    """
    if results.pose_landmarks:
        pose = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]
        ).flatten()  # shape (132,)
    else:
        pose = np.zeros(33 * 4)
    return pose


def mediapipe_detection(frame: np.ndarray, model):
    """
    Lance la détection MediaPipe sur une frame BGR.
    Retourne (frame_rgb, results).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = model.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return frame_rgb, results


def draw_landmarks(frame: np.ndarray, results):
    """Dessine le squelette sur la frame."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )


def draw_styled_landmarks(frame: np.ndarray, results):
    """Dessine le squelette avec un style personnalisé (plus visible)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )
