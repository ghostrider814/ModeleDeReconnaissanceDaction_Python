# ============================================================
#  utils/model.py — Architecture LSTM
# ============================================================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam


def build_lstm_model(sequence_length: int, n_features: int, n_classes: int):
    """
    Construit le modèle LSTM pour la reconnaissance d'actions.

    Args:
        sequence_length : nombre de frames par séquence (ex: 30)
        n_features      : taille du vecteur keypoints (132 pour MediaPipe Pose)
        n_classes       : nombre d'actions à reconnaître

    Returns:
        model compilé
    """
    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),

            LSTM(64, return_sequences=True, activation="tanh"),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(128, return_sequences=True, activation="tanh"),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(64, return_sequences=False, activation="tanh"),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),

            Dense(n_classes, activation="softmax"),
        ],
        name="ActionLSTM",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path: str):
    """Retourne les callbacks d'entraînement."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
