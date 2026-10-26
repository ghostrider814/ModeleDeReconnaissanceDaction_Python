# ============================================================
#  config.py — Configuration centrale du projet
# ============================================================

# Actions à reconnaître (modifie cette liste selon tes besoins)
ACTIONS = ["marcher", "sauter", "tomber", "courir", "s_asseoir"]

# Nombre de frames par séquence
SEQUENCE_LENGTH = 30

# Nombre de séquences collectées par action
SEQUENCES_PER_ACTION = 40

# Dossier de sauvegarde des données
DATA_PATH = "data"

# Dossier de sauvegarde du modèle
MODEL_PATH = "models/action_lstm.keras"
LABEL_PATH = "models/labels.npy"

# Seuil de confiance pour l'affichage de la prédiction
CONFIDENCE_THRESHOLD = 0.75

# Paramètres MediaPipe
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE  = 0.5

# Paramètres d'entraînement
EPOCHS     = 50
BATCH_SIZE = 16
TEST_SIZE  = 0.2
