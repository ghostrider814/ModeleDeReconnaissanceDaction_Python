# Reconnaissance d'Actions en Temps Réel

Système de reconnaissance d'actions du corps entier via webcam, basé sur **MediaPipe Pose** + **LSTM** (TensorFlow/Keras).

---

##  Structure du projet

```
action_recognition/
├── config.py                  #   Tous les paramètres
├── requirements.txt           #  Dépendances
│
├── 1_collect_data.py          #  Étape 1 : Collecte des données
├── 2_train_model.py           #  Étape 2 : Entraînement
├── 3_realtime_inference.py    #  Étape 3 : Inférence temps réel
│
├── utils/
│   ├── __init__.py
│   ├── mediapipe_utils.py     # Extraction des keypoints MediaPipe
│   └── model.py               # Architecture LSTM
│
├── data/                      # Données collectées (auto-créé)
├── models/                    # Modèle sauvegardé (auto-créé)
└── captures/                  # Screenshots (auto-créé)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

##  Utilisation (3 étapes)

### Étape 1 — Collecte des données

```bash
python 1_collect_data.py
```

- Pour **chaque action**, positionne-toi devant la webcam
- Appuie sur **ESPACE** pour démarrer l'enregistrement
- Le script enregistre automatiquement `SEQUENCES_PER_ACTION` séquences de `SEQUENCE_LENGTH` frames
- Les données sont sauvegardées dans `data/<action>/<seq>/<frame>.npy`

### Étape 2 — Entraînement

```bash
python 2_train_model.py
```

- Charge toutes les séquences collectées
- Entraîne le modèle LSTM avec early stopping
- Affiche les courbes d'apprentissage et la matrice de confusion
- Sauvegarde le modèle dans `models/action_lstm.keras`

### Étape 3 — Inférence temps réel

```bash
python 3_realtime_inference.py
```

| Touche | Action               |
|--------|----------------------|
| `Q`    | Quitter              |
| `R`    | Réinitialiser buffer |
| `S`    | Screenshot           |

---

## ⚙️ Configuration (`config.py`)

| Paramètre              | Défaut                                         | Description                          |
|------------------------|------------------------------------------------|--------------------------------------|
| `ACTIONS`              | `["marcher", "sauter", "tomber", "courir", "s_asseoir"]` | Actions à reconnaître |
| `SEQUENCE_LENGTH`      | `30`                                           | Frames par séquence                  |
| `SEQUENCES_PER_ACTION` | `40`                                           | Séquences collectées par action      |
| `CONFIDENCE_THRESHOLD` | `0.75`                                         | Seuil d'affichage de la prédiction   |
| `EPOCHS`               | `50`                                           | Époques max d'entraînement           |

### Ajouter une nouvelle action

1. Ajoute le nom dans `ACTIONS` dans `config.py`
2. Relance la collecte : `python 1_collect_data.py`
3. Ré-entraîne : `python 2_train_model.py`

---

##  Architecture

```
Webcam (frame)
    ↓
MediaPipe Pose  →  33 keypoints × 4 valeurs (x, y, z, visibility)  =  132 features
    ↓
Buffer circulaire de 30 frames
    ↓
LSTM(64)  →  BatchNorm  →  Dropout
LSTM(128) →  BatchNorm  →  Dropout
LSTM(64)  →  BatchNorm  →  Dropout
Dense(64) →  Dense(32)  →  Dense(n_actions, softmax)
    ↓
Lissage sur 10 prédictions (vote majoritaire)
    ↓
Affichage de l'action + barres de probabilités
```

---

##  Conseils pour une bonne collecte

- **Éclairage** : bonne lumière frontale, évite le contre-jour
- **Distance** : 2 à 3 mètres de la caméra (corps entier visible)
- **Variété** : varie vitesse, direction, style pour chaque action
- **Minimum** : 40 séquences par action (plus = mieux)
- **Fond** : fond uni de préférence (pas trop de mouvement derrière toi)
