#!/usr/bin/env python3
# ============================================================
#  app.py — Interface graphique principale
#  Usage : python app.py
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os
import queue
from collections import deque, Counter
from datetime import datetime

from config import (
    ACTIONS, SEQUENCE_LENGTH, SEQUENCES_PER_ACTION,
    DATA_PATH, MODEL_PATH, LABEL_PATH,
    CONFIDENCE_THRESHOLD, EPOCHS, BATCH_SIZE, TEST_SIZE,
    MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE,
)
from utils import (
    get_pose_model, extract_keypoints,
    mediapipe_detection, draw_styled_landmarks,
)

# Palette
BG        = "#0d0d1a"
PANEL     = "#13132b"
CARD      = "#1a1a35"
BORDER    = "#2a2a50"
PRIMARY   = "#00e5c0"
SECONDARY = "#7c6cf0"
DANGER    = "#ff4d6d"
WARNING   = "#ffb347"
SUCCESS   = "#00e5c0"
TEXT      = "#e8e8f0"
TEXT_DIM  = "#8888aa"
WHITE     = "#ffffff"

ACTION_COLORS = [
    "#00e5c0","#7c6cf0","#ff6b9d",
    "#ffb347","#4ecdc4","#ff6b6b","#a8e6cf",
]

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


class ActionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Action Recognition")
        self.root.configure(bg=BG)
        self.root.geometry("1280x760")
        self.root.minsize(1100, 680)

        self.mode           = "idle"
        self.running        = False
        self.frame_queue    = queue.Queue(maxsize=2)
        self.log_queue      = queue.Queue()
        self.sequence       = deque(maxlen=SEQUENCE_LENGTH)
        self.preds_buf      = deque(maxlen=10)
        self.action_hist    = deque(maxlen=20)
        self.current_action = ""
        self.current_conf   = 0.0
        self.label_names    = []

        self._build_ui()
        self._update_frame_display()
        self._poll_log_queue()

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self):
        header = tk.Frame(self.root, bg=PANEL, height=55)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="⚡  ACTION RECOGNITION",
                 font=("Segoe UI", 15, "bold"),
                 bg=PANEL, fg=PRIMARY).pack(side="left", padx=20, pady=10)
        self.status_lbl = tk.Label(header, text="● EN ATTENTE",
                                    font=FONT_BODY, bg=PANEL, fg=TEXT_DIM)
        self.status_lbl.pack(side="right", padx=20)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Gauche : webcam
        left = tk.Frame(body, bg=CARD, highlightthickness=1,
                         highlightbackground=BORDER)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.cam_label = tk.Label(left, bg="#000010",
                                   text="Webcam inactive",
                                   fg=TEXT_DIM, font=FONT_BODY)
        self.cam_label.pack(fill="both", expand=True)
        cam_bar = tk.Frame(left, bg=PANEL, height=28)
        cam_bar.pack(fill="x", side="bottom")
        cam_bar.pack_propagate(False)
        self.cam_info = tk.Label(cam_bar, text="—", font=FONT_SMALL,
                                  bg=PANEL, fg=TEXT_DIM)
        self.cam_info.pack(side="left", padx=10, pady=5)

        # Droite : contrôles
        right = tk.Frame(body, bg=BG, width=340)
        right.pack(side="right", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("C.TNotebook", background=BG, borderwidth=0)
        style.configure("C.TNotebook.Tab", background=CARD,
                         foreground=TEXT_DIM, padding=[12, 5],
                         font=FONT_SMALL, borderwidth=0)
        style.map("C.TNotebook.Tab",
                  background=[("selected", PRIMARY)],
                  foreground=[("selected", "#000000")])

        nb = ttk.Notebook(right, style="C.TNotebook")
        nb.pack(fill="both", expand=True)

        t1 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="  Contrôles  ")
        self._build_controls(t1)

        t2 = tk.Frame(nb, bg=BG)
        nb.add(t2, text="  Détection  ")
        self._build_detection(t2)

        t3 = tk.Frame(nb, bg=BG)
        nb.add(t3, text="  Console  ")
        self._build_console(t3)

    def _section(self, p, txt):
        tk.Label(p, text=txt, font=("Segoe UI", 9, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(anchor="w", padx=12, pady=(8,3))

    def _btn(self, p, txt, color, cmd):
        return tk.Button(p, text=txt, bg=color, fg="#000000",
                          font=("Segoe UI", 10, "bold"),
                          bd=0, pady=9, cursor="hand2",
                          activebackground=color, activeforeground="#000000",
                          command=cmd)

    def _build_controls(self, p):
        self._section(p, "ETAPE 1 — Collecte")
        acts = tk.Frame(p, bg=CARD, highlightthickness=1,
                         highlightbackground=BORDER)
        acts.pack(fill="x", padx=12, pady=(0,6))
        for i, a in enumerate(ACTIONS):
            c = ACTION_COLORS[i % len(ACTION_COLORS)]
            r = tk.Frame(acts, bg=CARD); r.pack(fill="x", padx=8, pady=2)
            tk.Label(r, text="●", fg=c, bg=CARD, font=("Segoe UI",9)).pack(side="left")
            tk.Label(r, text=f"  {a}", fg=TEXT, bg=CARD, font=FONT_SMALL).pack(side="left")
            tk.Label(r, text=f"{SEQUENCES_PER_ACTION} séq.", fg=TEXT_DIM,
                     bg=CARD, font=FONT_SMALL).pack(side="right", padx=4)

        self.collect_prog_lbl = tk.Label(p, text="", bg=BG,
                                          fg=TEXT_DIM, font=FONT_SMALL)
        self.collect_prog_lbl.pack(padx=12, pady=(4,2))
        self.collect_progress = ttk.Progressbar(p, length=300, mode="determinate")
        self.collect_progress.pack(padx=12, pady=(0,4), fill="x")
        self.btn_collect = self._btn(p, "Demarrer la collecte", PRIMARY, self._start_collection)
        self.btn_collect.pack(fill="x", padx=12, pady=(0,10))

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=12, pady=6)
        self._section(p, "ETAPE 2 — Entrainement")

        info = tk.Frame(p, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        info.pack(fill="x", padx=12, pady=(0,6))
        for lbl, val in [("Epoques max", str(EPOCHS)),
                         ("Batch size", str(BATCH_SIZE)),
                         ("Split test", f"{int(TEST_SIZE*100)}%")]:
            r = tk.Frame(info, bg=CARD); r.pack(fill="x", padx=8, pady=2)
            tk.Label(r, text=lbl, fg=TEXT_DIM, bg=CARD, font=FONT_SMALL).pack(side="left")
            tk.Label(r, text=val, fg=PRIMARY, bg=CARD,
                     font=("Segoe UI",9,"bold")).pack(side="right", padx=4)

        self.train_progress = ttk.Progressbar(p, length=300, mode="indeterminate")
        self.train_progress.pack(padx=12, pady=(6,4), fill="x")
        self.train_prog_lbl = tk.Label(p, text="", bg=BG, fg=TEXT_DIM, font=FONT_SMALL)
        self.train_prog_lbl.pack(padx=12, pady=(0,4))
        self.btn_train = self._btn(p, "Entrainer le modele", SECONDARY, self._start_training)
        self.btn_train.pack(fill="x", padx=12, pady=(0,10))

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=12, pady=6)
        self._section(p, "ETAPE 3 — Inference temps reel")

        self.model_status = tk.Label(p, text="Modele non charge",
                                      bg=BG, fg=WARNING, font=FONT_SMALL)
        self.model_status.pack(padx=12, pady=(0,4))
        self._check_model()

        self.btn_infer = self._btn(p, "Lancer l'inference", WARNING, self._start_inference)
        self.btn_infer.pack(fill="x", padx=12, pady=(0,6))
        self.btn_stop = self._btn(p, "Arreter", DANGER, self._stop)
        self.btn_stop.pack(fill="x", padx=12)
        self.btn_stop.config(state="disabled")

    def _build_detection(self, p):
        self.action_big = tk.Label(p, text="—",
                                    font=("Segoe UI", 38, "bold"),
                                    bg=BG, fg=PRIMARY)
        self.action_big.pack(pady=(20,4))
        self.conf_lbl = tk.Label(p, text="Confiance : —",
                                  font=FONT_BODY, bg=BG, fg=TEXT_DIM)
        self.conf_lbl.pack()
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=12, pady=12)
        self._section(p, "Probabilites par action")

        self.prob_bars = {}
        self.prob_pcts = {}
        for i, action in enumerate(ACTIONS):
            color = ACTION_COLORS[i % len(ACTION_COLORS)]
            row = tk.Frame(p, bg=BG); row.pack(fill="x", padx=12, pady=3)
            tk.Label(row, text=action, fg=TEXT, bg=BG,
                     font=FONT_SMALL, width=12, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=CARD, height=14, width=160,
                               highlightthickness=1, highlightbackground=BORDER)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg=color, height=14, width=0)
            bar.place(x=0, y=0, height=14)
            self.prob_bars[action] = (bar, bar_bg, color)
            pct = tk.Label(row, text="0%", fg=TEXT_DIM, bg=BG,
                            font=FONT_SMALL, width=5)
            pct.pack(side="left")
            self.prob_pcts[action] = pct

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=12, pady=12)
        self._section(p, "Historique")
        self.hist_frame = tk.Frame(p, bg=BG)
        self.hist_frame.pack(fill="x", padx=12, pady=4)

    def _build_console(self, p):
        self.log_text = tk.Text(p, bg="#080812", fg="#00ff88",
                                 font=FONT_MONO, wrap="word",
                                 state="disabled", relief="flat")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        tk.Button(p, text="Effacer", bg=CARD, fg=TEXT_DIM,
                   font=FONT_SMALL, bd=0, cursor="hand2",
                   activebackground=BORDER, activeforeground=WHITE,
                   command=self._clear_log).pack(pady=4)

    # ── Helpers ───────────────────────────────────────────
    def _check_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
            self.model_status.config(text="Modele pret", fg=SUCCESS)
        else:
            self.model_status.config(text="Modele non trouve — entraine d'abord", fg=WARNING)

    def _set_status(self, txt, color=TEXT_DIM):
        self.status_lbl.config(text=txt, fg=color)

    def _lock_buttons(self):
        self.btn_collect.config(state="disabled")
        self.btn_train.config(state="disabled")
        self.btn_infer.config(state="disabled")
        self.btn_stop.config(state="normal")

    def _stop(self):
        self.running = False
        self._set_status("● EN ATTENTE", TEXT_DIM)
        self.btn_stop.config(state="disabled")
        self.btn_collect.config(state="normal")
        self.btn_train.config(state="normal")
        self.btn_infer.config(state="normal")
        self.cam_label.config(image="", text="Webcam inactive", fg=TEXT_DIM)
        self.log("Session arretee.")

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {msg}\n")

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(100, self._poll_log_queue)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _update_frame_display(self):
        try:
            frame = self.frame_queue.get_nowait()
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w = max(self.cam_label.winfo_width(), 860)
            h = max(self.cam_label.winfo_height(), 620)
            img   = img.resize((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.cam_label.config(image=photo, text="")
            self.cam_label._photo = photo
        except queue.Empty:
            pass
        self.root.after(16, self._update_frame_display)

    def _push_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())

    # ── Collecte ─────────────────────────────────────────
    def _start_collection(self):
        self._lock_buttons()
        self.mode    = "collect"
        self.running = True
        self._set_status("● COLLECTE EN COURS", PRIMARY)
        self.log("Demarrage de la collecte...")
        threading.Thread(target=self._collect_loop, daemon=True).start()

    def _collect_loop(self):
        for action in ACTIONS:
            for seq in range(SEQUENCES_PER_ACTION):
                os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

        total   = len(ACTIONS) * SEQUENCES_PER_ACTION
        done    = 0
        cap     = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with get_pose_model(MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE) as pose:
            for action_idx, action in enumerate(ACTIONS):
                if not self.running: break
                self.log(f"Action : {action.upper()} ({action_idx+1}/{len(ACTIONS)})")
                self.root.after(0, self._set_status,
                                f"APPUIE SUR ESPACE — {action.upper()}", WARNING)

                # Attente ESPACE
                while self.running:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    _, results = mediapipe_detection(frame, pose)
                    draw_styled_landmarks(frame, results)
                    cv2.rectangle(frame, (0,0), (frame.shape[1],60), (20,20,30), -1)
                    cv2.putText(frame, f"ACTION : {action.upper()}",
                                (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,229,192), 2)
                    cv2.putText(frame, "Appuie sur ESPACE pour commencer",
                                (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                    self._push_frame(frame)
                    if cv2.waitKey(1) & 0xFF == ord(" "): break

                for seq in range(SEQUENCES_PER_ACTION):
                    if not self.running: break
                    # Countdown 2s
                    t0 = time.time()
                    while time.time()-t0 < 2 and self.running:
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        _, results = mediapipe_detection(frame, pose)
                        draw_styled_landmarks(frame, results)
                        rem = int(2-(time.time()-t0))+1
                        cv2.putText(frame, str(rem),
                                    (frame.shape[1]//2-30, frame.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0,165,255), 6)
                        self._push_frame(frame)
                        cv2.waitKey(1)

                    # Enregistrement
                    for f_num in range(SEQUENCE_LENGTH):
                        if not self.running: break
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        _, results = mediapipe_detection(frame, pose)
                        draw_styled_landmarks(frame, results)
                        kp = extract_keypoints(results)
                        np.save(os.path.join(DATA_PATH, action, str(seq), str(f_num)), kp)
                        h, w = frame.shape[:2]
                        prog = int((f_num/SEQUENCE_LENGTH)*w)
                        cv2.rectangle(frame, (0,h-10), (prog,h), (0,229,192), -1)
                        cv2.rectangle(frame, (0,0), (w,55), (20,20,30), -1)
                        cv2.putText(frame,
                                    f"{action.upper()}  Seq {seq+1}/{SEQUENCES_PER_ACTION}"
                                    f"  Frame {f_num+1}/{SEQUENCE_LENGTH}",
                                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,229,192), 2)
                        self._push_frame(frame)
                        cv2.waitKey(1)

                    done += 1
                    pct = done/total*100
                    self.root.after(0, self._upd_collect, pct, action, seq)
                    self.log(f"  seq {seq+1}/{SEQUENCES_PER_ACTION} — {action}")

                self.log(f"Action '{action}' terminee !")

        cap.release()
        if self.running:
            self.log("Collecte terminee ! Lance maintenant l'entrainement.")
            self.root.after(0, self._stop)

    def _upd_collect(self, pct, action, seq):
        self.collect_progress["value"] = pct
        self.collect_prog_lbl.config(
            text=f"{action} — seq {seq+1}/{SEQUENCES_PER_ACTION}  ({pct:.0f}%)")

    # ── Entrainement ─────────────────────────────────────
    def _start_training(self):
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            messagebox.showerror("Erreur", "Aucune donnee. Lance d'abord la collecte.")
            return
        self._lock_buttons()
        self.mode    = "train"
        self.running = True
        self._set_status("● ENTRAINEMENT EN COURS", SECONDARY)
        self.train_progress.start(10)
        self.train_prog_lbl.config(text="Chargement des donnees...")
        self.log("Demarrage de l'entrainement...")
        threading.Thread(target=self._train_loop, daemon=True).start()

    def _train_loop(self):
        try:
            import tensorflow as tf
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from tensorflow.keras.utils import to_categorical
            from utils import build_lstm_model, get_callbacks

            X, y_raw = [], []
            for action in ACTIONS:
                adir = os.path.join(DATA_PATH, action)
                if not os.path.isdir(adir): continue
                for seq in sorted(os.listdir(adir), key=lambda x: int(x)):
                    frames, valid = [], True
                    for fn in range(SEQUENCE_LENGTH):
                        fp = os.path.join(adir, seq, f"{fn}.npy")
                        if not os.path.exists(fp): valid=False; break
                        frames.append(np.load(fp))
                    if valid: X.append(frames); y_raw.append(action)

            if not X:
                self.log("Aucune donnee valide.")
                self.root.after(0, self._stop); return

            X = np.array(X, dtype=np.float32)
            self.log(f"  {len(X)} sequences chargees")

            le    = LabelEncoder()
            y_enc = le.fit_transform(y_raw)
            y_cat = to_categorical(y_enc, num_classes=len(le.classes_))
            lnames = list(le.classes_)
            np.save(LABEL_PATH, lnames)
            self.log(f"  Classes : {lnames}")

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_cat, test_size=TEST_SIZE, random_state=42, stratify=y_enc)

            model = build_lstm_model(SEQUENCE_LENGTH, X.shape[2], len(lnames))
            os.makedirs("models", exist_ok=True)
            cbs = get_callbacks(MODEL_PATH)

            class UiCb(tf.keras.callbacks.Callback):
                def on_epoch_end(cb, epoch, logs=None):
                    if logs:
                        self.log(f"  Ep {epoch+1}/{EPOCHS}  "
                                 f"acc={logs.get('accuracy',0):.3f}  "
                                 f"val_acc={logs.get('val_accuracy',0):.3f}")
                        self.root.after(0, self.train_prog_lbl.config,
                                        {"text": f"Ep {epoch+1}/{EPOCHS} — "
                                                 f"val_acc "
                                                 f"{logs.get('val_accuracy',0)*100:.1f}%"})
            cbs.append(UiCb())

            model.fit(X_tr, y_tr, validation_data=(X_te,y_te),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=cbs, verbose=0)

            loss, acc = model.evaluate(X_te, y_te, verbose=0)
            self.log(f"Entrainement termine ! Accuracy: {acc*100:.2f}%")
            self.root.after(0, self._check_model)
        except Exception as e:
            self.log(f"Erreur : {e}")
        finally:
            self.root.after(0, self.train_progress.stop)
            self.root.after(0, self._stop)

    # ── Inference ────────────────────────────────────────
    def _start_inference(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Erreur", "Modele introuvable. Entraine d'abord.")
            return
        self._lock_buttons()
        self.mode    = "inference"
        self.running = True
        self._set_status("● INFERENCE EN COURS", SUCCESS)
        self.sequence.clear(); self.preds_buf.clear()
        self.log("Chargement du modele...")
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def _inference_loop(self):
        try:
            import tensorflow as tf
            model      = tf.keras.models.load_model(MODEL_PATH)
            lnames     = list(np.load(LABEL_PATH, allow_pickle=True))
            self.label_names = lnames
            self.log(f"  Modele charge | {len(lnames)} actions : {lnames}")

            cur_probs  = np.zeros(len(lnames))
            cur_action = ""
            cur_conf   = 0.0

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            with get_pose_model(MP_DETECTION_CONFIDENCE, MP_TRACKING_CONFIDENCE) as pose:
                while self.running:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    _, results = mediapipe_detection(frame, pose)
                    draw_styled_landmarks(frame, results)
                    kp = extract_keypoints(results)
                    self.sequence.append(kp)

                    if len(self.sequence) == SEQUENCE_LENGTH:
                        inp   = np.expand_dims(np.array(self.sequence), axis=0)
                        probs = model.predict(inp, verbose=0)[0]
                        idx   = int(np.argmax(probs))
                        conf  = float(probs[idx])
                        self.preds_buf.append(idx)
                        smoothed = Counter(self.preds_buf).most_common(1)[0][0]
                        cur_probs = probs
                        if conf >= CONFIDENCE_THRESHOLD:
                            cur_action = lnames[smoothed]
                            cur_conf   = conf
                            if not self.action_hist or self.action_hist[-1] != cur_action:
                                self.action_hist.append(cur_action)
                        else:
                            cur_action = "..."; cur_conf = conf
                        self.root.after(0, self._upd_inference_ui,
                                        cur_action, cur_conf, cur_probs, lnames)

                    # Overlay
                    h, w = frame.shape[:2]
                    buf_r = len(self.sequence)/SEQUENCE_LENGTH
                    cv2.rectangle(frame, (0,h-8), (int(buf_r*w),h), (0,229,192), -1)
                    if cur_action and cur_action != "...":
                        cv2.rectangle(frame,(0,h-58),(w,h-10),(15,15,25),-1)
                        cv2.putText(frame, cur_action.upper(),
                                    (12,h-20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.3, (0,229,192), 3)
                        cv2.putText(frame, f"{cur_conf*100:.1f}%",
                                    (12,h-8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45, (180,180,180), 1)
                    self._push_frame(frame)

            cap.release()
        except Exception as e:
            self.log(f"Erreur inference : {e}")
        finally:
            self.root.after(0, self._stop)

    def _upd_inference_ui(self, action, conf, probs, lnames):
        if action and action != "...":
            idx   = lnames.index(action) if action in lnames else 0
            color = ACTION_COLORS[idx % len(ACTION_COLORS)]
            self.action_big.config(text=action.upper(), fg=color)
            self.conf_lbl.config(text=f"Confiance : {conf*100:.1f}%", fg=TEXT)
        else:
            self.action_big.config(text="—", fg=TEXT_DIM)
            self.conf_lbl.config(text="En attente...", fg=TEXT_DIM)

        for i, act in enumerate(lnames):
            if act not in self.prob_bars: continue
            bar, bar_bg, color = self.prob_bars[act]
            p  = float(probs[i]) if i < len(probs) else 0.0
            bw = bar_bg.winfo_width() or 160
            bar.place(x=0, y=0, height=14, width=int(p*bw))
            self.prob_pcts[act].config(text=f"{p*100:.0f}%")

        for w in self.hist_frame.winfo_children(): w.destroy()
        for a in list(self.action_hist)[-12:]:
            idx   = lnames.index(a) if a in lnames else 0
            color = ACTION_COLORS[idx % len(ACTION_COLORS)]
            tk.Label(self.hist_frame, text=a, bg=color, fg="#000000",
                     font=FONT_SMALL, padx=5, pady=2).pack(side="left", padx=2, pady=2)

    def on_closing(self):
        self.running = False
        time.sleep(0.2)
        self.root.destroy()


def main():
    root = tk.Tk()
    app  = ActionRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"1280x760+{(sw-1280)//2}+{(sh-760)//2}")
    root.mainloop()

if __name__ == "__main__":
    main()
