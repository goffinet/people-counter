"""
People counter + door status detector
ASUS IoT PE1103N — JetPack 6 / YOLOv8n / ByteTrack

Configuration :
  LINE_Y        : position de la ligne de comptage (0.0 à 1.0, fraction de hauteur)
  DOOR_ROI      : zone d'intérêt de la porte (x1, y1, x2, y2 en fractions)
  DOOR_THRESHOLD: sensibilité détection porte (0-100, augmenter si fausses alarmes)
"""

import cv2
import sqlite3
import time
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB             = "/data/counts.db"
MODEL_PATH     = "yolov8n.pt"       # remplacer par yolov8n.engine après export TensorRT

LINE_Y         = 0.5                # ligne de comptage à 50% de la hauteur
DOOR_ROI       = (0.2, 0.1, 0.8, 0.9)  # ROI porte (à ajuster selon cadrage)
DOOR_THRESHOLD = 25                 # seuil diff frames (augmenter si éclairage variable)
REF_DELAY      = 3                  # secondes avant capture de la frame de référence

# ---------------------------------------------------------------------------
# Base de données
# ---------------------------------------------------------------------------

def init_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            ts        INTEGER,
            direction TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS door_status (
            ts     INTEGER,
            status TEXT
        )
    """)
    con.commit()
    return con

# ---------------------------------------------------------------------------
# Détection porte (différence de frames)
# ---------------------------------------------------------------------------

def get_roi_gray(frame: np.ndarray, roi: tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = int(roi[0] * w), int(roi[1] * h)
    x2, y2 = int(roi[2] * w), int(roi[3] * h)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

def detect_door(frame: np.ndarray, reference: np.ndarray, roi: tuple) -> str:
    current = get_roi_gray(frame, roi)
    diff    = cv2.absdiff(reference, current)
    score   = float(np.mean(diff))
    return "open" if score > DOOR_THRESHOLD else "closed"

# ---------------------------------------------------------------------------
# Utilitaire : sauvegarder une image de vérification du ROI
# ---------------------------------------------------------------------------

def save_roi_check(frame: np.ndarray, roi: tuple, path: str = "/data/roi_check.png"):
    h, w = frame.shape[:2]
    x1, y1 = int(roi[0] * w), int(roi[1] * h)
    x2, y2 = int(roi[2] * w), int(roi[3] * h)
    debug = frame.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(debug, (0, int(h * LINE_Y)), (w, int(h * LINE_Y)), (0, 0, 255), 2)
    cv2.imwrite(path, debug)
    print(f"[INFO] Image de vérification enregistrée : {path}")

# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def main():
    print("[INFO] Chargement du modèle YOLOv8...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Ouverture de la caméra...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir /dev/video0")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    con  = init_db()
    prev = {}               # {track_id: y_centre_précédent}
    last_door_status = None

    # Capture de la frame de référence (porte fermée)
    print(f"[INFO] Capture de la référence dans {REF_DELAY}s — porte doit être fermée...")
    time.sleep(REF_DELAY)
    ok, ref_frame = cap.read()
    if not ok:
        raise RuntimeError("Impossible de lire la caméra pour la frame de référence")
    reference_roi = get_roi_gray(ref_frame, DOOR_ROI)
    save_roi_check(ref_frame, DOOR_ROI)
    print("[INFO] Référence capturée. Démarrage de la détection.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame manquée, on continue...")
            continue

        h = frame.shape[0]
        threshold = int(h * LINE_Y)

        # --- Détection statut porte ---
        door = detect_door(frame, reference_roi, DOOR_ROI)
        if door != last_door_status:
            con.execute("INSERT INTO door_status VALUES (?,?)",
                        (int(time.time()), door))
            con.commit()
            print(f"[DOOR] Statut changé : {door}")
            last_door_status = door

        # --- Détection + tracking personnes ---
        results = model.track(frame, classes=[0], persist=True, verbose=False)

        if results[0].boxes.id is None:
            continue

        for box, tid in zip(results[0].boxes.xyxy,
                            results[0].boxes.id.int()):
            cy  = int((box[1] + box[3]) / 2)
            tid = int(tid)

            if tid in prev:
                if prev[tid] < threshold <= cy:
                    con.execute("INSERT INTO events VALUES (?,?)",
                                (int(time.time()), "in"))
                    con.commit()
                    print(f"[COUNT] Entrée — track {tid}")
                elif prev[tid] > threshold >= cy:
                    con.execute("INSERT INTO events VALUES (?,?)",
                                (int(time.time()), "out"))
                    con.commit()
                    print(f"[COUNT] Sortie — track {tid}")

            prev[tid] = cy


if __name__ == "__main__":
    main()
