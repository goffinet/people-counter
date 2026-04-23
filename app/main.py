"""
People counter + door status detector — YOLOv8n + ByteTrack (Ultralytics)
ASUS IoT PE1103N — JetPack 6

Configuration :
  LINE_Y          : position de la ligne de comptage (0.0 à 1.0, fraction de hauteur)
  DOOR_ROI        : zone d'intérêt de la porte (x1, y1, x2, y2 en fractions)
  DOOR_THRESHOLD  : sensibilité détection porte (0-100, augmenter si fausses alarmes)
  MODEL_PATH      : yolov8n.pt (PyTorch) ou /data/yolov8n.engine (TensorRT après export)
"""

import cv2
import sqlite3
import time
import pickle
import numpy as np
from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB             = "/data/counts.db"
MODEL_PATH     = "/data/yolov8n.pt"      # remplacer par /data/yolov8n.engine après export TRT
CAMERA_SOURCE  = 0                       # /dev/video0
REF_PATH       = "/data/door_reference.pkl"

LINE_Y         = 0.5                    # ligne de comptage à 50 % de la hauteur
DOOR_ROI       = (0.2, 0.1, 0.8, 0.9)  # ROI porte (à ajuster selon cadrage)
DOOR_THRESHOLD = 25                     # seuil diff frames (augmenter si éclairage variable)
REF_DELAY      = 3                      # secondes avant capture de la frame de référence

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
# Frame de référence porte
# ---------------------------------------------------------------------------

def load_or_capture_reference(frame: np.ndarray) -> np.ndarray:
    """Charge la référence persistée ou en capture une nouvelle depuis frame."""
    ref_file = Path(REF_PATH)
    if ref_file.exists():
        with open(ref_file, "rb") as f:
            ref = pickle.load(f)
        print(f"[INFO] Référence porte chargée depuis {REF_PATH}")
        return ref
    ref = get_roi_gray(frame, DOOR_ROI)
    with open(ref_file, "wb") as f:
        pickle.dump(ref, f)
    print(f"[INFO] Référence porte capturée et sauvegardée dans {REF_PATH}")
    return ref

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    db    = init_db()
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {CAMERA_SOURCE}")

    print(f"[INFO] Attente {REF_DELAY}s avant capture de la référence porte...")
    time.sleep(REF_DELAY)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Impossible de lire la première frame")

    reference = load_or_capture_reference(frame)
    save_roi_check(frame, DOOR_ROI)

    h = frame.shape[0]
    line_px = int(h * LINE_Y)

    prev_centers: dict[int, float] = {}  # track_id → y-centre de la frame précédente
    door_prev = ""

    print("[INFO] Pipeline démarré.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame manquante, tentative suivante...")
            time.sleep(0.05)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],        # classe 0 = personne (COCO)
            verbose=False,
        )

        boxes = results[0].boxes
        active_ids: set[int] = set()

        if boxes is not None and boxes.id is not None:
            ids   = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().numpy()

            for tid, xyxy in zip(ids, xyxys):
                active_ids.add(tid)
                cy = float((xyxy[1] + xyxy[3]) / 2)

                if tid in prev_centers:
                    prev_cy = prev_centers[tid]
                    # y croissant = descend dans l'image = entrée
                    if prev_cy < line_px <= cy:
                        direction = "in"
                    elif prev_cy > line_px >= cy:
                        direction = "out"
                    else:
                        direction = None

                    if direction:
                        db.execute(
                            "INSERT INTO events (ts, direction) VALUES (?, ?)",
                            (int(time.time()), direction),
                        )
                        db.commit()
                        print(f"[EVENT] {direction.upper()} — track {tid}")

                prev_centers[tid] = cy

        # Supprime les tracks qui ne sont plus visibles
        prev_centers = {k: v for k, v in prev_centers.items() if k in active_ids}

        # Détection porte — ne loggue que les changements d'état
        door_status = detect_door(frame, reference, DOOR_ROI)
        if door_status != door_prev:
            db.execute(
                "INSERT INTO door_status (ts, status) VALUES (?, ?)",
                (int(time.time()), door_status),
            )
            db.commit()
            print(f"[DOOR] {door_status.upper()}")
            door_prev = door_status

    cap.release()
    db.close()


if __name__ == "__main__":
    main()
