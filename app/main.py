"""
People counter + door status detector — code partagé
ASUS IoT PE1103N — JetPack 6

Ce fichier est la base commune aux deux stacks de détection.
Merger une branche feature pour obtenir un pipeline complet :
  feature/yolo     → YOLOv8n + ByteTrack (Ultralytics)
  feature/nanoowl  → OWL-ViT TensorRT + tracker IoU (NanoOWL)

Configuration :
  LINE_Y          : position de la ligne de comptage (0.0 à 1.0, fraction de hauteur)
  DOOR_ROI        : zone d'intérêt de la porte (x1, y1, x2, y2 en fractions)
  DOOR_THRESHOLD  : sensibilité détection porte (0-100, augmenter si fausses alarmes)
"""

import cv2
import sqlite3
import time
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB             = "/data/counts.db"

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
