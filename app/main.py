"""
People counter + door status detector
ASUS IoT PE1103N — JetPack 6 / NanoOWL (OWL-ViT TensorRT) / IoU Tracker

Configuration :
  LINE_Y          : position de la ligne de comptage (0.0 à 1.0, fraction de hauteur)
  DOOR_ROI        : zone d'intérêt de la porte (x1, y1, x2, y2 en fractions)
  DOOR_THRESHOLD  : sensibilité détection porte (0-100, augmenter si fausses alarmes)
  DETECT_PROMPTS  : texte libre décrivant ce qu'on détecte (ex. "a person")
  DETECT_THRESHOLD: score minimal OWL-ViT pour valider une détection (0.0-1.0)
"""

import cv2
import sqlite3
import time
import numpy as np
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB               = "/data/counts.db"
MODEL_NAME       = "google/owlvit-base-patch32"
ENGINE_PATH      = "/data/owl_image_encoder_patch32.engine"  # pré-construit par build_engine.py
DETECT_PROMPTS   = ["a person"]
DETECT_THRESHOLD = 0.1

LINE_Y           = 0.5
DOOR_ROI         = (0.2, 0.1, 0.8, 0.9)
DOOR_THRESHOLD   = 25
REF_DELAY        = 3

# ---------------------------------------------------------------------------
# Tracker IoU simple (remplace ByteTrack intégré à Ultralytics)
# ---------------------------------------------------------------------------

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Tracker greedy par recouvrement IoU entre détections successives."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.tracks = {}
        self.next_id = 1
        self.iou_thr = iou_threshold
        self.max_age = max_age

    def update(self, detections: list) -> list:
        track_ids   = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]

        assignments  = {}
        matched_trks = set()

        if track_boxes and detections:
            iou_mat = np.array([[_iou(d, tb) for tb in track_boxes]
                                for d in detections])
            while iou_mat.max() >= self.iou_thr:
                i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                if i not in assignments and j not in matched_trks:
                    assignments[i] = track_ids[j]
                    matched_trks.add(j)
                iou_mat[i, :] = 0
                iou_mat[:, j] = 0

        to_delete = []
        for j, tid in enumerate(track_ids):
            if j not in matched_trks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        result = []
        for i, box in enumerate(detections):
            if i in assignments:
                tid = assignments[i]
            else:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"box": list(box), "age": 0}
            result.append((tid, box))

        return result

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
    print(f"[INFO] Chargement du modèle NanoOWL ({MODEL_NAME})...")
    predictor = OwlPredictor(MODEL_NAME, image_encoder_engine=ENGINE_PATH)

    print(f"[INFO] Encodage des prompts : {DETECT_PROMPTS}")
    text_encodings = predictor.encode_text(DETECT_PROMPTS)

    tracker = IouTracker()

    print("[INFO] Ouverture de la caméra...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir /dev/video0")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    con  = init_db()
    prev = {}
    last_door_status = None

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

        # --- Détection personnes (NanoOWL) ---
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = predictor.predict(
            image=pil_frame,
            text=DETECT_PROMPTS,
            text_encodings=text_encodings,
            threshold=DETECT_THRESHOLD,
        )
        detections = output.boxes.tolist() if len(output.boxes) > 0 else []

        # --- Tracking IoU ---
        tracks = tracker.update(detections)

        # --- Comptage des passages ---
        for tid, box in tracks:
            cy = int((box[1] + box[3]) / 2)

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
