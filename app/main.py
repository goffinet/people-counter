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
import threading
import numpy as np
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB             = "/data/counts.db"
MODEL_PATH     = "/data/yolov8n.pt"      # remplacer par /data/yolov8n.engine après export TRT
CAMERA_SOURCE  = 0                       # /dev/video0
REF_PATH       = "/data/door_reference.pkl"
RESET_SIGNAL   = "/data/.reset_reference"  # créé par reset_reference.py
DEBUG_PORT     = 8080                    # 0 pour désactiver le serveur de visualisation

LINE_Y              = 0.60               # ligne de comptage à 60 % de la hauteur
DOOR_ROI            = (0.2, 0.1, 0.9, 0.9)  # ROI porte (à ajuster selon cadrage)
DOOR_PIXEL_DIFF     = 30                 # diff par pixel pour le compter comme "changé" (0-255)
DOOR_THRESHOLD      = 0.08              # fraction de pixels changés pour déclarer "open" (0.0-1.0)
DOOR_HYSTERESIS     = 8                 # frames consécutives avant de valider un changement d'état
REF_DELAY      = 3                       # secondes avant capture de la frame de référence

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
    if current.shape != reference.shape:
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]))
    diff          = cv2.absdiff(reference, current)
    changed_ratio = float(np.sum(diff > DOOR_PIXEL_DIFF)) / diff.size
    return "open" if changed_ratio > DOOR_THRESHOLD else "closed"

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
# Serveur MJPEG de visualisation (port DEBUG_PORT)
# Accessible sur http://<ip>:8080 pendant que le pipeline tourne.
# ---------------------------------------------------------------------------

_debug_lock:  threading.Lock = threading.Lock()
_debug_frame: bytes | None   = None


def _annotate_debug(frame: np.ndarray, line_px: int) -> bytes:
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.line(out, (0, line_px), (w, line_px), (0, 0, 255), 2)
    cv2.putText(out, f"LINE_Y={LINE_Y:.2f}", (10, line_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    x1, y1 = int(DOOR_ROI[0] * w), int(DOOR_ROI[1] * h)
    x2, y2 = int(DOOR_ROI[2] * w), int(DOOR_ROI[3] * h)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"DOOR_ROI={DOOR_ROI}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    _, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpeg.tobytes()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/snapshot":
            self._snapshot()
        else:
            self._stream()

    def _snapshot(self):
        with _debug_lock:
            jpeg = _debug_frame
        if jpeg is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _debug_lock:
                    jpeg = _debug_frame
                if jpeg:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _start_debug_server(port: int) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[INFO] Visualisation live : http://0.0.0.0:{port}  (snapshot : /snapshot)")

# ---------------------------------------------------------------------------
# Chevauchement personne / ROI porte
# ---------------------------------------------------------------------------

def _person_overlaps_roi(boxes, roi: tuple, frame_shape: tuple) -> bool:
    """Retourne True si au moins une boîte YOLO chevauche le ROI porte."""
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return False
    h, w = frame_shape[:2]
    rx1, ry1 = int(roi[0] * w), int(roi[1] * h)
    rx2, ry2 = int(roi[2] * w), int(roi[3] * h)
    for xyxy in boxes.xyxy.cpu().numpy():
        bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        if bx1 < rx2 and bx2 > rx1 and by1 < ry2 and by2 > ry1:
            return True
    return False

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

    h, w   = frame.shape[:2]
    line_px = int(h * LINE_Y)

    if DEBUG_PORT:
        _start_debug_server(DEBUG_PORT)

    prev_centers: dict[int, float] = {}  # track_id → y-centre de la frame précédente
    door_prev        = ""
    door_candidate   = ""
    door_consec      = 0

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

        # Recapture de référence demandée par reset_reference.py
        if Path(RESET_SIGNAL).exists():
            reference = get_roi_gray(frame, DOOR_ROI)
            with open(REF_PATH, "wb") as f:
                pickle.dump(reference, f)
            Path(RESET_SIGNAL).unlink()
            print("[INFO] Référence porte recapturée.")

        # Détection porte — ignorée si une personne chevauche le ROI
        person_in_roi = _person_overlaps_roi(boxes, DOOR_ROI, frame.shape)
        raw_status = detect_door(frame, reference, DOOR_ROI) if not person_in_roi else door_prev or "closed"
        if raw_status == door_candidate:
            door_consec += 1
        else:
            door_candidate = raw_status
            door_consec    = 1
        if door_consec >= DOOR_HYSTERESIS and door_candidate != door_prev:
            db.execute(
                "INSERT INTO door_status (ts, status) VALUES (?, ?)",
                (int(time.time()), door_candidate),
            )
            db.commit()
            print(f"[DOOR] {door_candidate.upper()}")
            door_prev  = door_candidate
            door_consec = 0

        # Mise à jour du buffer de visualisation
        if DEBUG_PORT:
            global _debug_frame
            with _debug_lock:
                _debug_frame = _annotate_debug(frame, line_px)

    cap.release()
    db.close()


if __name__ == "__main__":
    main()
