"""
Serveur de visualisation standalone pour calibrer LINE_Y et DOOR_ROI.

NOTE : main.py intègre déjà un serveur MJPEG sur le port DEBUG_PORT (8080).
       Ce script est utile uniquement quand le pipeline est ARRÊTÉ
       (main.py détient la caméra exclusivement).

Utilisation :
  docker compose stop app
  docker compose run --rm -p 8080:8080 app python /app/debug_view.py --line-y 0.4
  # ouvrir http://<ip>:8080
  docker compose up -d app   # relancer le pipeline

Ouvrir http://<ip-du-pe1103n>:8080 dans un navigateur pour voir :
  - rectangle VERT  = DOOR_ROI
  - ligne ROUGE     = LINE_Y
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Valeurs par défaut (identiques à main.py — modifier ici pour tester)
# ---------------------------------------------------------------------------
DEFAULT_LINE_Y        = 0.5
DEFAULT_DOOR_ROI      = (0.2, 0.1, 0.8, 0.9)
DEFAULT_CAMERA_SOURCE = 0
DEFAULT_PORT          = 8080

# ---------------------------------------------------------------------------
# Buffer partagé entre la boucle caméra et le serveur HTTP
# ---------------------------------------------------------------------------
_lock   = threading.Lock()
_frame  = None   # dernière frame annotée (BGR numpy array)


def annotate(frame: np.ndarray, line_y: float, roi: tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    # Ligne de comptage (rouge)
    ly = int(h * line_y)
    cv2.line(out, (0, ly), (w, ly), (0, 0, 255), 2)
    cv2.putText(out, f"LINE_Y={line_y:.2f}", (10, ly - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ROI porte (vert)
    x1, y1 = int(roi[0] * w), int(roi[1] * h)
    x2, y2 = int(roi[2] * w), int(roi[3] * h)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"DOOR_ROI={roi}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return out


def capture_loop(camera: int, line_y: float, roi: tuple):
    global _frame
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {camera}")

    while True:
        ok, frame = cap.read()
        if ok:
            annotated = annotate(frame, line_y, roi)
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _lock:
                _frame = jpeg.tobytes()
        time.sleep(0.05)   # ~20 fps max


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence les logs HTTP

    def do_GET(self):
        if self.path == "/snapshot":
            self._serve_snapshot()
        else:
            self._serve_stream()

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _lock:
                    jpeg = _frame
                if jpeg:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_snapshot(self):
        with _lock:
            jpeg = _frame
        if jpeg is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-y",  type=float,       default=DEFAULT_LINE_Y)
    parser.add_argument("--roi",     type=str,          default=None,
                        help="x1,y1,x2,y2 en fractions ex: 0.2,0.1,0.8,0.9")
    parser.add_argument("--camera",  type=int,          default=DEFAULT_CAMERA_SOURCE)
    parser.add_argument("--port",    type=int,          default=DEFAULT_PORT)
    args = parser.parse_args()

    roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else DEFAULT_DOOR_ROI

    print(f"[INFO] LINE_Y    = {args.line_y}")
    print(f"[INFO] DOOR_ROI  = {roi}")
    print(f"[INFO] Flux MJPEG disponible sur http://0.0.0.0:{args.port}")
    print(f"[INFO] Snapshot  disponible sur http://0.0.0.0:{args.port}/snapshot")

    t = threading.Thread(target=capture_loop, args=(args.camera, args.line_y, roi), daemon=True)
    t.start()

    time.sleep(1)   # laisse la caméra s'initialiser
    HTTPServer(("0.0.0.0", args.port), MJPEGHandler).serve_forever()


if __name__ == "__main__":
    main()
