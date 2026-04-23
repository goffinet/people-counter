"""
Recapture la frame de référence (porte fermée) sans redémarrer le conteneur.
Utile si l'éclairage a changé et génère de fausses alarmes.

Récupère la frame via le serveur MJPEG de main.py (port 8080) — la caméra
n'est pas ouverte directement (elle est déjà tenue par le pipeline).

Usage depuis l'hôte :
  docker compose exec app python /app/reset_reference.py
"""

import pickle
import time
import urllib.request

import cv2
import numpy as np

DOOR_ROI       = (0.2, 0.1, 0.8, 0.9)
REF_PATH       = "/data/door_reference.pkl"
SNAPSHOT_URL   = "http://localhost:8080/snapshot"
CAPTURE_DELAY  = 3   # secondes pour se préparer

print(f"Assurez-vous que la porte est FERMÉE. Capture dans {CAPTURE_DELAY}s...")
time.sleep(CAPTURE_DELAY)

try:
    with urllib.request.urlopen(SNAPSHOT_URL, timeout=5) as resp:
        jpeg_bytes = resp.read()
except Exception as e:
    raise RuntimeError(
        f"Impossible de joindre {SNAPSHOT_URL} : {e}\n"
        "Vérifiez que le pipeline (main.py) tourne bien dans ce conteneur."
    )

frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
if frame is None:
    raise RuntimeError("Snapshot reçu mais décodage JPEG échoué.")

h, w = frame.shape[:2]
x1, y1 = int(DOOR_ROI[0] * w), int(DOOR_ROI[1] * h)
x2, y2 = int(DOOR_ROI[2] * w), int(DOOR_ROI[3] * h)
roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

with open(REF_PATH, "wb") as f:
    pickle.dump(roi_gray, f)

debug = frame.copy()
cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("/data/roi_check.png", debug)

print(f"Référence sauvegardée dans {REF_PATH}")
print("Image de vérification : /data/roi_check.png")
