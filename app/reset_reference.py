"""
Recapture la frame de référence (porte fermée) sans redémarrer le conteneur.
Utile si l'éclairage a changé et génère de fausses alarmes.

Usage depuis l'hôte :
  docker compose exec app python /app/reset_reference.py
"""

import cv2
import numpy as np
import pickle

DOOR_ROI  = (0.2, 0.1, 0.8, 0.9)
REF_PATH  = "/data/door_reference.pkl"

cap = cv2.VideoCapture(0)
print("Assurez-vous que la porte est FERMÉE. Capture dans 3s...")
import time; time.sleep(3)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Impossible de lire la caméra")

h, w = frame.shape[:2]
x1, y1 = int(DOOR_ROI[0]*w), int(DOOR_ROI[1]*h)
x2, y2 = int(DOOR_ROI[2]*w), int(DOOR_ROI[3]*h)
roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

with open(REF_PATH, "wb") as f:
    pickle.dump(roi_gray, f)

debug = frame.copy()
cv2.rectangle(debug, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.imwrite("/data/roi_check.png", debug)
print(f"Référence sauvegardée dans {REF_PATH}")
print("Image de vérification : /data/roi_check.png")
