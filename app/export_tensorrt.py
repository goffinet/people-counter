"""
Export YOLOv8n vers moteur TensorRT pour gain de performance (~2-3x).
À exécuter une seule fois après le premier démarrage.

Usage :
  docker compose exec app python /app/export_tensorrt.py

Puis modifier MODEL_PATH dans main.py :
  MODEL_PATH = "/data/yolov8n.engine"
"""

from ultralytics import YOLO
import shutil

print("[INFO] Chargement du modèle PyTorch...")
model = YOLO("/data/yolov8n.pt")

print("[INFO] Export TensorRT en cours (peut prendre 5-10 min sur Jetson Orin)...")
model.export(
    format="engine",
    device=0,
    half=True,        # FP16 pour maximiser les performances sur Ampere
    workspace=4,      # GB de VRAM alloués pour la compilation
)

shutil.copy("yolov8n.engine", "/data/yolov8n.engine")
print("[OK] Moteur exporté : /data/yolov8n.engine")
print("[INFO] Modifier MODEL_PATH dans main.py :")
print('       MODEL_PATH = "/data/yolov8n.engine"')
