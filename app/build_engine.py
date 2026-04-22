"""
Construit le moteur TensorRT pour NanoOWL (OWL-ViT image encoder).
À exécuter une seule fois avant le premier démarrage du service principal.

Usage :
  docker compose exec app python /app/build_engine.py

Durée : 10-20 min sur Jetson Orin (compilation TensorRT FP16).
Le fichier résultant est persisté dans le volume /data.
"""

import subprocess
import sys

MODEL_NAME  = "google/owlvit-base-patch32"
ENGINE_PATH = "/data/owl_image_encoder_patch32.engine"

print(f"[INFO] Construction du moteur TensorRT pour {MODEL_NAME}...")
print(f"[INFO] Destination : {ENGINE_PATH}")
print("[INFO] Cela peut prendre 10-20 min sur Jetson Orin...")

subprocess.run(
    [
        sys.executable, "-m", "nanoowl.build_image_encoder_engine",
        ENGINE_PATH,
        f"--model_name={MODEL_NAME}",
    ],
    check=True,
)

print(f"[OK] Moteur TensorRT enregistré : {ENGINE_PATH}")
print("[INFO] Vous pouvez maintenant (re)démarrer le service principal.")
