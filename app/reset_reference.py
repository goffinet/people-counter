"""
Demande au pipeline (main.py) de recapturer la frame de référence porte.

Crée un fichier signal que main.py détecte à la prochaine frame traitée
(< 1 s). Pas d'accès caméra ni réseau — fonctionne pendant que le
pipeline tourne.

Usage depuis l'hôte :
  docker compose exec app python /app/reset_reference.py
"""

import sys
import time
from pathlib import Path

RESET_SIGNAL = "/data/.reset_reference"
REF_PATH     = "/data/door_reference.pkl"

print("Assurez-vous que la porte est FERMÉE. Recapture dans 3s...")
time.sleep(3)

mtime_before = Path(REF_PATH).stat().st_mtime if Path(REF_PATH).exists() else None

Path(RESET_SIGNAL).touch()
print("Signal envoyé — en attente de la confirmation du pipeline...", end="", flush=True)

for _ in range(30):   # attente max 3 s
    time.sleep(0.1)
    if not Path(RESET_SIGNAL).exists():
        mtime_after = Path(REF_PATH).stat().st_mtime if Path(REF_PATH).exists() else None
        if mtime_after != mtime_before:
            print(" OK")
            print(f"Nouvelle référence sauvegardée dans {REF_PATH}")
            sys.exit(0)

print(" TIMEOUT")
print("Le pipeline n'a pas répondu dans les 3 s — vérifiez que main.py tourne.")
sys.exit(1)
