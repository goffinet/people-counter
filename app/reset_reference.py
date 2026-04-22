"""
Recapture la frame de référence porte sans redémarrer le service.
Envoie SIGUSR1 au processus principal via le fichier /data/app.pid.

Usage :
  docker compose exec app python /app/reset_reference.py

Prérequis : l'application doit être en cours d'exécution et la porte fermée.
"""

import os
import signal

PID_FILE = "/data/app.pid"

if not os.path.exists(PID_FILE):
    print("[ERREUR] /data/app.pid introuvable — l'application est-elle démarrée ?")
    raise SystemExit(1)

pid = int(open(PID_FILE).read().strip())
try:
    os.kill(pid, signal.SIGUSR1)
    print(f"[OK] Signal SIGUSR1 envoyé au processus {pid}.")
    print("[INFO] La référence sera recapturée sur la prochaine frame de l'appsink.")
except ProcessLookupError:
    print(f"[ERREUR] Aucun processus avec le PID {pid}. Redémarrez le service.")
    raise SystemExit(1)
