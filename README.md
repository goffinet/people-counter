# People Counter — ASUS IoT PE1103N

Système de comptage de personnes et de surveillance de porte par caméra USB,
déployé en conteneurs Docker sur ASUS IoT PE1103N (NVIDIA Jetson Orin, JetPack 6.x).

Stack de détection : **NanoOWL / OWL-ViT TensorRT / tracker IoU**

---

## Branches

| Branche | Stack de détection | Image Docker |
| --- | --- | --- |
| `feature/yolo` | YOLOv8n + ByteTrack (Ultralytics) | `ultralytics/ultralytics:latest-jetson-jetpack6` |
| **`feature/nanoowl`** | **OWL-ViT TensorRT + tracker IoU (NanoOWL)** | **`dustynv/nanoowl:r36.4.0`** |
| `feature/deepstream` | PeopleNet v2.6 + NvDCF (DeepStream 7) | `nvcr.io/nvidia/deepstream:7.0-samples` |

`main` est la branche d'intégration neutre (code partagé uniquement).

---

## Fonctionnalités

- Comptage entrées / sorties par ligne virtuelle (OWL-ViT + tracker IoU)
- Détection par prompt texte libre (`"a person"`, modifiable sans réentraînement)
- Détection visuelle de l'état de la porte (ouverte / fermée) par différence de frames
- Persistance des événements dans SQLite
- Dashboard Grafana temps réel (rafraîchissement 5s)
- Alertes email : porte ouverte > 5 min, salle en suroccupation

---

## Prérequis

| Élément | Version requise |
| --- | --- |
| ASUS IoT PE1103N | JetPack 6.x (L4T r36.x) |
| Docker Engine | ≥ 24 |
| NVIDIA Container Toolkit | ≥ 1.14 |
| Webcam USB | Trust Trino sur `/dev/video0` |

---

## Structure du projet

```text
people-counter/
├── docker-compose.yml
├── app/
│   ├── main.py                # pipeline principal NanoOWL + IoU tracker
│   ├── build_engine.py        # compilation moteur TensorRT (une seule fois)
│   └── reset_reference.py     # recapture frame de référence porte
└── grafana/
    └── provisioning/
        ├── datasources/sqlite.yaml
        ├── dashboards/
        │   ├── dashboard.yaml
        │   └── people_counter.json
        └── alerting/door_alert.yaml
```

---

## Installation

### 1. Vérifier le runtime NVIDIA

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
cat /etc/docker/daemon.json
# doit contenir : "default-runtime": "nvidia"
```

### 2. Vérifier l'accès à la caméra

```bash
ls -la /dev/video0
sudo usermod -aG video $USER
# se déconnecter / reconnecter pour appliquer le groupe
```

### 3. Construire le moteur TensorRT (une seule fois)

```bash
docker compose pull
docker compose run --rm app python /app/build_engine.py
```

Durée : 10-20 min sur Jetson Orin. Le moteur est mis en cache dans `/data/` pour les démarrages suivants.

### 4. Démarrer la stack

```bash
docker compose up -d
```

Grafana est accessible sur `http://<ip-du-pe1103n>:3000`

---

## Configuration

### Prompt de détection (`DETECT_PROMPTS` dans `app/main.py`)

OWL-ViT détecte par description textuelle — aucun réentraînement nécessaire.

```python
DETECT_PROMPTS   = ["a person"]
DETECT_THRESHOLD = 0.1   # score minimal de confiance (0.0-1.0)
```

Augmenter `DETECT_THRESHOLD` (0.2-0.4) pour réduire les faux positifs.

### Ligne de comptage (`LINE_Y` dans `app/main.py`)

```python
LINE_Y = 0.5   # 0.5 = milieu de l'image
```

Une personne qui descend (y croissant) en franchissant la ligne est comptée **entrée**.
Une personne qui monte (y décroissant) est comptée **sortie**.

### Zone de détection porte (`DOOR_ROI`)

Coordonnées en fractions de l'image `(x1, y1, x2, y2)`.

```python
DOOR_ROI = (0.2, 0.1, 0.8, 0.9)  # valeur par défaut
```

Pour vérifier visuellement la zone :

```bash
docker compose exec app python3 -c "
import cv2
cap = cv2.VideoCapture(0)
_, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
roi = (0.2, 0.1, 0.8, 0.9)
cv2.rectangle(frame,
  (int(roi[0]*w), int(roi[1]*h)),
  (int(roi[2]*w), int(roi[3]*h)),
  (0,255,0), 2)
cv2.imwrite('/data/roi_check.png', frame)
"
```

### Sensibilité détection porte (`DOOR_THRESHOLD`)

```python
DOOR_THRESHOLD = 25
```

- Augmenter (40-60) si l'éclairage est variable
- Diminuer (10-15) si la porte est peu visible

### Seuils d'alerte Grafana

Dans `grafana/provisioning/alerting/door_alert.yaml` :

```yaml
for: 5m          # durée avant alerte porte ouverte
params: [20]     # capacité max de la salle
```

---

## Alertes email

Décommenter et adapter les variables SMTP dans `docker-compose.yml` :

```yaml
- GF_SMTP_ENABLED=true
- GF_SMTP_HOST=smtp.votreserveur.com:587
- GF_SMTP_USER=alerte@votredomaine.com
- GF_SMTP_PASSWORD=motdepasse
- GF_SMTP_FROM_ADDRESS=alerte@votredomaine.com
- GF_SMTP_FROM_NAME=Grafana PE1103N
```

Modifier l'adresse destinataire dans `door_alert.yaml` :

```yaml
addresses: admin@votredomaine.com
```

---

## Maintenance

### Recapturer la frame de référence (porte fermée)

```bash
# Assurez-vous que la porte est fermée avant de lancer
docker compose exec app python /app/reset_reference.py
```

### Vérifier les logs

```bash
docker compose logs -f app        # pipeline NanoOWL
docker compose logs -f dashboard  # Grafana
```

### Compter les événements depuis la base

```bash
docker compose exec dashboard \
  sqlite3 /data/counts.db \
  "SELECT direction, count(*) FROM events GROUP BY direction;"
```

### Nettoyage SQLite hebdomadaire (optionnel)

```bash
# crontab -e
0 3 * * 0 docker compose -f /chemin/vers/docker-compose.yml exec -T dashboard \
  sqlite3 /data/counts.db "VACUUM"
```

---

## Dépannage

| Symptôme | Solution |
| --- | --- |
| `ENGINE_PATH` introuvable au démarrage | Lancer `build_engine.py` avant le premier `up` |
| Détections manquées | Baisser `DETECT_THRESHOLD` (0.05) |
| Trop de faux positifs | Monter `DETECT_THRESHOLD` (0.3-0.5) |
| `CUDA: False` dans le conteneur | Vérifier `--runtime nvidia` et `daemon.json` |
| Caméra non détectée | Vérifier droits groupe `video` et `devices:` dans compose |
| Fausses alarmes porte | Augmenter `DOOR_THRESHOLD` ou recapturer la référence |
| Double comptage | Diminuer `LINE_Y` ou ajuster la hauteur caméra |
| Grafana vide au démarrage | Attendre 30-60s le temps de l'installation du plugin SQLite |

---

## Notes RGPD

- Aucune image n'est stockée — uniquement les timestamps et directions de passage
- La détection de porte ne stocke que l'état (open/closed) et l'horodatage
- Exposer Grafana uniquement sur le réseau local (ne pas ouvrir le port 3000 sur internet)
