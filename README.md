# People Counter — ASUS IoT PE1103N

Système de comptage de personnes et de surveillance de porte par caméra USB,
déployé en conteneurs Docker sur ASUS IoT PE1103N (NVIDIA Jetson Orin, JetPack 6.x).

---

## Fonctionnalités

- Comptage entrées / sorties par ligne virtuelle (YOLOv8n + ByteTrack)
- Détection visuelle de l'état de la porte (ouverte / fermée) par différence de frames
- Persistance des événements dans SQLite
- Dashboard Grafana temps réel (rafraîchissement 5s)
- Alertes email : porte ouverte > 5 min, salle en suroccupation

---

## Prérequis

| Élément | Version requise |
|---|---|
| ASUS IoT PE1103N | JetPack 6.x (L4T r36.x) |
| Docker Engine | ≥ 24 |
| NVIDIA Container Toolkit | ≥ 1.14 |
| Webcam USB | Trust Trino sur `/dev/video0` |

---

## Structure du projet

```
people-counter/
├── docker-compose.yml
├── app/
│   ├── main.py                # pipeline principal
│   ├── reset_reference.py     # recapture frame de référence
│   └── export_tensorrt.py     # export TensorRT (optionnel)
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── sqlite.yaml
        ├── dashboards/
        │   ├── dashboard.yaml
        │   └── people_counter.json
        └── alerting/
            └── door_alert.yaml
```

---

## Installation

### 1. Vérifier le runtime NVIDIA

```bash
# Configurer nvidia comme runtime par défaut
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

# Vérifier
cat /etc/docker/daemon.json
# doit contenir : "default-runtime": "nvidia"
```

### 2. Vérifier l'accès à la caméra

```bash
ls -la /dev/video0
sudo usermod -aG video $USER
# se déconnecter / reconnecter pour appliquer le groupe
```

### 3. Valider le GPU dans le conteneur

```bash
docker run --rm --runtime nvidia --device /dev/video0 \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  python3 -c "
import torch, cv2
print('CUDA:', torch.cuda.is_available())
print('GPU :', torch.cuda.get_device_name(0))
cap = cv2.VideoCapture(0)
print('Cam :', cap.isOpened())
cap.release()
"
# attendu : CUDA: True | GPU: Orin | Cam: True
```

### 4. Démarrer la stack

```bash
docker compose pull
docker compose up -d
```

Grafana est accessible sur `http://<ip-du-pe1103n>:3000`

Le plugin SQLite est installé automatiquement au premier démarrage (~30s).

---

## Configuration

### Ligne de comptage (`LINE_Y` dans `app/main.py`)

Position de la ligne virtuelle en fraction de la hauteur de l'image.

```python
LINE_Y = 0.5   # 0.5 = milieu de l'image
```

Une personne qui descend (y croissant) en franchissant la ligne est comptée **entrée**.
Une personne qui monte (y décroissant) est comptée **sortie**.
Ajuster selon l'orientation de votre caméra.

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

Récupérer `/data/roi_check.png` et vérifier que le rectangle vert encadre bien le battant.

### Sensibilité détection porte (`DOOR_THRESHOLD`)

```python
DOOR_THRESHOLD = 25   # valeur par défaut
```

- Augmenter (40-60) si l'éclairage est variable → moins de fausses alarmes
- Diminuer (10-15) si la porte est peu visible dans l'image

### Seuils d'alerte Grafana

Dans `grafana/provisioning/alerting/door_alert.yaml` :

```yaml
for: 5m          # durée avant alerte porte ouverte
params: [20]     # capacité max de la salle
```

Dans le dashboard (`people_counter.json`, panel "Présence actuelle estimée") :

```json
{ "color": "yellow", "value": 10 },   // alerte jaune
{ "color": "red",    "value": 20 }    // alerte rouge
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

## Performance optionnelle : export TensorRT

Gain de 2-3× en débit d'inférence. À faire une seule fois.

```bash
docker compose exec app python /app/export_tensorrt.py
```

Puis modifier `MODEL_PATH` dans `app/main.py` :

```python
MODEL_PATH = "/data/yolov8n.engine"
```

```bash
docker compose restart app
```

---

## Maintenance

### Recapturer la frame de référence (porte fermée)

Utile si l'éclairage a changé et génère de fausses alarmes.

```bash
# Assurez-vous que la porte est fermée avant de lancer
docker compose exec app python /app/reset_reference.py
```

### Vérifier les logs

```bash
docker compose logs -f app        # pipeline vision
docker compose logs -f dashboard  # Grafana
```

### Compter les événements depuis la base

```bash
docker compose exec dashboard \
  sqlite3 /data/counts.db \
  "SELECT direction, count(*) FROM events GROUP BY direction;"
```

### Nettoyage SQLite hebdomadaire (optionnel)

Ajouter en cron sur l'hôte :

```bash
# crontab -e
0 3 * * 0 docker compose -f /chemin/vers/docker-compose.yml exec -T app \
  python3 -c "import sqlite3; sqlite3.connect('/data/counts.db').execute('VACUUM')"
```

### Vérifier la persistance après reboot

```bash
sudo reboot
# après redémarrage
docker compose ps   # doit afficher "running" pour app et dashboard
```

---

## Dépannage

| Symptôme | Solution |
|---|---|
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
