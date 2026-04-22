# People Counter — ASUS IoT PE1103N

Système de comptage de personnes et de surveillance de porte par caméra USB,
déployé en conteneurs Docker sur ASUS IoT PE1103N (NVIDIA Jetson Orin, JetPack 6.x).

---

## Branches

| Branche | Stack de détection | Image Docker |
| --- | --- | --- |
| `feature/yolo` | YOLOv8n + ByteTrack (Ultralytics) | `ultralytics/ultralytics:latest-jetson-jetpack6` |
| `feature/nanoowl` | OWL-ViT TensorRT + tracker IoU (NanoOWL) | `dustynv/nanoowl:r36.4.0` |
| `feature/deepstream` | PeopleNet v2.6 + NvDCF (DeepStream 7) | `nvcr.io/nvidia/deepstream:7.0-samples` |

`main` est la branche d'intégration : elle contient le code partagé (détection porte, base de données, dashboard Grafana). Merger une branche `feature/*` pour obtenir une stack complète et déployable.

---

## Comparaison des stacks de détection

### Performance et complexité

| Critère | `feature/yolo` | `feature/nanoowl` | `feature/deepstream` |
| --- | --- | --- | --- |
| FPS (Jetson Orin) | ~30 (PyTorch) / ~60 (TRT) | ~8-15 | le plus élevé (pipeline GPU INT8) |
| Démarrage à froid | immédiat | construction moteur TRT (~15 min) | compilation moteur (~5 min) |
| Complexité du code | faible (Python + OpenCV) | moyenne (Python + OpenCV + HF) | élevée (GStreamer + DeepStream + pyds) |
| Accès caméra | OpenCV `VideoCapture` | OpenCV `VideoCapture` | GStreamer `v4l2src` (exclusif) |

### Détection et tracking

| Critère | `feature/yolo` | `feature/nanoowl` | `feature/deepstream` |
| --- | --- | --- | --- |
| Modèle | YOLOv8n (COCO) | OWL-ViT base patch32 | PeopleNet v2.6 (ResNet34) |
| Vocabulaire | fixe — classe `person` (COCO) | **ouvert — prompt texte libre** | fixe — `person / bag / face` |
| Réentraînement | non | **non** (prompt suffisant) | non |
| Tracker | ByteTrack (intégré Ultralytics) | IoU greedy (minimal) | **NvDCF (production, robuste aux occlusions)** |
| Précision piétons | bonne | variable selon threshold | **optimisée** (modèle dédié) |

### Quand choisir quelle branche

**`feature/yolo`** — le point de départ naturel.
Déploiement immédiat, écosystème bien documenté, performances suffisantes pour la majorité des installations. Exporter le moteur TensorRT (`export_tensorrt.py`) pour doubler les FPS en production.

**`feature/nanoowl`** — si le vocabulaire "personne" ne suffit pas.
Permet de détecter par description textuelle sans réentraîner de modèle : `"a person in a hard hat"`, `"a security agent"`, etc. Contrepartie : FPS plus faible et étape de build obligatoire avant le premier démarrage.

**`feature/deepstream`** — pour la production et la haute fiabilité.
Pipeline entièrement GPU (GStreamer + TensorRT INT8), modèle PeopleNet spécialement entraîné pour la détection de piétons, tracker NvDCF robuste aux occultations partielles. Architecture plus complexe, mais la plus proche d'un déploiement industriel.

---

## Fonctionnalités communes

- Détection visuelle de l'état de la porte (ouverte / fermée) par différence de frames
- Comptage entrées / sorties par ligne virtuelle
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
│   └── main.py                # code partagé (détection porte, base de données)
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

### 1. Choisir une branche feature

```bash
# Stack YOLOv8 + ByteTrack
git checkout feature/yolo

# Stack NanoOWL (OWL-ViT TensorRT)
git checkout feature/nanoowl

# Stack DeepStream / PeopleNet / NvDCF
git checkout feature/deepstream
```

### 2. Vérifier le runtime NVIDIA

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
cat /etc/docker/daemon.json
# doit contenir : "default-runtime": "nvidia"
```

### 3. Vérifier l'accès à la caméra

```bash
ls -la /dev/video0
sudo usermod -aG video $USER
# se déconnecter / reconnecter pour appliquer le groupe
```

### 4. Démarrer la stack

```bash
docker compose pull
docker compose up -d
```

Grafana est accessible sur `http://<ip-du-pe1103n>:3000`

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

Une image de vérification `/data/roi_check.png` est générée automatiquement au démarrage par chaque branche feature (rectangle vert = ROI porte, ligne rouge = ligne de comptage). Consulter le README de la branche concernée pour la procédure de recapture.

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

## Maintenance

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
| --- | --- |
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
