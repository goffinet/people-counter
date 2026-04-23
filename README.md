# People Counter — ASUS IoT PE1103N

Système de comptage de personnes et de surveillance de porte par caméra USB,
déployé en conteneurs Docker sur ASUS IoT PE1103N (NVIDIA Jetson Orin, JetPack 6.x).

Stack de détection : **YOLOv8n + ByteTrack (Ultralytics)**

---

## Branches

| Branche | Stack de détection | Image Docker |
| --- | --- | --- |
| **`feature/yolo`** | **YOLOv8n + ByteTrack (Ultralytics)** | **`ultralytics/ultralytics:latest-jetson-jetpack6`** |
| `feature/nanoowl` | OWL-ViT TensorRT + tracker IoU (NanoOWL) | `dustynv/nanoowl:r36.4.0` |
| `feature/deepstream` | PeopleNet v2.6 + NvDCF (DeepStream 7) | `nvcr.io/nvidia/deepstream:7.0-samples` |

`main` est la branche d'intégration neutre (code partagé uniquement).

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

## Fonctionnalités

- Comptage entrées / sorties par ligne virtuelle (YOLOv8n + ByteTrack)
- Détection visuelle de l'état de la porte (ouverte / fermée) par différence de frames
- Persistance des événements dans SQLite
- Dashboard Grafana temps réel (rafraîchissement 5s)
- Alertes email : porte ouverte > 5 min, salle en suroccupation
- Visualisation live pour calibrage (flux MJPEG sur port 8080)

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
│   ├── main.py                # pipeline principal YOLOv8 + ByteTrack
│   ├── debug_view.py          # visualisation live pour calibrage (port 8080)
│   ├── reset_reference.py     # recapture frame de référence porte
│   └── export_tensorrt.py     # export moteur TensorRT (optionnel)
└── grafana/
    ├── plugins/
    │   └── frser-sqlite-datasource/   # plugin SQLite pré-installé (hors ligne)
    └── provisioning/
        ├── datasources/sqlite.yaml
        ├── dashboards/
        │   ├── dashboard.yaml
        │   └── people_counter.json
        ├── alerting/door_alert.yaml
        ├── plugins/
        └── notifiers/
```

---

## Installation et mise en service

### 1. Choisir la branche feature

```bash
git checkout feature/yolo
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

> Le plugin Grafana SQLite (`frser-sqlite-datasource`) est fourni dans `grafana/plugins/`
> et chargé localement — aucun accès internet requis pour Grafana.

### 5. Calibrer la ligne de comptage et le ROI porte

Avant de valider le déploiement, vérifier que `LINE_Y` et `DOOR_ROI` correspondent
au cadrage réel de la caméra.

**Lancer le serveur de visualisation :**

```bash
docker compose exec -d app python /app/debug_view.py
```

**Ouvrir dans un navigateur :**

```
http://<ip-du-pe1103n>:8080
```

Le flux live affiche :
- **Ligne rouge** = position de `LINE_Y` (ligne virtuelle de comptage)
- **Rectangle vert** = zone `DOOR_ROI` (zone analysée pour détecter la porte)

**Tester d'autres valeurs sans modifier le code :**

```bash
docker compose exec -d app python /app/debug_view.py --line-y 0.4 --roi 0.15,0.05,0.85,0.95
```

**Récupérer une image fixe :**

```bash
curl http://<ip-du-pe1103n>:8080/snapshot -o snapshot.jpg
```

Une fois les valeurs correctes identifiées, les reporter dans `app/main.py` :

```python
LINE_Y   = 0.4
DOOR_ROI = (0.15, 0.05, 0.85, 0.95)
```

Puis redémarrer le pipeline :

```bash
docker compose restart app
```

### 6. Capturer la référence porte (porte fermée)

La frame de référence sert à détecter si la porte est ouverte ou fermée.
Elle est capturée automatiquement au premier démarrage.
Pour la recapturer manuellement (après un changement d'éclairage par exemple) :

```bash
# S'assurer que la porte est bien FERMÉE avant d'exécuter
docker compose exec app python /app/reset_reference.py
```

La nouvelle référence est sauvegardée dans `/data/door_reference.pkl`
et une image de vérification dans `/data/roi_check.png`.

### 7. Vérifier le dashboard Grafana

Grafana est accessible sur `http://<ip-du-pe1103n>:3000`

Le dashboard **"Compteur de personnes"** s'affiche directement (accès anonyme activé).
Il se rafraîchit toutes les 5 secondes.

---

## Configuration

### Ligne de comptage (`LINE_Y`)

Position de la ligne virtuelle en fraction de la hauteur de l'image.

```python
LINE_Y = 0.5   # 0.5 = milieu de l'image
```

Une personne qui descend (y croissant) en franchissant la ligne est comptée **entrée**.
Une personne qui monte (y décroissant) est comptée **sortie**.
Ajuster selon l'orientation de la caméra. Utiliser `debug_view.py` pour visualiser.

### Zone de détection porte (`DOOR_ROI`)

Coordonnées en fractions de l'image `(x1, y1, x2, y2)`.

```python
DOOR_ROI = (0.2, 0.1, 0.8, 0.9)  # valeur par défaut
```

Utiliser `debug_view.py` pour vérifier que le rectangle vert couvre bien la porte.

### Sensibilité détection porte (`DOOR_THRESHOLD`)

```python
DOOR_THRESHOLD = 25   # valeur par défaut
```

- Augmenter (40-60) si l'éclairage est variable → moins de fausses alarmes
- Diminuer (10-15) si la porte est peu visible dans l'image

### Modèle de détection (`MODEL_PATH`)

```python
MODEL_PATH = "yolov8n.pt"           # PyTorch — téléchargé au premier démarrage
MODEL_PATH = "/data/yolov8n.engine" # TensorRT — après export (voir section Optimisation)
```

### Seuils d'alerte Grafana

Dans `grafana/provisioning/alerting/door_alert.yaml` :

```yaml
for: 5m          # durée avant alerte porte ouverte
params: [20]     # capacité max de la salle
```

Dans le dashboard (`people_counter.json`, panel "Présence actuelle estimée") :

```json
{ "color": "yellow", "value": 10 },
{ "color": "red",    "value": 20 }
```

---

## Optimisation TensorRT (optionnel)

L'export TensorRT permet de passer de ~30 FPS à ~60 FPS sur Jetson Orin.
À effectuer une seule fois après le premier démarrage réussi.

```bash
docker compose exec app python /app/export_tensorrt.py
```

La compilation dure 5 à 10 minutes. Une fois terminée, modifier `MODEL_PATH`
dans `app/main.py` :

```python
MODEL_PATH = "/data/yolov8n.engine"
```

Puis redémarrer :

```bash
docker compose restart app
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

Modifier l'adresse destinataire dans `grafana/provisioning/alerting/door_alert.yaml` :

```yaml
addresses: admin@votredomaine.com
```

Redémarrer le dashboard pour appliquer :

```bash
docker compose restart dashboard
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
0 3 * * 0 docker compose -f /chemin/vers/docker-compose.yml exec -T dashboard \
  sqlite3 /data/counts.db "VACUUM"
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
| `yolov8n.pt` ne se télécharge pas | Le conteneur n'a pas accès internet — copier manuellement : `docker cp yolov8n.pt people-counter-app-1:/usr/src/app/` |
| Grafana ne démarre pas | Vérifier les logs : `docker compose logs dashboard` — si erreur plugin, le répertoire `grafana/plugins/` doit être présent |
| Grafana vide au démarrage | Attendre 15-20s, puis rafraîchir |
| Fausses alarmes porte | Augmenter `DOOR_THRESHOLD` ou recapturer la référence |
| Double comptage | Ajuster `LINE_Y` avec `debug_view.py` |
| Dashboard Grafana corrompu | Supprimer le volume et redémarrer : `docker compose down -v && docker compose up -d` |

---

## Notes RGPD

- Aucune image n'est stockée — uniquement les timestamps et directions de passage
- La détection de porte ne stocke que l'état (open/closed) et l'horodatage
- Exposer Grafana uniquement sur le réseau local (ne pas ouvrir le port 3000 sur internet)
- Le port 8080 (visualisation live) est destiné à la phase de calibrage uniquement — le fermer en production si non nécessaire
