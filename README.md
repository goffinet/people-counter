# People Counter — ASUS IoT PE1103N

Système de comptage de personnes et de surveillance de porte par caméra USB,
déployé en conteneurs Docker sur ASUS IoT PE1103N (NVIDIA Jetson Orin, JetPack 6.x).

Deux stacks de détection coexistent dans ce dépôt, sélectionnables via un profil Docker Compose.

---

## Choisir sa stack

```bash
docker compose --profile yolo        up -d   # YOLOv8n + ByteTrack
docker compose --profile deepstream  up -d   # PeopleNet v2.6 + NvDCF
```

| Critère | YOLO (`--profile yolo`) | DeepStream (`--profile deepstream`) |
| --- | --- | --- |
| Image Docker | build local (`Dockerfile`) | `nvcr.io/nvidia/deepstream:7.0-samples-multiarch (+ Dockerfile.deepstream)` |
| Modèle | YOLOv8n (COCO) | PeopleNet v2.6 (ResNet34, dédié piétons) |
| Tracker | ByteTrack (intégré Ultralytics) | NvDCF (production, robuste aux occlusions) |
| Pipeline | Python + OpenCV | GStreamer + pyds (GPU bout en bout) |
| FPS Jetson Orin | ~30 (PyTorch) / ~60 (TRT) | le plus élevé (INT8 GPU) |
| Démarrage à froid | immédiat | ~5 min (compilation moteur TRT) |
| Visualisation live | MJPEG sur port 8080 | image fixe `/data/roi_check.png` |
| Complexité | faible | élevée |

**Règle de choix :**
- Commencer par **YOLO** : déploiement immédiat, calibrage visuel live, écosystème bien documenté.
- Passer à **DeepStream** pour la production et la haute fiabilité : pipeline entièrement GPU, modèle PeopleNet spécialisé piétons, tracker NvDCF robuste aux occultations partielles.

---

## Prérequis communs

| Élément | Version requise |
| --- | --- |
| ASUS IoT PE1103N | JetPack 6.x (L4T r36.x) |
| Docker Engine | ≥ 24 |
| NVIDIA Container Toolkit | ≥ 1.14 |
| Webcam USB | Trust Trino sur `/dev/video0` |

### Vérifier le runtime NVIDIA

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
cat /etc/docker/daemon.json
# doit contenir : "default-runtime": "nvidia"
```

### Vérifier l'accès à la caméra

```bash
ls -la /dev/video0
sudo usermod -aG video $USER
# se déconnecter / reconnecter pour appliquer le groupe
```

---

## Structure du projet

```text
people-counter/
├── Dockerfile                         # image YOLO (build local)
├── docker-compose.yml                 # profils yolo / deepstream
├── wheels/
│   └── lapx-*.whl                     # dépendance ByteTrack (aarch64, hors ligne)
├── app/
│   ├── main.py                        # pipeline YOLO + ByteTrack + MJPEG (port 8080)
│   ├── deepstream_main.py             # pipeline DeepStream + GStreamer
│   ├── bytetrack_low.yaml             # config ByteTrack (seuils abaissés pour conf=0.10)
│   ├── reset_reference.py             # recapture frame de référence porte
│   ├── export_tensorrt.py             # export moteur TensorRT YOLO (optionnel)
│   └── config/                        # (DeepStream uniquement)
│       ├── pgie_peoplenet.txt         # config nvinfer PeopleNet
│       ├── tracker_nvdcf.yml          # config tracker NvDCF
│       └── peoplenet_labels.txt       # labels des classes
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

## Stack YOLO — YOLOv8n + ByteTrack

### Installation

#### 1. Télécharger le modèle YOLOv8n

Le conteneur n'a pas accès internet. Télécharger le modèle depuis l'hôte :

```bash
curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt \
     -o /tmp/yolov8n.pt
```

#### 2. Construire l'image et démarrer

```bash
docker compose --profile yolo build
docker compose --profile yolo up -d
docker cp /tmp/yolov8n.pt people-counter-app-yolo-1:/data/yolov8n.pt
```

> Le plugin Grafana SQLite (`frser-sqlite-datasource`) est fourni dans `grafana/plugins/`
> et chargé localement — aucun accès internet requis.

#### 3. Calibrer la ligne de comptage et le ROI porte

Le serveur de visualisation démarre automatiquement avec le pipeline.
Ouvrir dans un navigateur :

```
http://<ip-du-pe1103n>:8080
```

Le flux live affiche :
- **Ligne rouge** = position de `LINE_Y`
- **Rectangle vert** = zone `DOOR_ROI`
- **Boîtes oranges** = détections YOLO avec `id`, `cy` et `conf`

Récupérer une image fixe :

```bash
curl http://<ip-du-pe1103n>:8080/snapshot -o snapshot.jpg
```

Une fois les valeurs identifiées, les modifier dans `app/main.py`, puis redémarrer :

```bash
docker compose --profile yolo restart app-yolo
```

#### 4. Capturer la référence porte (porte fermée)

```bash
# S'assurer que la porte est bien FERMÉE avant d'exécuter
docker compose exec app-yolo python /app/reset_reference.py
```

#### 5. Vérifier le dashboard Grafana

```
http://<ip-du-pe1103n>:3000
```

---

### Configuration YOLO

#### Ligne de comptage (`LINE_Y`)

```python
LINE_Y = 0.79   # ligne à 79 % de la hauteur (calibrée sur cette installation)
```

| Montage caméra | `LINE_Y` recommandé |
| --- | --- |
| Face à la porte, vue couloir | 0.55 – 0.70 (à calibrer) |
| Au-dessus de la porte (vue plongeante) | 0.45 – 0.55 |
| Caméra basse regardant vers le haut | 0.30 – 0.45 |

#### Zone de détection porte (`DOOR_ROI`)

`DOOR_ROI = (x1, y1, x2, y2)` — fractions de l'image (0.0 à 1.0), origine en haut à gauche.
Toujours mettre `x1 < x2` et `y1 < y2`.

```python
DOOR_ROI = (0.6, 0.1, 0.8, 0.9)   # bande verticale 60–80 % de la largeur
```

#### Détection porte — paramètres fins

| Paramètre | Valeur | Rôle |
| --- | --- | --- |
| `DOOR_PIXEL_DIFF` | `30` | Différence minimale (0–255) pour qu'un pixel soit "changé" |
| `DOOR_THRESHOLD` | `0.35` | Fraction (0.0–1.0) de pixels changés pour déclarer la porte ouverte |
| `DOOR_HYSTERESIS` | `20` | Frames consécutives avant de valider un changement d'état |

#### Comptage conditionné à la porte (`COUNT_ONLY_DOOR_OPEN`)

```python
COUNT_ONLY_DOOR_OPEN = False   # compter en toutes circonstances (recommandé)
COUNT_ONLY_DOOR_OPEN = True    # ne compter que quand la porte est détectée ouverte
```

> **Attention avec `True` :** le délai `DOOR_HYSTERESIS` peut masquer les franchissements rapides.
> Voir la section Tuning pour le détail.

#### Seuil de confiance YOLO (`conf`)

```python
conf=0.10   # dans model.track() — app/main.py
```

La valeur par défaut Ultralytics est `0.25`, mais des vues latérales nécessitent souvent `0.10`–`0.15`.

#### Modèle (`MODEL_PATH`)

```python
MODEL_PATH = "/data/yolov8n.pt"     # PyTorch — copié dans /data au démarrage
MODEL_PATH = "/data/yolov8n.engine" # TensorRT — après export (voir section Optimisation)
```

---

### Comprendre la logique de détection YOLO

#### Système de coordonnées

OpenCV utilise un repère où **y=0 est en haut** et **y augmente vers le bas**.

```
y=0   ┌─────────────────────────────┐  ← haut de l'image
      │                             │
y=230 │  - - - - ligne rouge  - - - │  ← LINE_Y=0.48 → line_px=230
      │                             │
      │   ┌───────┐                 │
      │   │       │     YOLO bbox   │
y=302 │   │  ● cy │  ← cy = (by1+by2)/2
      │   └───────┘                 │
      │                             │
y=480 └─────────────────────────────┘  ← bas de l'image
```

#### Condition de franchissement

```
prev_cy < line_px  ET  cy >= line_px  →  "in"   (descend dans l'image)
prev_cy > line_px  ET  cy <= line_px  →  "out"  (remonte dans l'image)
```

#### Calibrer `LINE_Y` depuis les logs

```bash
docker compose logs app-yolo | grep "\[TRACK" | awk '{print $4}' | sort -t= -k2 -n
```

Méthode :

```
1. cy_min (personne apparaît dans le champ) → ex. 338
2. cy_max (personne sort du champ)          → ex. 426
3. LINE_Y = (338 + 426) / 2 / hauteur_image = 382 / 480 = 0.796 → 0.79
```

Vérification en temps réel :

```bash
docker compose logs -f app-yolo | grep -E "TRACK|EVENT"
# [TRACK 10:01:05.230] id=12 cy=290 line=302   ← pas encore passé
# [TRACK 10:01:05.263] id=12 cy=335 line=302   ← passé → EVENT
# [EVENT 10:01:05.263] IN — track 12
```

---

### Tuning de la détection YOLO — expériences de terrain

#### Symptôme : `tracks=0` malgré `détections=1`

ByteTrack possède des seuils internes indépendants du paramètre `conf` de YOLO.
Si les détections ont un score inférieur à `new_track_thresh` (défaut 0.25), ByteTrack refuse de créer un track.

```
[DBG 10:01:00.123] détections=1 tracks=0 conf=[0.11]
```

**Solution :** `app/bytetrack_low.yaml` abaisse ces seuils :

```yaml
track_high_thresh: 0.10   # défaut 0.25
new_track_thresh:  0.10   # défaut 0.25 — doit être ≤ conf passé à model.track()
track_buffer:      45     # défaut 30 — compense les détections intermittentes
```

#### Symptôme : personne trackée mais aucun `[EVENT]`

`cy` ne croise jamais `line_px`. Recalibrer `LINE_Y` avec la méthode ci-dessus.

**Exemple réel de cette installation :**

| Mesure | Valeur |
| --- | --- |
| `cy` à l'entrée dans le champ | 338 |
| `cy` à la sortie du champ | 426 |
| `line_px` optimal | 382 (LINE_Y = 0.79) |
| Conf détection | 0.88–0.94 (vue latérale stable) |

#### Symptôme : `COUNT_ONLY_DOOR_OPEN=True` bloque tous les passages

`DOOR_HYSTERESIS=25` ≈ 0.8 s → la personne franchit la ligne pendant la fenêtre d'attente.

```
[TRACK 09:25:10.200] id=14 cy=377 line=379   ← franchissement
[DOOR  09:25:10.650] OPEN                    ← trop tard (+450 ms)
→ EVENT bloqué car door_prev="closed"
```

**Solution :** garder `COUNT_ONLY_DOOR_OPEN = False`. Le filtrage des faux positifs repose sur `DOOR_THRESHOLD`.

#### Symptôme : la porte oscille OPEN/CLOSED en continu

La référence est obsolète (éclairage changé, caméra déplacée).

```bash
docker compose exec app-yolo python /app/reset_reference.py
```

Si l'oscillation persiste, `DOOR_THRESHOLD` est trop bas. Mesurer le ratio réel :

```python
# Ajouter temporairement dans detect_door() :
print(f"[DOOR_RATIO {_ts()}] {changed_ratio:.3f}")
```

Observer les valeurs porte fermée → fixer `DOOR_THRESHOLD` légèrement au-dessus du bruit.

#### Réglage `DOOR_THRESHOLD` / `DOOR_HYSTERESIS` selon l'environnement

| Environnement | `DOOR_PIXEL_DIFF` | `DOOR_THRESHOLD` | `DOOR_HYSTERESIS` |
| --- | --- | --- | --- |
| ROI étroit (≤ 20 % largeur), passages fréquents | 30 | 0.30–0.40 | 20–25 |
| ROI large (≥ 40 % largeur) | 30 | 0.15–0.25 | 15–20 |
| Éclairage variable (soleil direct) | 40–50 | 0.35–0.45 | 25–30 |
| Porte peu contrastée / vitrée | 15–20 | 0.20–0.30 | 20 |

#### Cas limite : plusieurs personnes simultanées

Le pipeline traite chaque track indépendamment — 1 sortie + 2 entrées génèrent bien 3 événements.
Limite : si deux personnes sont très proches, YOLO peut fusionner leurs bboxes. Vérifier `tracks=` dans `[DBG]`.

---

### Optimisation TensorRT (optionnel)

L'export TensorRT permet de passer de ~30 FPS à ~60 FPS sur Jetson Orin.

```bash
docker compose exec app-yolo python /app/export_tensorrt.py
```

La compilation dure 5–10 minutes. Ensuite, modifier `MODEL_PATH` dans `app/main.py` :

```python
MODEL_PATH = "/data/yolov8n.engine"
```

```bash
docker compose --profile yolo restart app-yolo
```

---

## Stack DeepStream — PeopleNet v2.6 + NvDCF

### Architecture du pipeline

```
v4l2src ─┬─ queue → videoconvert → appsink          [détection porte, CPU]
          └─ queue → nvvideoconvert → nvinfer
                     → nvtracker → probe             [comptage personnes, GPU]
```

La caméra n'est ouverte qu'**une seule fois** par GStreamer (`v4l2src`).
Un `tee` distribue le flux à deux branches indépendantes.

### Installation

#### 1. S'authentifier sur le registre NVIDIA NGC

L'image de base DeepStream est hébergée sur `nvcr.io`.

1. Créer un compte sur [https://ngc.nvidia.com](https://ngc.nvidia.com)
2. Générer une clé API : menu utilisateur → **Setup** → **Generate API Key**
3. S'authentifier :

```bash
docker login nvcr.io
# Username : $oauthtoken        ← littéralement cette chaîne
# Password : <votre-clé-api>
```

#### 2. Télécharger le modèle PeopleNet

Le modèle PeopleNet v2.6 n'est pas inclus dans l'image Docker — il doit être téléchargé séparément depuis NVIDIA TAO.

```bash
# Créer le répertoire modèle dans le volume data
mkdir -p /tmp/peoplenet

# Télécharger depuis le dépôt TAO (nécessite un compte NGC)
# Option A : via NGC CLI (si installé)
ngc registry model download-version \
  "nvidia/tao/peoplenet:deployable_quantized_v2.6" \
  --dest /tmp/peoplenet

# Option B : téléchargement direct du fichier etlt
# Accéder à https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet
# → Télécharger "deployable_quantized_v2.6" → resnet34_peoplenet_int8.etlt
```

Copier le modèle dans le volume Docker :

```bash
docker compose --profile deepstream run --rm app-deepstream \
  mkdir -p /data/models/peoplenet
docker cp /tmp/peoplenet/resnet34_peoplenet_int8.etlt \
  $(docker compose --profile deepstream ps -q app-deepstream):/data/models/peoplenet/
```

> Alternativement, copier le fichier directement dans le volume nommé `data` :
> ```bash
> docker run --rm -v people-counter_data:/data busybox mkdir -p /data/models/peoplenet
> docker run --rm -v people-counter_data:/data \
>   -v /tmp/peoplenet:/src busybox \
>   cp /src/resnet34_peoplenet_int8.etlt /data/models/peoplenet/
> ```

#### 3. Construire l'image et démarrer

```bash
docker compose --profile deepstream build   # ~5 min (télécharge pyds, installe deps)
docker compose --profile deepstream up -d
```

**Premier démarrage :** `nvinfer` compile le modèle PeopleNet en moteur TensorRT INT8.
Durée : ~5 min sur Jetson Orin. Le moteur est mis en cache dans `/data/models/peoplenet/`.

#### 3. Vérifier le ROI porte

```bash
# Copier l'image de vérification générée au démarrage
docker compose cp app-deepstream:/data/roi_check.png ./roi_check.png
```

L'image montre le rectangle vert (`DOOR_ROI`) et la ligne rouge (`LINE_Y`) superposés à la scène.
Si le placement ne convient pas, ajuster `LINE_Y` et `DOOR_ROI` dans `app/deepstream_main.py` et redémarrer.

#### 4. Recapturer la référence porte

La caméra est verrouillée par GStreamer. La recapture s'effectue via signal UNIX sans redémarrer :

```bash
# Porte bien FERMÉE, puis :
docker compose exec app-deepstream python3 /app/reset_reference.py
```

`reset_reference.py` envoie `SIGUSR1` au processus principal, qui recapture la prochaine frame.

### Configuration DeepStream

#### Ligne de comptage et zone porte

Dans `app/deepstream_main.py` :

```python
LINE_Y         = 0.5              # fraction de hauteur (à calibrer)
DOOR_ROI       = (0.2, 0.1, 0.8, 0.9)
DOOR_THRESHOLD = 25               # différence moyenne (0–255) pour "open"
```

#### Seuil de détection PeopleNet

Dans `app/config/pgie_peoplenet.txt` :

```ini
[class-attrs-0]
pre-cluster-threshold=0.4   # confiance minimale pour la classe "person"
```

- Augmenter (0.6–0.8) pour réduire les faux positifs
- Diminuer (0.2–0.3) si des personnes ne sont pas détectées

#### Tracker NvDCF

Dans `app/config/tracker_nvdcf.yml` :

```yaml
NvDCF:
  probationAge: 3         # frames avant de valider une piste
  keepLostFrameNum: 0     # suppression immédiate après disparition
  maxTargetsPerStream: 30
```

### Dépannage DeepStream

| Symptôme | Solution |
| --- | --- |
| Démarrage lent (~5 min) | Normal au premier lancement — compilation TRT en cours |
| `nvinfer` erreur modèle introuvable | Vérifier que l'image `deepstream:9.0-samples-multiarch` est bien tirée |
| Pas de détection après démarrage | Vérifier `pre-cluster-threshold` dans `pgie_peoplenet.txt` |
| Erreur `pyds` introuvable | S'assurer d'utiliser l'image `deepstream:9.0-samples-multiarch` (pyds inclus) |
| `GStreamer element not found` | L'image `9.0-samples-multiarch` inclut tous les plugins — vérifier le tag |

---

## Dashboard Grafana

Commun aux deux stacks. Accessible sur `http://<ip-du-pe1103n>:3000` (accès anonyme).

Le dashboard **"Compteur de personnes"** se rafraîchit toutes les 5 secondes.

### Seuils d'alerte

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

### Alertes email

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

```bash
docker compose --profile <yolo|deepstream> restart dashboard
```

---

## Maintenance

### Recapturer la frame de référence porte

```bash
# YOLO
docker compose exec app-yolo python /app/reset_reference.py

# DeepStream (via SIGUSR1 — sans redémarrer le pipeline)
docker compose exec app-deepstream python3 /app/reset_reference.py
```

### Vérifier les logs

```bash
docker compose --profile yolo       logs -f app-yolo
docker compose --profile deepstream logs -f app-deepstream
docker compose --profile yolo       logs -f dashboard
```

### Compter les événements depuis la base

```bash
docker compose exec app-yolo python3 -c "
import sqlite3
con = sqlite3.connect('/data/counts.db')
for row in con.execute('SELECT direction, count(*) FROM events GROUP BY direction'):
    print(row)
"
```

### Remettre les compteurs à zéro

```bash
docker compose exec app-yolo python3 -c "
import sqlite3
con = sqlite3.connect('/data/counts.db')
con.execute('DELETE FROM events')
con.execute('DELETE FROM door_status')
con.commit()
con.execute('VACUUM')
print('Base remise à zéro.')
"
```

### Nettoyage SQLite hebdomadaire (optionnel)

```bash
# crontab -e
0 3 * * 0 docker compose --profile yolo -f /chemin/vers/docker-compose.yml \
  exec -T app-yolo python3 -c \
  "import sqlite3; con=sqlite3.connect('/data/counts.db'); con.execute('VACUUM')"
```

### Vérifier la persistance après reboot

```bash
sudo reboot
# après redémarrage
docker compose --profile yolo ps
```

---

## Dépannage général

| Symptôme | Solution |
| --- | --- |
| `CUDA: False` dans le conteneur | Vérifier `--runtime nvidia` et `daemon.json` |
| Caméra non détectée | Vérifier droits groupe `video` et `devices:` dans compose |
| Grafana ne démarre pas | `docker compose logs dashboard` — vérifier `GF_PATHS_PLUGINS=/grafana-plugins` |
| Grafana vide au démarrage | Attendre 15–20 s, puis rafraîchir |
| `module.js` 404 dans les logs Grafana | Conflit volume — vérifier `GF_PATHS_PLUGINS` et `./grafana/plugins:/grafana-plugins` |
| Fausses alarmes porte | Augmenter `DOOR_THRESHOLD` ou recapturer la référence |
| Porte oscille OPEN/CLOSED en continu | Référence obsolète — recapturer (porte bien fermée) |
| Dashboard Grafana corrompu | `docker compose down -v && docker compose --profile <stack> up -d` |

**YOLO uniquement :**

| Symptôme | Solution |
| --- | --- |
| `détections=1 tracks=0` | `new_track_thresh` ByteTrack trop élevé — vérifier `bytetrack_low.yaml` |
| Passages non comptés, `cy` constant | `LINE_Y` mal calibré — lire `cy` dans `[TRACK]` et recalculer |
| `COUNT_ONLY_DOOR_OPEN=True` ne compte rien | Délai hysteresis > temps de traversée — utiliser `False` |
| `yolov8n.pt` absent | Copier manuellement : `docker cp /tmp/yolov8n.pt people-counter-app-yolo-1:/data/` |
| Double comptage | Ajuster `LINE_Y` via le flux live `http://<ip>:8080` |
| Sous-comptage groupes | YOLO fusionne les bbox proches — limite monoculaire, vérifier `tracks=` dans `[DBG]` |

**DeepStream uniquement :**

| Symptôme | Solution |
| --- | --- |
| Démarrage lent (~5 min) | Normal au premier lancement — compilation TRT PeopleNet |
| `nvinfer` erreur modèle introuvable | Vérifier que l'image `deepstream:9.0-samples-multiarch` est bien tirée |
| Pas de détection | Abaisser `pre-cluster-threshold` dans `pgie_peoplenet.txt` |

---

## Notes RGPD

- Aucune image n'est stockée — uniquement les timestamps et directions de passage
- La détection de porte ne stocke que l'état (open/closed) et l'horodatage
- Exposer Grafana uniquement sur le réseau local (ne pas ouvrir le port 3000 sur internet)
- Le port 8080 (visualisation live YOLO) est destiné au calibrage uniquement — le fermer en production
