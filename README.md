# People Counter — ASUS IoT PE1103N

Système de comptage de personnes et de surveillance de porte par caméra USB,
déployé en conteneurs Docker sur ASUS IoT PE1103N (NVIDIA Jetson Orin, JetPack 6.x).

Stack de détection : **YOLOv8n + ByteTrack (Ultralytics)**

---

## Branches

| Branche | Stack de détection | Image Docker |
| --- | --- | --- |
| **`feature/yolo`** | **YOLOv8n + ByteTrack (Ultralytics)** | **`people-counter-app:latest` (build local)** |
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
├── Dockerfile                 # image dérivée ultralytics + lapx (ByteTrack)
├── docker-compose.yml
├── wheels/
│   └── lapx-*.whl             # dépendance ByteTrack pré-téléchargée (aarch64)
├── app/
│   ├── main.py                # pipeline YOLOv8 + ByteTrack + serveur MJPEG (port 8080)
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

### 4. Télécharger le modèle YOLOv8n

Le conteneur n'a pas accès internet. Télécharger le modèle depuis l'hôte :

```bash
curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt \
     -o /tmp/yolov8n.pt
```

Il sera copié dans le volume `/data` au premier `docker compose up`.

### 5. Construire l'image et démarrer la stack

```bash
docker compose build          # construit people-counter-app:latest (~2 min)
docker compose up -d
docker cp /tmp/yolov8n.pt people-counter-app-1:/data/yolov8n.pt
```

> Le plugin Grafana SQLite (`frser-sqlite-datasource`) est fourni dans `grafana/plugins/`
> et chargé localement — aucun accès internet requis pour Grafana.

### 6. Calibrer la ligne de comptage et le ROI porte

Le serveur de visualisation démarre automatiquement avec le pipeline.
Ouvrir dans un navigateur :

```
http://<ip-du-pe1103n>:8080
```

Le flux live affiche :
- **Ligne rouge** = position de `LINE_Y` (ligne virtuelle de comptage)
- **Rectangle vert** = zone `DOOR_ROI` (zone analysée pour détecter la porte)

**Récupérer une image fixe :**

```bash
curl http://<ip-du-pe1103n>:8080/snapshot -o snapshot.jpg
```

Une fois les valeurs correctes identifiées, les modifier dans `app/main.py` :

```python
LINE_Y   = 0.4
DOOR_ROI = (0.15, 0.05, 0.85, 0.95)
```

Puis redémarrer le pipeline :

```bash
docker compose restart app
```

### 7. Capturer la référence porte (porte fermée)

La frame de référence sert à détecter si la porte est ouverte ou fermée.
Elle est capturée automatiquement au premier démarrage.
Pour la recapturer manuellement (après un changement d'éclairage par exemple) :

```bash
# S'assurer que la porte est bien FERMÉE avant d'exécuter
docker compose exec app python /app/reset_reference.py
```

La nouvelle référence est sauvegardée dans `/data/door_reference.pkl`
et une image de vérification dans `/data/roi_check.png`.

### 8. Vérifier le dashboard Grafana

Grafana est accessible sur `http://<ip-du-pe1103n>:3000`

Le dashboard **"Compteur de personnes"** s'affiche directement (accès anonyme activé).
Il se rafraîchit toutes les 5 secondes.

---

## Comprendre la logique de détection

### Système de coordonnées de l'image

OpenCV (et donc ce pipeline) utilise un repère où **y=0 est en haut** de l'image et **y augmente vers le bas**.
Pour une image de 480 pixels de haut :

```
y=0   ┌─────────────────────────────┐  ← haut de l'image
      │                             │
y=230 │  - - - - ligne rouge  - - - │  ← LINE_Y=0.48 → line_px=230
      │                             │
      │   ┌───────┐  ← bounding box │
      │   │ conf  │     YOLO        │
y=302 │   │  ● cy │  ← centre de   │  ← cy = (by1+by2)/2
      │   └───────┘     la bbox     │
      │                             │
y=480 └─────────────────────────────┘  ← bas de l'image
```

Les valeurs clés du pipeline :

| Variable | Type | Description |
| --- | --- | --- |
| `LINE_Y` | float 0.0–1.0 | Position de la ligne en **fraction** de la hauteur |
| `line_px` | int (pixels) | `line_px = int(h * LINE_Y)` — calculé une seule fois au démarrage |
| `cy` | float (pixels) | Centre vertical de la bounding box : `cy = (y_haut + y_bas) / 2` |

### Comment un passage est compté

À chaque frame, le pipeline mémorise `cy` par identifiant de track (`prev_centers[tid]`).
Un franchissement est détecté quand `cy` passe **d'un côté à l'autre** de `line_px` entre deux frames consécutives :

```
prev_cy < line_px  ET  cy >= line_px  →  direction = "in"   (descend dans l'image)
prev_cy > line_px  ET  cy <= line_px  →  direction = "out"  (remonte dans l'image)
```

Concrètement : si la caméra est placée **face à la porte** et regarde le couloir, une personne qui entre dans la salle s'approche de la caméra et son `cy` **augmente** (elle descend dans l'image). Elle passe donc de `cy < line_px` à `cy > line_px` → comptée **entrée**.

> **Règle pratique :** la ligne doit couper la trajectoire de la personne entre sa position de départ (avant le seuil) et sa position d'arrivée (après le seuil). Si la personne est toujours du même côté de la ligne rouge dans le flux live, le comptage ne se déclenche pas.

### Calibrer `LINE_Y` avec le flux de visualisation

1. Ouvrir `http://<ip>:8080` dans un navigateur — le flux affiche en **orange** la bounding box détectée avec son `cy`, et en **rouge** la `LINE_Y` courante.
2. Marcher devant la caméra en simulant un passage complet (entrer puis sortir).
3. Observer les valeurs `cy` affichées à l'écran :
   - Note le `cy` **avant** le seuil de la porte (ex. 285)
   - Note le `cy` **après** le seuil (ex. 350)
4. Placer `LINE_Y` entre ces deux valeurs : `LINE_Y = (285 + 350) / 2 / hauteur_image`
5. Vérifier dans les logs que des lignes `[EVENT] IN` ou `[EVENT] OUT` apparaissent.

Les logs utiles pendant le calibrage :

```bash
docker compose logs -f app | grep -E "TRACK|EVENT"
# [TRACK] id=12 cy=290 line=302 conf=0.93   ← cy < line : personne pas encore passée
# [TRACK] id=12 cy=335 line=302 conf=0.91   ← cy > line : personne passée → EVENT
# [EVENT] IN — track 12
```

---

## Configuration

### Ligne de comptage (`LINE_Y`)

```python
LINE_Y = 0.63   # ligne à 63 % de la hauteur → line_px = 302 pour une image 480p
```

Valeurs typiques selon l'installation :

| Montage caméra | `LINE_Y` recommandé |
| --- | --- |
| Face à la porte, vue couloir | 0.55 – 0.70 (à calibrer) |
| Au-dessus de la porte (vue plongeante) | 0.45 – 0.55 |
| Caméra basse regardant vers le haut | 0.30 – 0.45 |

### Zone de détection porte (`DOOR_ROI`)

`DOOR_ROI = (x1, y1, x2, y2)` — les quatre valeurs sont des **fractions** de la taille de l'image (0.0 à 1.0). L'origine (0, 0) est en **haut à gauche**.

```
(0,0) ──────────────────────── (1,0)
  │                               │
  │   x1,y1 ┌──────────┐         │
  │          │  DOOR    │         │
  │          │  ROI     │         │
  │   x1,y2 └──────────┘ x2,y2   │
  │                               │
(0,1) ──────────────────────── (1,1)
```

Exemple : `DOOR_ROI = (0.6, 0.1, 0.8, 0.9)` couvre la bande verticale entre 60 % et 80 % de la largeur, de 10 % à 90 % de la hauteur.

**Important :** `x1 < x2` et `y1 < y2` — toujours mettre la coordonnée la plus petite en premier. Le rectangle vert visible sur le flux live correspond exactement à cette zone.

Pour recalibrer :
1. Regarder le flux `http://<ip>:8080` — le rectangle vert doit couvrir la porte entière.
2. Modifier `DOOR_ROI` dans `app/main.py`, redémarrer avec `docker compose restart app`.
3. Recapturer la référence porte (porte fermée) : `docker compose exec app python /app/reset_reference.py`.

### Détection porte : paramètres fins

La détection compare chaque frame à une **image de référence** (porte fermée) pixel par pixel dans la zone `DOOR_ROI`.

| Paramètre | Valeur | Rôle |
| --- | --- | --- |
| `DOOR_PIXEL_DIFF` | `30` | Différence minimale (0–255) pour qu'un pixel soit considéré "changé". Augmenter si l'éclairage fluctue. |
| `DOOR_THRESHOLD` | `0.35` | Fraction (0.0–1.0) de pixels changés pour déclarer la porte ouverte. |
| `DOOR_HYSTERESIS` | `25` | Nombre de frames **consécutives** avant de valider un changement d'état. |

#### Pourquoi une personne qui passe déclenche un faux positif

Le pipeline possède déjà une protection primaire : si YOLO détecte une personne dont la bounding box chevauche le `DOOR_ROI`, la comparaison de pixels est suspendue pour cette frame. Mais quand YOLO rate une détection (confiance insuffisante, personne partiellement hors cadre), la comparaison s'exécute et voit une silhouette là où la référence montrait une porte vide.

La distinction entre une personne qui passe et une porte qui s'ouvre repose sur deux observations :

```
Personne qui passe devant une DOOR_ROI étroite (20 % de la largeur image)
  → silhouette couvre ~20 % du ROI
  → passage dure ~0.3–0.5 s  (9–15 frames à 30 fps)

Porte réellement ouverte
  → fond du couloir remplace toute la surface de la porte
  → ~70–100 % du ROI change
  → état maintenu plusieurs secondes
```

`DOOR_THRESHOLD = 0.35` : en demandant que 35 % des pixels aient changé, une silhouette (~20 %) ne suffit plus à déclencher "open". Une vraie ouverture (~70–100 %) la déclenche toujours.

`DOOR_HYSTERESIS = 25` : à 30 fps, 25 frames = ~0.8 s. Un passage rapide (<0.5 s) ne cumule pas assez de frames consécutives. Une porte ouverte maintient l'état bien au-delà de 0.8 s.

#### Calibrer ces valeurs sur site

Pour trouver les bonnes valeurs, activer temporairement les logs de ratio en ajoutant dans `detect_door()` :

```python
print(f"[DOOR_RATIO] {changed_ratio:.3f}")
```

Puis mesurer :
- `changed_ratio` quand une personne passe devant (sans ouvrir) → valeur **A**
- `changed_ratio` quand la porte est ouverte → valeur **B**

Régler `DOOR_THRESHOLD` entre A et B, avec une marge : `DOOR_THRESHOLD = A + (B - A) * 0.4`.

Réglage selon l'environnement :

| Environnement | `DOOR_PIXEL_DIFF` | `DOOR_THRESHOLD` | `DOOR_HYSTERESIS` |
| --- | --- | --- | --- |
| ROI étroit (≤ 20 % largeur), passages fréquents | 30 | 0.30–0.40 | 20–25 |
| ROI large (≥ 40 % largeur) | 30 | 0.15–0.25 | 15–20 |
| Éclairage variable (soleil direct) | 40–50 | 0.35–0.45 | 25–30 |
| Porte peu contrastée / vitrée | 15–20 | 0.20–0.30 | 20 |

### Conditionner le comptage à l'état de la porte (`COUNT_ONLY_DOOR_OPEN`)

```python
COUNT_ONLY_DOOR_OPEN = False   # comportement par défaut : compter en toutes circonstances
COUNT_ONLY_DOOR_OPEN = True    # ne compter que quand la porte est détectée ouverte
```

| Valeur | Comportement |
| --- | --- |
| `False` | Chaque franchissement de la ligne est enregistré, quel que soit l'état de la porte. Utile si la caméra couvre un couloir sans porte, ou si la détection de porte est désactivée. |
| `True` | Un franchissement n'est enregistré que si `door_prev == "open"`. Évite de compter des mouvements derrière une porte fermée (ex. ombre, reflet). |

> **Attention avec `True` :** le premier état de porte est inconnu au démarrage (`door_prev = ""`). Les passages qui surviennent avant la première détection d'état ne seront pas comptés. Attendre quelques secondes après le démarrage avant de faire circuler des personnes.

### Seuil de confiance de détection (`conf`)

```python
conf=0.10   # dans model.track() — app/main.py
```

YOLOv8 filtre les détections dont le score est inférieur à ce seuil. La valeur par défaut Ultralytics est `0.25`, mais des vues en angle nécessitent souvent `0.10`–`0.15`. En dessous de `0.10`, le risque de faux positifs (détection d'objets non-personnes) augmente.

### Modèle de détection (`MODEL_PATH`)

```python
MODEL_PATH = "/data/yolov8n.pt"     # PyTorch — copié dans /data au démarrage (étape 4)
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
docker compose exec app python3 -c "
import sqlite3
con = sqlite3.connect('/data/counts.db')
for row in con.execute('SELECT direction, count(*) FROM events GROUP BY direction'):
    print(row)
"
```

### Remettre les compteurs à zéro

```bash
docker compose exec app python3 -c "
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
| `yolov8n.pt` ne se télécharge pas | Le conteneur n'a pas accès internet — suivre l'étape 4 : télécharger sur l'hôte puis `docker cp /tmp/yolov8n.pt people-counter-app-1:/data/` |
| Grafana ne démarre pas | Vérifier les logs : `docker compose logs dashboard` — si erreur plugin, le répertoire `grafana/plugins/` doit être présent |
| Grafana vide au démarrage | Attendre 15-20s, puis rafraîchir |
| Fausses alarmes porte | Augmenter `DOOR_THRESHOLD` ou recapturer la référence |
| Double comptage | Ajuster `LINE_Y` via le flux live `http://<ip>:8080` |
| Dashboard Grafana corrompu | Supprimer le volume et redémarrer : `docker compose down -v && docker compose up -d` |

---

## Notes RGPD

- Aucune image n'est stockée — uniquement les timestamps et directions de passage
- La détection de porte ne stocke que l'état (open/closed) et l'horodatage
- Exposer Grafana uniquement sur le réseau local (ne pas ouvrir le port 3000 sur internet)
- Le port 8080 (visualisation live) est destiné à la phase de calibrage uniquement — le fermer en production si non nécessaire
