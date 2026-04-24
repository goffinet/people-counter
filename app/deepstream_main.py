"""
People counter + door status detector
ASUS IoT PE1103N — JetPack 6 / DeepStream 7 / PeopleNet v2.6 / NvDCF

Pipeline GStreamer (une seule ouverture caméra) :
  v4l2src ─┬─ queue → videoconvert → appsink          [détection porte, CPU]
            └─ queue → nvvideoconvert → nvinfer
                       → nvtracker → probe             [comptage personnes, GPU]

Configuration :
  LINE_Y          : ligne de comptage (fraction de hauteur, 0.0-1.0)
  DOOR_ROI        : zone porte (x1, y1, x2, y2 en fractions)
  DOOR_THRESHOLD  : sensibilité détection porte (différence moyenne 0-255)
  PERSON_CLASS    : indice de classe PeopleNet (0=personne, 1=sac, 2=visage)
"""

import os
import signal
import threading
import time
import sqlite3

import cv2
import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB             = "/data/counts.db"
LINE_Y         = 0.5
DOOR_ROI       = (0.2, 0.1, 0.8, 0.9)
DOOR_THRESHOLD = 25
REF_DELAY      = 3
PERSON_CLASS   = 0
FRAME_W        = 640
FRAME_H        = 480

# ---------------------------------------------------------------------------
# État partagé
# ---------------------------------------------------------------------------

_db_lock       = threading.Lock()
_frame_lock    = threading.Lock()
_latest_bgr    = None    # dernière frame CPU disponible (BGR)
_reference     = None    # ROI de référence porte (niveaux de gris)
_last_door     = None    # dernier statut porte connu
_prev_y        = {}      # {track_id: y_centre_précédent}
_reset_ref     = False   # flag SIGUSR1 : recapturer la référence

# ---------------------------------------------------------------------------
# Base de données (thread-safe)
# ---------------------------------------------------------------------------

def init_db():
    with _db_lock:
        con = sqlite3.connect(DB)
        con.execute("CREATE TABLE IF NOT EXISTS events (ts INTEGER, direction TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS door_status (ts INTEGER, status TEXT)")
        con.commit()
        con.close()

def db_write(sql, params):
    with _db_lock:
        con = sqlite3.connect(DB)
        con.execute(sql, params)
        con.commit()
        con.close()

# ---------------------------------------------------------------------------
# Recapture de la référence porte via SIGUSR1
#   docker compose exec app-deepstream kill -USR1 $(cat /data/app.pid)
# ---------------------------------------------------------------------------

def _sigusr1_handler(signum, frame_arg):
    global _reset_ref
    _reset_ref = True
    print("[INFO] SIGUSR1 reçu : recapture de la référence porte au prochain frame.")

signal.signal(signal.SIGUSR1, _sigusr1_handler)

def _capture_reference(frame_bgr):
    global _reference, _reset_ref
    h, w = frame_bgr.shape[:2]
    x1, y1 = int(DOOR_ROI[0]*w), int(DOOR_ROI[1]*h)
    x2, y2 = int(DOOR_ROI[2]*w), int(DOOR_ROI[3]*h)
    _reference  = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    _reset_ref  = False

    debug = frame_bgr.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(debug, (0, int(h * LINE_Y)), (w, int(h * LINE_Y)), (0, 0, 255), 2)
    cv2.imwrite("/data/roi_check.png", debug)
    print("[INFO] Référence porte capturée — /data/roi_check.png mis à jour.")

# ---------------------------------------------------------------------------
# Callback appsink — stocke la dernière frame + détection porte
# ---------------------------------------------------------------------------

def _on_new_sample(sink):
    global _latest_bgr, _last_door

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK

    try:
        frame = np.ndarray(
            shape=(FRAME_H, FRAME_W, 3),
            dtype=np.uint8,
            buffer=map_info.data,
        ).copy()
    finally:
        buf.unmap(map_info)

    with _frame_lock:
        _latest_bgr = frame

    if _reset_ref:
        _capture_reference(frame)
        return Gst.FlowReturn.OK

    if _reference is None:
        return Gst.FlowReturn.OK

    h, w = frame.shape[:2]
    x1, y1 = int(DOOR_ROI[0]*w), int(DOOR_ROI[1]*h)
    x2, y2 = int(DOOR_ROI[2]*w), int(DOOR_ROI[3]*h)
    roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    diff     = float(np.mean(cv2.absdiff(_reference, roi_gray)))
    status   = "open" if diff > DOOR_THRESHOLD else "closed"

    if status != _last_door:
        db_write("INSERT INTO door_status VALUES (?,?)", (int(time.time()), status))
        print(f"[DOOR] Statut changé : {status}")
        _last_door = status

    return Gst.FlowReturn.OK

# ---------------------------------------------------------------------------
# Probe GStreamer — comptage des passages (thread streaming DeepStream)
# ---------------------------------------------------------------------------

def _counting_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    threshold  = int(FRAME_H * LINE_Y)

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == PERSON_CLASS:
                rect = obj_meta.rect_params
                cy   = int(rect.top + rect.height / 2)
                tid  = int(obj_meta.object_id)

                if tid in _prev_y:
                    if _prev_y[tid] < threshold <= cy:
                        db_write("INSERT INTO events VALUES (?,?)",
                                 (int(time.time()), "in"))
                        print(f"[COUNT] Entrée — track {tid}")
                    elif _prev_y[tid] > threshold >= cy:
                        db_write("INSERT INTO events VALUES (?,?)",
                                 (int(time.time()), "out"))
                        print(f"[COUNT] Sortie — track {tid}")
                _prev_y[tid] = cy

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# ---------------------------------------------------------------------------
# Construction du pipeline GStreamer
# ---------------------------------------------------------------------------

def _make(factory, name):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"Élément GStreamer introuvable : '{factory}' — "
                           f"DeepStream est-il correctement installé ?")
    return el

def build_pipeline():
    pipeline = Gst.Pipeline.new("people-counter")

    src      = _make("v4l2src",        "src")
    src.set_property("device", "/dev/video0")

    caps_src = _make("capsfilter",     "caps_src")
    caps_src.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw,width={FRAME_W},height={FRAME_H},framerate=30/1"))

    tee      = _make("tee",            "tee")

    # --- Branche 1 : détection porte (CPU → appsink) ---
    q_door   = _make("queue",          "q_door")
    q_door.set_property("max-size-buffers", 1)
    q_door.set_property("leaky", 2)

    vconv    = _make("videoconvert",   "vconv")

    caps_bgr = _make("capsfilter",     "caps_bgr")
    caps_bgr.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={FRAME_W},height={FRAME_H}"))

    appsink  = _make("appsink",        "appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers",  1)
    appsink.set_property("drop",         True)
    appsink.set_property("sync",         False)
    appsink.connect("new-sample", _on_new_sample)

    # --- Branche 2 : inférence GPU (DeepStream) ---
    q_det    = _make("queue",          "q_det")

    nvconv   = _make("nvvideoconvert", "nvconv")

    caps_nv  = _make("capsfilter",     "caps_nv")
    caps_nv.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=NV12"))

    pgie     = _make("nvinfer",        "pgie")
    pgie.set_property("config-file-path", "/app/config/pgie_peoplenet.txt")

    tracker  = _make("nvtracker",      "tracker")
    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "/app/config/tracker_nvdcf.yml")
    tracker.set_property("tracker-width",  FRAME_W)
    tracker.set_property("tracker-height", FRAME_H)

    sink     = _make("fakesink",       "sink")
    sink.set_property("sync", False)

    for el in [src, caps_src, tee,
               q_door, vconv, caps_bgr, appsink,
               q_det, nvconv, caps_nv, pgie, tracker, sink]:
        pipeline.add(el)

    src.link(caps_src)
    caps_src.link(tee)

    tee.get_request_pad("src_%u").link(q_door.get_static_pad("sink"))
    q_door.link(vconv)
    vconv.link(caps_bgr)
    caps_bgr.link(appsink)

    tee.get_request_pad("src_%u").link(q_det.get_static_pad("sink"))
    q_det.link(nvconv)
    nvconv.link(caps_nv)
    caps_nv.link(pgie)
    pgie.link(tracker)
    tracker.link(sink)

    tracker.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, _counting_probe, 0)

    return pipeline

# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    Gst.init(None)
    init_db()

    with open("/data/app.pid", "w") as f:
        f.write(str(os.getpid()))

    print("[INFO] Construction du pipeline DeepStream...")
    pipeline = build_pipeline()
    pipeline.set_state(Gst.State.PLAYING)

    print(f"[INFO] Attente {REF_DELAY}s — porte doit être fermée...")
    time.sleep(REF_DELAY)

    with _frame_lock:
        ref_frame = _latest_bgr

    if ref_frame is not None:
        _capture_reference(ref_frame)
        print("[INFO] Détection active.")
    else:
        print("[WARN] Aucune frame disponible — détection porte désactivée.")

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_message(bus, msg):
        if msg.type == Gst.MessageType.EOS:
            print("[INFO] Fin du flux.")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {err} — {dbg}")
            loop.quit()

    bus.connect("message", _on_message)

    print("[INFO] Pipeline actif. Ctrl+C pour arrêter.")
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        if os.path.exists("/data/app.pid"):
            os.remove("/data/app.pid")


if __name__ == "__main__":
    main()
