"""
WEMO Command Center — Flask Backend  v4.7 (Fixed Alert Logic)
==========================================
"""

import copy
import csv
import json
import os
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import base64
from collections import deque
from datetime import datetime

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

# ─── Config from .env ─────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WA_FROM     = os.getenv("TWILIO_WA_FROM", "whatsapp:+14155238886")
TWILIO_WA_TO       = os.getenv("TWILIO_WA_TO",   "whatsapp:+923073690442")
WHATSAPP_ENABLED   = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)

MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL    = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_ENABLED  = bool(MISTRAL_API_KEY)

DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAINING_CSV = os.getenv("TRAINING_CSV", "wemo_training_data.csv")
KNN_PATH     = os.getenv("KNN_PATH",    "wemo_knn_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH", "wemo_scaler.pkl")
TRAINING_FIELDNAMES = ["power_W", "power_factor", "appliance"]

# ─── Dynamic Config System ────────────────────────────────────────────────────
WEMO_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wemo_config.json")
_config_lock = threading.Lock()

def get_default_config():
    return {
        "class_power_floor": {
            "Bulb + Fan":      12.3,
            "Fan":             20.0,
            "Laptop Charger": 35.0,
            "Iron":           700.0,
            "AC":             800.0
        },
        "health_thresholds": {
            "Laptop Charger":        [40,   95],
            "Phone Charger":         [5,    25],
            "Mobile Charger":        [5,    25],
            "LED Bulb":              [5,    20],
            "Bulb":                  [5,    20],
            "Fan":                   [14,   80],
            "Bulb + Fan":            [10,   25],
            "Bulb + Mobile Charger": [14,   30],
            "Iron":                  [800, 1200],
            "AC":                    [900, 2200],
            "No Load Connected":     [0,  9999]
        }
    }

def init_wemo_config():
    if not os.path.exists(WEMO_CONFIG_FILE):
        with open(WEMO_CONFIG_FILE, "w") as f:
            json.dump(get_default_config(), f, indent=4)

def read_wemo_config():
    try:
        with open(WEMO_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return get_default_config()

init_wemo_config()

def _mistral_chat(prompt: str) -> str:
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 450,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    url = "https://api.mistral.ai/v1/chat/completions"

    try:
        import httpx
        r = httpx.post(url, headers=headers, json=payload, timeout=25.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except ModuleNotFoundError:
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25.0) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Mistral HTTP {e.code}: {err_body}") from e
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"].strip()

FACTORY_SOURCES = [
    (os.path.join(DATA_DIR, "bulb.csv"),          "Bulb"),
    (os.path.join(DATA_DIR, "mobilecharger.csv"), "Mobile Charger"),
    (os.path.join(DATA_DIR, "bulbandfan.csv"),    "Bulb + Fan"),
]

STABILIZE_SECS    = 3
RECORD_TARGET     = 30
ANOMALY_THRESHOLD = 5.5
NO_LOAD_RECORD_THRESHOLD = 3.0

# ─── Stability Window Config ──────────────────────────────────────────────────
STABILITY_WINDOW       = 3    # packets
STABILITY_VARIANCE_MAX = 5.0  # Watts

# ─── User Presence Detection ──────────────────────────────────────────────────
# Only the browser dashboard updates this timestamp via GET /live.
# The ESP32 uses /api/relay-state-esp which does NOT update this.
# If no browser pinged /live in the last 30 seconds → user is away → send WhatsApp.
USER_PRESENT_TIMEOUT = 30.0
_last_dashboard_ping = 0.0    # starts at epoch = user NOT present on startup
_ping_lock = threading.Lock()

# ─── Training Mode — explicit "I'm consciously training" flag ─────────────────
# When ON:  WhatsApp alerts are suppressed regardless of user presence.
# When OFF: Normal logic — alert fires only when user is away from dashboard.
_training_mode_active = False
_training_mode_lock   = threading.Lock()

def _user_is_present() -> bool:
    age = time.time() - _last_dashboard_ping
    present = age < USER_PRESENT_TIMEOUT
    print(f"[PRESENCE] Last browser ping {age:.1f}s ago → user {'PRESENT' if present else 'AWAY'}", flush=True)
    return present

app = Flask(__name__)

_ml_lock = threading.Lock()
_knn     = None
_scaler  = None
ML_READY = False

def _get_ml():
    with _ml_lock:
        return _knn, _scaler, ML_READY

def load_models():
    global _knn, _scaler, ML_READY
    try:
        nk = joblib.load(KNN_PATH)
        ns = joblib.load(SCALER_PATH)
        with _ml_lock:
            _knn = nk; _scaler = ns; ML_READY = True
        print("[ML] Models loaded OK.", flush=True)
    except Exception as e:
        with _ml_lock:
            _knn = _scaler = None; ML_READY = False
        print(f"[WARN] Model files not found — AI disabled. ({e})", flush=True)

load_models()

FAULT_VOLTAGE_LOW  = 200.0
FAULT_VOLTAGE_HIGH = 250.0
FAULT_CURRENT_HIGH = 13.0
FAULT_PF_LOW       = 0.50
FAULT_POWER_HIGH   = 2000.0

# ─── Plausibility Guard — reject garbage ESP32 readings ──────────────────────
MAX_PLAUSIBLE_WATTS   = 3500.0
MAX_PLAUSIBLE_VOLTAGE = 300.0
MAX_PLAUSIBLE_CURRENT = 16.0

RELAY_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay_state.json")

def load_relay_state():
    return 0

def save_relay_state(state, user_set=False):
    try:
        with open(RELAY_STATE_FILE, "w") as f:
            json.dump({
                "relay_desired": int(state),
                "timestamp": datetime.now().isoformat(),
                "user_set": user_set,
            }, f)
    except Exception:
        pass

_initial_relay_desired = load_relay_state()
save_relay_state(_initial_relay_desired)

latest: dict = {
    "v": 0, "i": 0, "p": 0, "e": 0, "f": 0, "pf": 0,
    "relay": _initial_relay_desired, "rssi": 0, "uptime": "—",
    "prediction": "Waiting for data…", "device_status": "idle",
    "health_msg": "—", "anomaly_distance": None, "timestamp": None,
    "fault_voltage": False, "fault_current": False,
    "fault_pf": False, "fault_power": False, "fault_msg": "",
    "unknown_alert": False, "unknown_power": 0.0, "unknown_pf": 0.0,
    "recording_phase": "idle", "recording_count": 0,
    "recording_target": RECORD_TARGET, "relay_desired": _initial_relay_desired,
    "wemo_ai_response": None, "wemo_ai_ts": None,
    "retrain_count": 0,
}
history: deque = deque(maxlen=60)

_SMOOTH_N   = 1
_SMOOTH_THR = 1.0
_pred_buf:  list = []
_sm_label:  str  = None
_sm_status: str  = None
_sm_health: str  = None
_sm_dist         = None
_smooth_lock = threading.Lock()

def smooth_prediction(raw_label, raw_status, raw_health, raw_dist):
    global _pred_buf, _sm_label, _sm_status, _sm_health, _sm_dist
    with _smooth_lock:
        _pred_buf.append(raw_label)
        if len(_pred_buf) > _SMOOTH_N:
            _pred_buf = _pred_buf[-_SMOOTH_N:]
        votes = {}
        for p in _pred_buf:
            votes[p] = votes.get(p, 0) + 1
        winner, top = max(votes.items(), key=lambda x: x[1])
        frac = top / len(_pred_buf)
        if frac >= _SMOOTH_THR or len(_pred_buf) < 3:
            _sm_label = winner
            if winner == raw_label:
                _sm_status = raw_status
                _sm_health = raw_health
                _sm_dist   = raw_dist
        return (
            _sm_label  or raw_label,
            _sm_status or raw_status,
            _sm_health or raw_health,
            _sm_dist,
        )

def reset_smoother():
    global _pred_buf, _sm_label, _sm_status, _sm_health, _sm_dist
    with _smooth_lock:
        _pred_buf = []
        _sm_label = _sm_status = _sm_health = _sm_dist = None


# ─── Unknown Load Tracker ─────────────────────────────────────────────────────
_unk_lock = threading.Lock()
_unk = {
    "phase": "idle",
    "power_snap": 0.0,
    "pf_snap": 0.0,
    "i_snap": 0.0,
    "pending_name": "",
    "readings": [],
    "alert_sent": False,
    "stability_buffer": [],
    "suppress_alert": False,
}

def _reset_unk_locked():
    _unk.update({
        "phase": "idle",
        "power_snap": 0.0,
        "pf_snap": 0.0,
        "i_snap": 0.0,
        "pending_name": "",
        "readings": [],
        "alert_sent": False,
        "stability_buffer": [],
        "suppress_alert": False,
    })


def send_whatsapp_unknown_alert(power: float, pf: float, current: float):
    if not WHATSAPP_ENABLED:
        print("[WARN] WhatsApp not enabled — check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env", flush=True)
        return
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"

        message_body = (
            f"❓ *WEMO AI: Unknown Load Detected*\n\n"
            f"A new device was plugged in and needs a name!\n\n"
            f"⚡ *Power:* {power:.1f} W\n"
            f"🔌 *Current:* {current:.2f} A\n"
            f"📉 *PF:* {pf:.2f}\n\n"
           f"Tap to open your Command Center and name this load:\n"
           f"http://13.60.236.168/"
        )

        data = {
            "From": TWILIO_WA_FROM,
            "To":   TWILIO_WA_TO,
            "Body": message_body,
        }
        encoded_data = urllib.parse.urlencode(data).encode("utf-8")

        auth_str    = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}"
        auth_header = base64.b64encode(auth_str.encode("ascii")).decode("ascii")

        req = urllib.request.Request(url, data=encoded_data, method="POST")
        req.add_header("Authorization", f"Basic {auth_header}")

        with urllib.request.urlopen(req, timeout=10) as response:
            print(f"[INFO] WhatsApp Alert Sent! Power={power:.1f}W  HTTP={response.status}", flush=True)

    except Exception as e:
        print(f"[WARN] Twilio WhatsApp alert FAILED: {e}", flush=True)


def analyse_faults(v, i, p, pf):
    faults = []; fv = fi = fpf = fp = False
    if v > 0 and v < FAULT_VOLTAGE_LOW:
        fv = True; faults.append(f"Voltage sag {v:.0f} V")
    if v > FAULT_VOLTAGE_HIGH:
        fv = True; faults.append(f"Voltage surge {v:.0f} V")
    if i > FAULT_CURRENT_HIGH:
        fi = True; faults.append(f"High current {i:.2f} A")
    if p > 3.0 and pf > 0 and pf < FAULT_PF_LOW:
        fpf = True; faults.append(f"Poor PF {pf:.2f}")
    if p > FAULT_POWER_HIGH:
        fp = True; faults.append(f"Overload {p:.0f} W")
    return {"fault_voltage": fv, "fault_current": fi, "fault_pf": fpf, "fault_power": fp,
            "fault_msg": " | ".join(faults) if faults else ""}

def run_knn(power: float, pf: float) -> dict:
    if power <= 3.0:
        return {"is_no_load": True, "is_anomaly": False,
                "label": "No Load Connected", "dist": None,
                "status": "idle", "health": "No appliance detected."}

    knn, scaler, ready = _get_ml()
    if not ready:
        return {"is_no_load": False, "is_anomaly": False,
                "label": "Model not trained yet", "dist": None,
                "status": "idle", "health": "No trained model. Name a load to start."}

    cfg = read_wemo_config()
    class_power_floor = cfg.get("class_power_floor", {})
    health_thresholds = cfg.get("health_thresholds", {})

    scaled     = scaler.transform([[power, pf]])
    n_neighbors = knn.n_neighbors
    try:
        n_query = min(max(n_neighbors * 3, 5), len(knn._fit_X))
        distances, indices = knn.kneighbors(scaled, n_neighbors=n_query)
    except Exception:
        distances, indices = knn.kneighbors(scaled)
        n_query = n_neighbors

    dist = float(distances[0][0])

    if dist > ANOMALY_THRESHOLD:
        return {"is_no_load": False, "is_anomaly": True,
                "label": "Unknown Load Connected", "dist": round(dist, 3),
                "status": "warning",
                "health": f"New signature found ({power:.1f} W). Waiting for you to name it."}

    label = knn.predict(scaled)[0]

    floor = class_power_floor.get(label)
    if floor is not None and power < float(floor):
        for idx in indices[0]:
            try:
                candidate = knn.classes_[knn._y[idx]]
            except Exception:
                continue
            cf = class_power_floor.get(candidate)
            if cf is None or power >= float(cf):
                label = candidate
                break

    try:
        nn_label = knn.classes_[knn._y[indices[0][0]]]
        nn_floor  = class_power_floor.get(nn_label)
        if nn_label != label and (nn_floor is None or power >= float(nn_floor)):
            vote_score: dict = {}
            for rank, (d_val, idx) in enumerate(zip(distances[0][:n_neighbors],
                                                    indices[0][:n_neighbors])):
                cand = knn.classes_[knn._y[idx]]
                w = 1.0 / (d_val + 1e-9)
                vote_score[cand] = vote_score.get(cand, 0.0) + w
            best = max(vote_score, key=lambda k: vote_score[k])
            best_floor = class_power_floor.get(best)
            if best_floor is None or power >= float(best_floor):
                label = best
    except Exception:
        pass

    lo, hi = health_thresholds.get(label, [0, 9999])
    if power < float(lo):
        status = "warning"; health = f"{label} below normal ({power:.1f} W < {lo} W)."
    elif power > float(hi):
        status = "danger";  health = f"⚠ {label} over-consuming ({power:.1f} W > {hi} W)!"
    else:
        status = "normal";  health = f"{label} operating within normal range ({lo}–{hi} W)."

    return {"is_no_load": False, "is_anomaly": False,
            "label": label, "dist": round(dist, 3), "status": status, "health": health}

_retrain_lock = threading.Lock()

def retrain_model():
    global _knn, _scaler, ML_READY
    with _retrain_lock:
        try:
            import pandas as pd
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler

            if not os.path.exists(TRAINING_CSV):
                return
            df = pd.read_csv(TRAINING_CSV)
            df = df[df["power_W"] > 0.5].dropna()
            if len(df) < 3:
                return

            counts = df["appliance"].value_counts()
            X  = df[["power_W", "power_factor"]].values
            y  = df["appliance"].values
            ns = StandardScaler()
            Xs = ns.fit_transform(X)
            k  = max(1, min(5, int(counts.min())))
            nk = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
            nk.fit(Xs, y)

            joblib.dump(nk, KNN_PATH)
            joblib.dump(ns, SCALER_PATH)

            with _ml_lock:
                _knn = nk; _scaler = ns; ML_READY = True

            latest["retrain_count"] = latest.get("retrain_count", 0) + 1
            reset_smoother()
            print("[ML] Model retrained successfully.", flush=True)
        except Exception as e:
            print(f"[WARN] Retrain failed: {e}", flush=True)

def seed_training_csv(overwrite=False):
    if os.path.exists(TRAINING_CSV) and not overwrite:
        return
    with open(TRAINING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_FIELDNAMES)
        w.writeheader()
        for fpath, label in FACTORY_SOURCES:
            if not os.path.exists(fpath):
                continue
            with open(fpath, newline="") as src:
                for row in csv.DictReader(src):
                    try:
                        pw = float(row["power_W"]); pf = float(row["power_factor"])
                        if pw > 0.5:
                            w.writerow({"power_W": pw, "power_factor": pf, "appliance": label})
                    except (ValueError, KeyError):
                        continue

seed_training_csv(overwrite=False)
if not ML_READY:
    retrain_model()

_ai_lock = threading.Lock()
_ai_busy = False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/data", methods=["POST"])
def ingest():
    try:
        data  = request.get_json(force=True)
        power = float(data.get("power", data.get("p", 0)))
        pf    = float(data.get("pf", 0))
        v     = float(data.get("v",  latest["v"]))
        i     = float(data.get("i",  latest["i"]))

        # ── Plausibility guard — reject garbage ESP32 readings ────────────────
        if power > MAX_PLAUSIBLE_WATTS or v > MAX_PLAUSIBLE_VOLTAGE or i > MAX_PLAUSIBLE_CURRENT:
            print(f"[WARN] Rejected implausible reading: P={power}W V={v}V I={i}A", flush=True)
            return jsonify({"ok": False, "error": "Implausible sensor reading rejected"}), 400

        if not (0.0 <= pf <= 1.0):
            pf = 0.0   # clamp bad PF silently

        esp_relay = int(data.get("relay", latest["relay"]))
        if esp_relay != latest["relay"]:
            latest["relay_desired"] = esp_relay
            save_relay_state(esp_relay, user_set=True)

        knn_result = run_knn(power, pf)

        with _unk_lock:

            # ── Guard 1: load dropped ─────────────────────────────────────────
            if power < NO_LOAD_RECORD_THRESHOLD and _unk["phase"] in ("stabilizing", "alerted", "recording"):
                print(f"[STABILITY] Load dropped ({power:.1f} W). Resetting.", flush=True)
                _reset_unk_locked()

            # ── RECORDING phase ───────────────────────────────────────────────
            elif _unk["phase"] == "recording":
                _unk["readings"].append({
                    "power_W":      round(power, 3),
                    "power_factor": round(pf, 4),
                })

                if len(_unk["readings"]) >= RECORD_TARGET:
                    name = _unk["pending_name"]
                    pws  = [r["power_W"] for r in _unk["readings"]]
                    power_rep = float(np.median(pws))

                    new_rows = [
                        {"power_W": r["power_W"], "power_factor": r["power_factor"], "appliance": name}
                        for r in _unk["readings"]
                    ]
                    fe = os.path.exists(TRAINING_CSV)
                    with open(TRAINING_CSV, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=TRAINING_FIELDNAMES)
                        if not fe:
                            w.writeheader()
                        w.writerows(new_rows)

                    cfg = read_wemo_config()
                    if name not in cfg.get("health_thresholds", {}):
                        lo = round(max(1.0, power_rep * 0.60), 1)
                        hi = round(power_rep * 1.40, 1)
                        cfg.setdefault("health_thresholds", {})[name] = [lo, hi]
                        with _config_lock:
                            with open(WEMO_CONFIG_FILE, "w") as f:
                                json.dump(cfg, f, indent=4)

                    _reset_unk_locked()
                    reset_smoother()
                    threading.Thread(target=retrain_model, daemon=True).start()

            # ── STABILIZING / ALERTED phase ───────────────────────────────────
            elif knn_result["is_anomaly"] and power > NO_LOAD_RECORD_THRESHOLD:

                if _unk["phase"] == "idle":
                    _unk["phase"]            = "stabilizing"
                    _unk["stability_buffer"] = [power]
                    _unk["power_snap"]       = power
                    _unk["pf_snap"]          = pf
                    _unk["i_snap"]           = i
                    print(f"[STABILITY] Anomaly seen ({power:.1f} W). Starting stability window.", flush=True)

                elif _unk["phase"] == "stabilizing":
                    _unk["stability_buffer"].append(power)
                    buf = _unk["stability_buffer"]
                    print(f"[STABILITY] Window {len(buf)}/{STABILITY_WINDOW} — "
                          f"spread={max(buf)-min(buf):.2f} W", flush=True)

                    if len(buf) >= STABILITY_WINDOW:
                        spread = max(buf) - min(buf)

                        if spread < STABILITY_VARIANCE_MAX:
                            _unk["phase"]      = "alerted"
                            _unk["power_snap"] = float(np.median(buf))
                            _unk["pf_snap"]    = pf
                            _unk["i_snap"]     = i
                            print(f"[STABILITY] PASSED — stable unknown load "
                                  f"({_unk['power_snap']:.1f} W).", flush=True)

                            # ── Decide whether to send WhatsApp ───────────────
                            # Only send if ALL three conditions are met:
                            #   1. Alert not already sent for this event
                            #   2. Not a manual UI trigger (suppress_alert flag)
                            #   3. Training Mode is OFF (user is not consciously training)
                            #   4. Browser has NOT pinged /live in last 30 seconds (user away)
                            with _training_mode_lock:
                                _tm_on = _training_mode_active

                            if _unk["alert_sent"]:
                                print("[ALERT] Already sent for this event — skipping.", flush=True)

                            elif _unk["suppress_alert"]:
                                print("[ALERT] Suppressed — manual UI trigger (start-training button).", flush=True)

                            elif _tm_on:
                                # User explicitly toggled Training Mode ON on the dashboard —
                                # they know about this load, no need to disturb them.
                                print("[ALERT] Suppressed — Training Mode is ON (user is consciously training).", flush=True)

                            elif _user_is_present():
                                # User is actively watching the dashboard — they
                                # will see the banner there, no WhatsApp needed.
                                print("[ALERT] Suppressed — browser dashboard is open.", flush=True)

                            else:
                                # ✅ User is away AND not in training mode — fire the WhatsApp!
                                _unk["alert_sent"] = True
                                print(f"[ALERT] User is AWAY and Training Mode is OFF — firing WhatsApp!", flush=True)
                                threading.Thread(
                                    target=send_whatsapp_unknown_alert,
                                    args=(_unk["power_snap"], _unk["pf_snap"], _unk["i_snap"]),
                                    daemon=True,
                                ).start()
                        else:
                            print(f"[STABILITY] FAILED — spread={spread:.2f} W > "
                                  f"{STABILITY_VARIANCE_MAX} W. Noise spike, resetting.", flush=True)
                            _reset_unk_locked()

                # alerted phase: waiting for user to name it — no action needed

            # ── Guard 2: KNN recognised load during stability window ──────────
            elif not knn_result["is_anomaly"] and _unk["phase"] == "stabilizing":
                print(f"[STABILITY] Identified as '{knn_result['label']}' during window. Aborting.", flush=True)
                _reset_unk_locked()

        # ─── Build UI text ────────────────────────────────────────────────────
        faults = analyse_faults(v, i, power, pf)
        phase  = _unk["phase"]

        if phase == "stabilizing":
            buf         = _unk["stability_buffer"]
            cnt         = len(buf)
            spread      = round(max(buf) - min(buf), 1) if buf else 0
            pred_text   = "Verifying new load…"
            health_text = (f"Stability check {cnt}/{STABILITY_WINDOW} "
                           f"({power:.1f} W, spread {spread} W). Do not unplug!")
            dev_status  = "warning"

        elif phase == "alerted":
            pred_text   = "Unknown Load Detected"
            health_text = f"Waiting for name… ({_unk['power_snap']:.1f} W)"
            dev_status  = "warning"

        elif phase == "recording":
            cnt         = len(_unk["readings"])
            pred_text   = f"Training '{_unk['pending_name']}'"
            health_text = f"Capturing {cnt}/{RECORD_TARGET} readings. Do not unplug!"
            dev_status  = "warning"

        else:
            pred_text   = knn_result["label"]
            health_text = knn_result["health"]
            dev_status  = knn_result["status"]

        if phase not in ("stabilizing", "alerted", "recording"):
            pred_text, dev_status, health_text, smooth_dist = smooth_prediction(
                pred_text, dev_status, health_text, knn_result["dist"])
            anomaly_dist = smooth_dist
        else:
            reset_smoother()
            anomaly_dist = knn_result["dist"]

        latest.update({
            "v": v, "i": i, "p": power,
            "e":      float(data.get("e",    latest["e"])),
            "f":      float(data.get("f",    latest["f"])),
            "pf":     pf,
            "relay":  esp_relay,
            "rssi":   int(data.get("rssi",   latest["rssi"])),
            "uptime": str(data.get("uptime", latest["uptime"])),
            "timestamp":        datetime.now().strftime("%H:%M:%S"),
            "prediction":       pred_text,
            "device_status":    dev_status,
            "health_msg":       health_text,
            "anomaly_distance": anomaly_dist,
            "recording_phase":  phase,
            "recording_count":  len(_unk["readings"]),
            "unknown_alert":    (phase == "alerted"),
            "unknown_power":    _unk["power_snap"] if phase == "alerted" else 0.0,
            "unknown_pf":       _unk["pf_snap"]    if phase == "alerted" else 0.0,
            **faults,
        })
        history.append({"t": latest["timestamp"], "p": power, "pf": pf})

        threading.Thread(target=log_sensor_data, args=(latest.copy(),), daemon=True).start()

        return jsonify({"ok": True, "classified_as": pred_text}), 200
    except Exception as exc:
        print(f"[ERROR] ingest: {exc}", flush=True)
        return jsonify({"ok": False, "error": str(exc)}), 400


# ─── Browser dashboard endpoint — updates presence detection ──────────────────
# Only marks user as present when called with ?user=1 (from the browser dashboard)
# ESP32 and automated tools call /live without ?user=1 so they never block alerts
@app.route("/live")
def live():
    global _last_dashboard_ping
    if request.args.get("user") == "1":
        with _ping_lock:
            _last_dashboard_ping = time.time()
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        print(f"[PING] Browser ping from IP: {ip}", flush=True)
    with _training_mode_lock:
        tm = _training_mode_active
    response = dict(latest)
    response["training_mode_active"] = tm
    return jsonify(response)


# ─── Training Mode toggle — suppresses WhatsApp alerts when user is consciously
#     adding/testing a new appliance, even if they step away from the dashboard ─
@app.route("/api/training-mode", methods=["GET"])
def get_training_mode():
    with _training_mode_lock:
        return jsonify({"ok": True, "active": _training_mode_active})

@app.route("/api/training-mode", methods=["POST"])
def set_training_mode():
    global _training_mode_active
    try:
        data   = request.get_json(force=True)
        active = bool(data.get("active", False))
        with _training_mode_lock:
            _training_mode_active = active
        print(f"[TRAINING MODE] {'ENABLED — WhatsApp alerts suppressed.' if active else 'DISABLED — normal alert logic resumed.'}", flush=True)
        return jsonify({"ok": True, "active": active}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


# ─── ESP32 relay endpoint — does NOT update presence detection ────────────────
# Point your ESP32 firmware at /api/relay-state-esp instead of /live
# so it never accidentally marks the user as "present".
@app.route("/api/relay-state-esp")
def relay_state_esp():
    relay_desired = latest.get("relay_desired", 0)
    return jsonify({
        "ok":            True,
        "relay":         relay_desired,
        "relay_desired": relay_desired,
        "state":         "ON" if relay_desired == 1 else "OFF",
    }), 200


@app.route("/history")
def get_history():
    return jsonify(list(history))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        res  = run_knn(float(data["power"]), float(data["pf"]))
        return jsonify({"prediction": res["label"], "device_status": res["status"],
                        "health_msg": res["health"], "anomaly_distance": res["dist"]})
    except Exception as exc:
        return jsonify({"prediction": "Error", "device_status": "danger",
                        "health_msg": str(exc), "anomaly_distance": None}), 400

@app.route("/api/wemo-ai", methods=["POST"])
def wemo_ai():
    if not MISTRAL_ENABLED:
        return jsonify({"ok": False, "error": "MISTRAL_API_KEY not set in .env"}), 503

    try:
        p    = latest.get("p",  0)
        pf   = latest.get("pf", 0)
        v    = latest.get("v",  0)
        i    = latest.get("i",  0)
        f    = latest.get("f",  50)
        e    = latest.get("e",  0)
        pred = latest.get("prediction", "Unknown")
        fmsg = latest.get("fault_msg", "") or "None"
        rssi = latest.get("rssi", 0)
        up   = latest.get("uptime", "—")

        kwh24 = round((p / 1000) * 24, 3)
        pkr24 = round(kwh24 * 30, 2)

        prompt = f"""You are WEMO Tech AI — an expert electrical energy assistant for WEMO, a smart home IoT device in Karachi, Pakistan.

Live sensor snapshot right now:
  Voltage : {v:.1f} V
  Current : {i:.3f} A
  Power   : {p:.1f} W
  Power Factor: {pf:.2f}
  Frequency: {f:.1f} Hz
  Energy accumulated: {e:.4f} kWh
  WiFi RSSI : {rssi} dBm
  Uptime    : {up}
  AI detected appliance: {pred}
  Fault flags: {fmsg}
  If running 24 h non-stop: {kwh24} kWh ≈ PKR {pkr24}

Respond as WEMO Tech AI. Cover:
1. ✅/⚠️/❌ Is the system working correctly?
2. Does the appliance classification match the readings?
3. Any electrical risks or faults?
4. 24-hour energy cost in PKR.
5. One practical recommendation for the user.

Keep it under 220 words. Use ✅ ⚠️ ❌ 💡 where helpful. Be direct and technical."""

        text = _mistral_chat(prompt)
        return jsonify({"ok": True, "reply": text, "kwh_24h": kwh24, "pkr_24h": pkr24}), 200

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/api/name-load", methods=["POST"])
def name_load():
    try:
        data = request.get_json(force=True)
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"ok": False, "error": "Name cannot be empty."}), 400

        with _unk_lock:
            _unk["pending_name"] = name
            _unk["readings"]     = []
            _unk["phase"]        = "recording"

        return jsonify({
            "ok": True,
            "message": f"Saving '{name}'... Recording 30 samples now. Leave device plugged in."
        }), 200

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/reset-training", methods=["POST"])
def reset_training():
    try:
        for f in [TRAINING_CSV, KNN_PATH, SCALER_PATH]:
            if os.path.exists(f): os.remove(f)
        with _unk_lock: _reset_unk_locked()
        reset_smoother()
        global _knn, _scaler, ML_READY
        with _ml_lock: _knn = _scaler = None; ML_READY = False
        latest.update({"prediction": "Retraining…", "device_status": "idle",
                        "recording_phase": "idle", "recording_count": 0,
                        "unknown_alert": False})
        def _do():
            seed_training_csv(overwrite=True)
            retrain_model()
        threading.Thread(target=_do, daemon=True).start()
        return jsonify({"ok": True, "message": "Reset done. Retraining in background."}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/training-data")
def training_data():
    try:
        if not os.path.exists(TRAINING_CSV):
            return jsonify({"classes": [], "total": 0})
        import pandas as pd
        df = pd.read_csv(TRAINING_CSV)
        df = df[df["power_W"] > 0.5]
        s  = (df.groupby("appliance")
                .agg(count=("power_W","count"), avg_W=("power_W","mean"), avg_pf=("power_factor","mean"))
                .reset_index().rename(columns={"appliance":"label"}))
        s["avg_W"] = s["avg_W"].round(2); s["avg_pf"] = s["avg_pf"].round(3)
        return jsonify({"classes": s.to_dict(orient="records"), "total": int(len(df))})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/start-training", methods=["POST"])
def start_training():
    try:
        cur = latest.get("p", 0)
        if cur < NO_LOAD_RECORD_THRESHOLD:
            return jsonify({"ok": False,
                            "error": f"No load ({cur:.1f} W). Plug in first."}), 400
        with _unk_lock:
            if _unk["phase"] in ("recording", "alerted", "stabilizing"):
                return jsonify({"ok": True, "message": "Already gathering data.",
                                "phase": _unk["phase"]}), 200

            _unk["phase"]          = "alerted"
            _unk["power_snap"]     = cur
            _unk["pf_snap"]        = latest.get("pf", 0)
            _unk["suppress_alert"] = True   # manual trigger — user is on dashboard

        reset_smoother()
        return jsonify({"ok": True, "message": f"Ready to train on {cur:.1f} W. Please name it."}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/cancel-training", methods=["POST"])
def cancel_training():
    with _unk_lock: _reset_unk_locked()
    latest.update({"recording_phase": "idle", "recording_count": 0,
                   "unknown_alert": False, "unknown_power": 0.0, "unknown_pf": 0.0})
    return jsonify({"ok": True, "message": "Cancelled."}), 200

@app.route("/api/retrain", methods=["POST"])
def force_retrain():
    threading.Thread(target=retrain_model, daemon=True).start()
    return jsonify({"ok": True, "message": "Retraining triggered. Model will hot-swap in a few seconds."}), 200

@app.route("/api/relay", methods=["POST"])
def relay_control():
    try:
        data  = request.get_json(force=True)
        state = str(data.get("state", "")).upper()
        if state not in ("ON", "OFF"):
            return jsonify({"ok": False, "error": "state must be ON or OFF"}), 400

        relay_val = 1 if state == "ON" else 0
        latest["relay_desired"] = relay_val
        save_relay_state(relay_val, user_set=True)
        return jsonify({"ok": True, "relay": state, "relay_state": relay_val}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/relay-state")
def relay_state():
    relay_desired = latest.get("relay_desired", 0)
    state = "ON" if relay_desired == 1 else "OFF"

    resp = {
        "ok":            True,
        "relay":         relay_desired,
        "state":         state,
        "relay_desired": relay_desired,
    }

    config = read_wifi_config()
    if config.get("pending_update", False):
        resp["wifi_update"] = True
        resp["ssid"]        = config.get("ssid")
        resp["password"]    = config.get("password")
        config["pending_update"] = False
        with _wifi_lock:
            with open(WIFI_CONFIG_JSON, "w") as f:
                json.dump(config, f, indent=2)

    return jsonify(resp), 200

@app.route("/api/delete-class", methods=["POST"])
def delete_class():
    try:
        data = request.get_json(force=True)
        name = str(data.get("name", "")).strip()
        if not name or not os.path.exists(TRAINING_CSV):
            return jsonify({"ok": False, "error": "Not found."}), 400
        import pandas as pd
        df     = pd.read_csv(TRAINING_CSV)
        before = len(df)
        df     = df[df["appliance"] != name]
        removed = before - len(df)
        if removed == 0:
            return jsonify({"ok": False, "error": f"'{name}' not found."}), 404
        df.to_csv(TRAINING_CSV, index=False)

        cfg = read_wemo_config()
        if name in cfg.get("health_thresholds", {}):
            del cfg["health_thresholds"][name]
        if name in cfg.get("class_power_floor", {}):
            del cfg["class_power_floor"][name]
        with _config_lock:
            with open(WEMO_CONFIG_FILE, "w") as f:
                json.dump(cfg, f, indent=4)

        reset_smoother()
        threading.Thread(target=retrain_model, daemon=True).start()
        return jsonify({"ok": True, "removed": removed,
                        "message": f"Deleted {removed} samples of '{name}'."}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

# ─── Data Logging ─────────────────────────────────────────────────────────────
DATA_LOG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_data_log.csv")
DATA_LOG_FIELDNAMES = ["timestamp", "voltage_V", "current_A", "power_W", "power_factor",
                       "energy_kWh", "relay_state", "rssi", "prediction", "device_status"]
_log_lock = threading.Lock()

def init_data_log():
    if not os.path.exists(DATA_LOG_CSV):
        with open(DATA_LOG_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DATA_LOG_FIELDNAMES)
            w.writeheader()

def log_sensor_data(sensor_dict):
    try:
        with _log_lock:
            file_exists = os.path.exists(DATA_LOG_CSV)
            is_empty    = not file_exists or os.path.getsize(DATA_LOG_CSV) == 0
            with open(DATA_LOG_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=DATA_LOG_FIELDNAMES)
                if is_empty:
                    w.writeheader()
                w.writerow({
                    "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "voltage_V":     round(float(sensor_dict.get("v",  0)), 1),
                    "current_A":     round(float(sensor_dict.get("i",  0)), 3),
                    "power_W":       round(float(sensor_dict.get("p",  0)), 2),
                    "power_factor":  round(float(sensor_dict.get("pf", 0)), 3),
                    "energy_kWh":    round(float(sensor_dict.get("e",  0)), 4),
                    "relay_state":   int(sensor_dict.get("relay", 0)),
                    "rssi":          int(sensor_dict.get("rssi",  0)),
                    "prediction":    str(sensor_dict.get("prediction",    "—")),
                    "device_status": str(sensor_dict.get("device_status", "—")),
                })
    except Exception:
        pass

@app.route("/api/data-logs", methods=["GET"])
def get_data_logs():
    try:
        limit = request.args.get("limit", default=500, type=int)
        if not os.path.exists(DATA_LOG_CSV) or os.path.getsize(DATA_LOG_CSV) == 0:
            return jsonify({"logs": [], "total": 0})

        with open(DATA_LOG_CSV, mode="r", encoding="utf-8", errors="replace") as f:
            reader   = csv.DictReader(f)
            all_rows = list(reader)

        if not all_rows:
            return jsonify({"logs": [], "total": 0})

        all_rows.reverse()
        total_count = len(all_rows)
        if limit > 0:
            all_rows = all_rows[:limit]

        def safe_float(val):
            try:   return float(val) if val else 0.0
            except: return 0.0

        logs = []
        for idx, row in enumerate(all_rows):
            logs.append({
                "id":         idx,
                "timestamp":  str(row.get("timestamp", "—")),
                "voltage":    safe_float(row.get("voltage_V")),
                "current":    safe_float(row.get("current_A")),
                "power":      safe_float(row.get("power_W")),
                "pf":         safe_float(row.get("power_factor")),
                "energy":     safe_float(row.get("energy_kWh")),
                "relay":      int(safe_float(row.get("relay_state"))),
                "rssi":       int(safe_float(row.get("rssi"))),
                "prediction": str(row.get("prediction", "—")),
                "status":     str(row.get("device_status", "—")),
            })

        return jsonify({"logs": logs, "total": total_count, "count": len(logs)})
    except Exception as exc:
        return jsonify({"error": str(exc), "logs": [], "total": 0}), 400

@app.route("/api/data-logs/clear", methods=["POST"])
def clear_data_logs():
    try:
        data   = request.get_json(force=True)
        action = data.get("action", "all")

        if not os.path.exists(DATA_LOG_CSV):
            return jsonify({"ok": True, "message": "No logs to clear."})

        with _log_lock:
            if action == "all":
                with open(DATA_LOG_CSV, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=DATA_LOG_FIELDNAMES)
                    w.writeheader()
                return jsonify({"ok": True, "message": "All data logs cleared."}), 200
            else:
                import pandas as pd
                df      = pd.read_csv(DATA_LOG_CSV)
                before  = len(df)
                df      = df[df["timestamp"] != action]
                removed = before - len(df)
                if removed > 0:
                    df.to_csv(DATA_LOG_CSV, index=False)
                    return jsonify({"ok": True, "message": "Deleted 1 log entry."}), 200
                else:
                    return jsonify({"ok": False, "error": "Log entry not found."}), 404
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/data-logs/export", methods=["GET"])
def export_data_logs():
    try:
        if not os.path.exists(DATA_LOG_CSV):
            return jsonify({"error": "No logs to export."}), 404
        from flask import send_file
        return send_file(DATA_LOG_CSV, as_attachment=True,
                         download_name=f"sensor_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         mimetype="text/csv")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

# ─── ML Configuration ─────────────────────────────────────────────────────────
@app.route("/api/wemo-config", methods=["GET"])
def get_wemo_config_api():
    return jsonify({"ok": True, "config": read_wemo_config()})

@app.route("/api/wemo-config", methods=["POST"])
def set_wemo_config_api():
    try:
        new_cfg = request.get_json(force=True)
        with _config_lock:
            with open(WEMO_CONFIG_FILE, "w") as f:
                json.dump(new_cfg, f, indent=4)
        return jsonify({"ok": True, "message": "ML Configuration updated successfully!"}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

# ─── WiFi Configuration ───────────────────────────────────────────────────────
WIFI_CONFIG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wifi_config.json")
_wifi_lock = threading.Lock()

def init_wifi_config():
    if not os.path.exists(WIFI_CONFIG_JSON):
        with open(WIFI_CONFIG_JSON, "w") as f:
            json.dump({
                "ssid": "Your_WiFi_SSID",
                "password": "Your_WiFi_Password",
                "configured": False,
                "last_updated": None,
            }, f, indent=2)

def read_wifi_config():
    try:
        if os.path.exists(WIFI_CONFIG_JSON):
            with open(WIFI_CONFIG_JSON, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"ssid": "", "password": "", "configured": False}

@app.route("/api/wifi-config", methods=["GET"])
def get_wifi_config():
    try:
        config = read_wifi_config()
        return jsonify({
            "ok":           True,
            "ssid":         config.get("ssid", ""),
            "configured":   config.get("configured", False),
            "last_updated": config.get("last_updated", None),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.route("/api/wifi-config", methods=["POST"])
def set_wifi_config():
    try:
        data     = request.get_json(force=True)
        ssid     = str(data.get("ssid",     "")).strip()
        password = str(data.get("password", "")).strip()

        if not ssid or not password:
            return jsonify({"ok": False, "error": "SSID and Password cannot be empty."}), 400

        with _wifi_lock:
            config = {
                "ssid":           ssid,
                "password":       password,
                "configured":     True,
                "pending_update": True,
                "last_updated":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(WIFI_CONFIG_JSON, "w") as f:
                json.dump(config, f, indent=2)

        return jsonify({
            "ok":          True,
            "message":     "WiFi config saved. Upload this configuration to your ESP32.",
            "ssid":        ssid,
            "last_updated": config["last_updated"],
        }), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

# ─── Health check — safe for AWS ALB, does NOT affect presence detection ──────
@app.route("/health")
def health_check():
    return jsonify({"ok": True}), 200

init_data_log()
init_wifi_config()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)