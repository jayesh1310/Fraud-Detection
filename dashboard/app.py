"""
Fraud Detection Dashboard Backend  (fixed + start/stop support)
"""

import subprocess, json, re, os, glob, threading, time
from collections import deque
from datetime import datetime
from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
COMPOSE_DIR     = os.path.join(BASE_DIR, "..", "fraud-detection-pipeline-main")
COMPOSE_FILE    = os.path.join(COMPOSE_DIR, "docker-compose.yml")
TEMP_FLAGGED    = os.path.join(BASE_DIR, "..", "temp_flagged_dash")

# ── In-memory state ───────────────────────────────────────────
MAX_LOG  = 200
producer_logs    = deque(maxlen=MAX_LOG)
consumer_logs    = deque(maxlen=MAX_LOG)
live_feed        = deque(maxlen=100)   # last 100 txns for the feed panel

# Cumulative counters (survives the deque cap)
total_seen  = 0
total_fraud = 0
total_legit = 0
counter_lock = threading.Lock()

# Pipeline status cache
pipeline_status = {"running": None, "last_checked": 0}

# ── Model metrics (from training run) ─────────────────────────
MODEL_METRICS = {
    "auc_roc":          0.9821,
    "auc_pr":           0.7634,
    "f1_score":         0.9412,
    "fraud_catch_rate": 87.43,
    "false_alarm_rate": 0.0023,
    "num_trees":        100,
    "max_depth":        10,
    "features": ["amount","hour_of_day","is_night_transaction",
                 "geo_risk_score","lat","lon","merchant_category","amount_bucket"],
}

# ── Helpers ───────────────────────────────────────────────────
def _run(cmd, cwd=None, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, encoding='utf-8', timeout=timeout, cwd=cwd)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return str(e)

def _run_compose(subcmd):
    """Run a docker compose command in the compose directory."""
    return _run(f'docker compose {subcmd}', cwd=COMPOSE_DIR, timeout=90)

# ── Log parser ────────────────────────────────────────────────
TXN_RE = re.compile(
    r'\[Producer\] Sent \[(?P<label>FRAUD|legit)\] transaction '
    r'(?P<tid>[a-f0-9\-]{36}) \u2014 \$(?P<amount>[\d.]+)'
)

def parse_producer_line(line):
    line = line.strip()
    if not line:
        return None
    m = TXN_RE.search(line)
    if m:
        return {
            "type":   "transaction",
            "label":  m.group("label"),
            "tid":    m.group("tid"),
            "amount": float(m.group("amount")),
            "time":   datetime.now().isoformat(),
            "raw":    line,
        }
    return {"type": "log", "raw": line, "time": datetime.now().isoformat()}

# ── Background poller ─────────────────────────────────────────
def poll_logs():
    """
    Tail docker logs using --since to get only NEW lines each cycle.
    Maintains cumulative counters.
    """
    global total_seen, total_fraud, total_legit

    # Track the timestamp of the last line we processed
    last_producer_ts = None
    last_consumer_ts = None

    # Seed with recent history on first run (last 200 lines)
    first_run = True

    while True:
        try:
            # ── Producer logs ──────────────────────────────────
            if first_run:
                raw_p = _run("docker logs fraud-producer --tail 200 --timestamps 2>&1")
            else:
                since_arg = f'--since {last_producer_ts}' if last_producer_ts else '--tail 5'
                raw_p = _run(f"docker logs fraud-producer {since_arg} --timestamps 2>&1")

            for raw_line in raw_p.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                # Docker --timestamps format: "2026-08-13T03:44:06.123456789Z  actual message"
                parts = raw_line.split(None, 1)
                if len(parts) == 2 and parts[0].endswith('Z'):
                    ts_str, msg = parts[0], parts[1]
                else:
                    ts_str, msg = None, raw_line

                parsed = parse_producer_line(msg)
                if parsed:
                    producer_logs.append(parsed)
                    if parsed["type"] == "transaction":
                        live_feed.append(parsed)
                        with counter_lock:
                            total_seen  += 1
                            if parsed["label"] == "FRAUD":
                                total_fraud += 1
                            else:
                                total_legit += 1

                if ts_str:
                    last_producer_ts = ts_str

            # ── Consumer logs ──────────────────────────────────
            if first_run:
                raw_c = _run("docker logs fraud-consumer --tail 100 --timestamps 2>&1")
            else:
                since_arg = f'--since {last_consumer_ts}' if last_consumer_ts else '--tail 5'
                raw_c = _run(f"docker logs fraud-consumer {since_arg} --timestamps 2>&1")

            for raw_line in raw_c.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parts = raw_line.split(None, 1)
                if len(parts) == 2 and parts[0].endswith('Z'):
                    ts_str, msg = parts[0], parts[1]
                else:
                    ts_str, msg = None, raw_line

                consumer_logs.append({"raw": msg, "time": datetime.now().isoformat()})
                if ts_str:
                    last_consumer_ts = ts_str

            first_run = False

        except Exception as e:
            print(f"[poller] error: {e}")

        time.sleep(3)


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/containers")
def containers():
    ps_out = _run('docker ps --format "{{json .}}"')
    result = []
    for line in ps_out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj  = json.loads(line)
            stat = obj.get("Status", "")
            result.append({
                "name":    obj.get("Names", ""),
                "status":  stat,
                "ports":   obj.get("Ports", ""),
                "image":   obj.get("Image", ""),
                "running": "Up" in stat,
            })
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/transactions")
def transactions():
    with counter_lock:
        ts = total_seen
        tf = total_fraud
        tl = total_legit
    feed = list(live_feed)[-30:]
    return jsonify({
        "feed":        feed,
        "total_seen":  ts,
        "fraud_count": tf,
        "legit_count": tl,
        "fraud_rate":  round(tf / ts * 100, 2) if ts > 0 else 0,
    })


@app.route("/api/producer/logs")
def producer_log_api():
    return jsonify(list(producer_logs)[-60:])


@app.route("/api/consumer/logs")
def consumer_log_api():
    return jsonify(list(consumer_logs)[-60:])


@app.route("/api/model")
def model_metrics():
    metrics = dict(MODEL_METRICS)
    try:
        import pandas as pd
        train_dir = os.path.join(BASE_DIR, "..", "training_data")
        files = glob.glob(os.path.join(train_dir, "*.parquet"))
        if files:
            df = pd.read_parquet(files[0], columns=["is_fraud"])
            metrics["train_samples"]    = len(df)
            metrics["fraud_in_training"] = int(df["is_fraud"].sum())
    except Exception:
        pass
    return jsonify(metrics)


@app.route("/api/fraud-caught")
def fraud_caught():
    try:
        os.makedirs(TEMP_FLAGGED, exist_ok=True)
        _run(f'docker cp fraud-consumer:/data/output/fraud_flagged "{TEMP_FLAGGED}"')

        import pandas as pd
        files = glob.glob(os.path.join(TEMP_FLAGGED, "**", "*.parquet"), recursive=True)
        if not files:
            return jsonify({"records": [], "total": 0})

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                pass
        if not dfs:
            return jsonify({"records": [], "total": 0})

        df = pd.concat(dfs, ignore_index=True)
        if "flagged_fraud" in df.columns:
            flagged = df[df["flagged_fraud"] == True]
        else:
            flagged = df

        records = []
        for _, row in flagged.tail(25).iterrows():
            records.append({
                "transaction_id":    str(row.get("transaction_id", "")),
                "amount":            float(row.get("amount", 0)),
                "merchant_category": str(row.get("merchant_category", "")),
                "is_fraud":          bool(row.get("is_fraud", False)),
                "flagged_fraud":     bool(row.get("flagged_fraud", True)),
                "hour_of_day":       int(row.get("hour_of_day", 0)),
                "geo_risk_score":    float(row.get("geo_risk_score", 0)),
            })
        return jsonify({"records": records, "total": len(flagged)})
    except Exception as e:
        return jsonify({"records": [], "total": 0, "error": str(e)})


# ── Pipeline control ──────────────────────────────────────────

@app.route("/api/pipeline/status")
def pipeline_status_route():
    ps_out = _run('docker ps --format "{{.Names}}"')
    names  = [n.strip() for n in ps_out.splitlines() if n.strip()]
    core   = ["fraud-producer", "fraud-consumer", "kafka", "zookeeper"]
    running = any(c in names for c in core)
    return jsonify({"running": running, "containers": names})


@app.route("/api/pipeline/start", methods=["POST"])
def pipeline_start():
    out = _run_compose("up -d")
    return jsonify({"ok": True, "output": out[-800:]})


@app.route("/api/pipeline/stop", methods=["POST"])
def pipeline_stop():
    out = _run_compose("down")
    return jsonify({"ok": True, "output": out[-800:]})


# ── Start ─────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=poll_logs, daemon=True)
    t.start()
    print("[Dashboard] Backend running on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
