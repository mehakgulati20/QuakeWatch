from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory

# Pipeline prediction script output
PRED_FILE = Path("outputs/predictions_latest_month.csv")

# UI directory
UI_DIR = Path("ui")

app = Flask(__name__, static_folder=None)


def ensure_predictions() -> None:
    """
    If predictions file doesn't exist, attempt to generate it by running pipeline step 08.
    """
    if PRED_FILE.exists():
        return

    # Minimal: run step 08 only (assumes models exist)
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "pipeline/08_predict_latest_month.py"])


@app.get("/api/predictions/latest")
def api_predictions_latest():
    ensure_predictions()
    df = pd.read_csv(PRED_FILE)
    # return as JSON records
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/live")
def api_live_quakes():
    """
    Live earthquakes from USGS for the last 48 hours (min magnitude 4.0).
    If it fails, return a small fallback list so UI still shows points.
    """
    from datetime import datetime, timedelta, timezone

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=48)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 4.0,
        "orderby": "time",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        out = []
        for f in data.get("features", []):
            prop = f.get("properties", {})
            geom = f.get("geometry", {})
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]
            mag = prop.get("mag")
            t_ms = prop.get("time")

            t_str = ""
            if t_ms:
                t_str = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

            out.append({
                "lat": float(lat),
                "lon": float(lon),
                "mag": float(mag) if mag is not None else None,
                "time": t_str,
                "type": "Live Quake (USGS 48h)"
            })

        # If USGS returns nothing (rare), still return empty (real data)
        return jsonify(out)

    except Exception as e:
        # IMPORTANT: print error so you can see it in terminal
        print("USGS live quake fetch failed:", repr(e))

        # ✅ fallback like your teammate so you always see live points
        fallback = [
            {"lat": 35.6895, "lon": 139.6917, "mag": 4.5, "time": "Fallback", "type": "Live Quake"},
            {"lat": -12.0464, "lon": -77.0428, "mag": 5.2, "time": "Fallback", "type": "Live Quake"},
        ]
        return jsonify(fallback)

# ---- Static UI ----
@app.get("/")
def index():
    return send_from_directory(UI_DIR, "earthquake_dashboard.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(UI_DIR, "styles.css")


@app.get("/main.js")
def js():
    return send_from_directory(UI_DIR, "main.js")


if __name__ == "__main__":
    UI_DIR.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=8000, debug=True)