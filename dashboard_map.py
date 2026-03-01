from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests
import plotly.graph_objects as go


PRED_FILE = Path("outputs/predictions_latest_month.csv")
OUT_HTML = Path("ui/earthquake_dashboard_generated.html")


def fetch_live_quakes_usgs(min_mag: float = 4.0) -> pd.DataFrame:
    """
    Lightweight live feed (recommended): USGS GeoJSON (last 24 hours).
    """
    # Options:
    #  - 4.5_day.geojson, 2.5_day.geojson, 1.0_day.geojson, etc.
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{}.geojson".format(
        "4.0_day" if min_mag <= 4.0 else "4.5_day"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        rows = []
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
                t_str = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )

            if mag is None:
                continue
            if float(mag) < float(min_mag):
                continue

            rows.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "mag": float(mag),
                    "time": t_str,
                    "type": "Live Quake (USGS)",
                }
            )

        return pd.DataFrame(rows)

    except Exception:
        # If the network fails, just return empty (dashboard still renders)
        return pd.DataFrame(columns=["lat", "lon", "mag", "time", "type"])


def parse_cell_id(cell_id: str) -> tuple[float, float]:
    """
    cell_id must be 'lat_lon' like '10.0_72.5'
    """
    lat_str, lon_str = cell_id.split("_")
    return float(lat_str), float(lon_str)


def class_label(c: int) -> str:
    mapping = {0: "6.0–6.9", 1: "7.0–7.9", 2: "8.0+"}
    return mapping.get(int(c), "N/A")


def build_map(pred_df: pd.DataFrame, live_df: pd.DataFrame) -> go.Figure:
    # Ensure numeric
    pred_df["risk_prob"] = pd.to_numeric(pred_df["risk_prob"], errors="coerce").fillna(0)
    pred_df["predicted_quake"] = pd.to_numeric(pred_df["predicted_quake"], errors="coerce").fillna(0).astype(int)
    pred_df["predicted_class"] = pd.to_numeric(pred_df["predicted_class"], errors="coerce").fillna(-1).astype(int)

    # Parse lat/lon from cell_id
    lats, lons = [], []
    for cid in pred_df["cell_id"]:
        la, lo = parse_cell_id(cid)
        lats.append(la)
        lons.append(lo)
    pred_df["lat"] = lats
    pred_df["lon"] = lons

    fig = go.Figure()

    # 1) Prediction risk layer
    fig.add_trace(
        go.Scattermapbox(
            lat=pred_df["lat"],
            lon=pred_df["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=14,
                color=pred_df["risk_prob"],
                colorscale="Viridis",
                opacity=0.75,
                showscale=True,
                colorbar=dict(title="Risk", thickness=20, x=0.9),
            ),
            text=pred_df.apply(
                lambda r: (
                    f"<b>Cell:</b> {r['cell_id']}"
                    f"<br><b>Risk:</b> {float(r['risk_prob']):.2f}"
                    f"<br><b>Predicted quake:</b> {int(r['predicted_quake'])}"
                    f"<br><b>Magnitude:</b> {class_label(int(r['predicted_class']))}"
                ),
                axis=1,
            ),
            hoverinfo="text",
            name="Predicted Risk Zones",
        )
    )

    # 2) Alert overlay (predicted_quake == 1)
    alerts = pred_df[pred_df["predicted_quake"] == 1]
    if not alerts.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=alerts["lat"],
                lon=alerts["lon"],
                mode="markers+text",
                marker=go.scattermapbox.Marker(size=12, color="white"),
                text=alerts["predicted_class"].apply(lambda c: f"⚠️ {class_label(int(c))}"),
                textposition="top center",
                hoverinfo="none",
                name="Alert Zones",
            )
        )

    # 3) Live quakes overlay
    if live_df is not None and not live_df.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=live_df["lat"],
                lon=live_df["lon"],
                mode="markers",
                marker=go.scattermapbox.Marker(size=9, color="cyan", opacity=0.9),
                text=live_df.apply(
                    lambda r: f"<b>LIVE QUAKE</b><br>Mag: {r['mag']}<br>Time: {r['time']}",
                    axis=1,
                ),
                hoverinfo="text",
                name="Live Quakes (24h)",
            )
        )

    # Layout (dark premium)
    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=20, lon=0), zoom=1.5),
        margin=dict(r=0, t=60, l=0, b=0),
        title=dict(text="🌍 Global Earthquake Risk & Live Monitoring (Standalone)", x=0.05),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


def main() -> None:
    if not PRED_FILE.exists():
        raise FileNotFoundError(
            f"{PRED_FILE} not found. Run pipeline/08_predict_latest_month.py first."
        )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(PRED_FILE)
    live_df = fetch_live_quakes_usgs(min_mag=4.0)

    fig = build_map(pred_df, live_df)
    fig.write_html(OUT_HTML)

    print("✅ Standalone dashboard saved to:", OUT_HTML)


if __name__ == "__main__":
    main()