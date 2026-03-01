from __future__ import annotations

from pathlib import Path
import pandas as pd


IN_FILE = Path("data/processed/earthquakes_gridded.csv")
OUT_FILE = Path("data/processed/features.csv")


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}. Run 02_make_grid.py first.")

    df = pd.read_csv(IN_FILE, parse_dates=["month_date"])
    df = df.sort_values(["cell_id", "month_date"]).reset_index(drop=True)

    feature_rows = []

    for cell_id, g in df.groupby("cell_id"):
        g = g.sort_values("month_date").reset_index(drop=True)

        for _, row in g.iterrows():
            month = row["month_date"]
            past = g[g["month_date"] < month]

            last_1m = past[past["month_date"] >= month - pd.DateOffset(months=1)]
            last_3m = past[past["month_date"] >= month - pd.DateOffset(months=3)]
            last_6m = past[past["month_date"] >= month - pd.DateOffset(months=6)]
            last_12m = past[past["month_date"] >= month - pd.DateOffset(months=12)]

            if len(past) == 0:
                months_since_last_quake = 999
            else:
                last_quake_date = past["month_date"].max()
                months_since_last_quake = (month.year - last_quake_date.year) * 12 + (month.month - last_quake_date.month)

            trend_3m_minus_6m_avg = len(last_3m) - (len(last_6m) / 2.0)
            max_mag_last_12m = 0 if len(last_12m) == 0 else float(last_12m["magnitude"].max())

            def safe_max(s): return 0 if s.empty else float(s.max())
            def safe_mean(s): return 0 if s.empty else float(s.mean())

            feature_rows.append({
                "cell_id": cell_id,
                "month_date": month,

                "count_last_1m": len(last_1m),
                "count_last_3m": len(last_3m),
                "count_last_6m": len(last_6m),

                "max_mag_last_3m": safe_max(last_3m["magnitude"]),
                "max_mag_last_6m": safe_max(last_6m["magnitude"]),
                "avg_mag_last_6m": safe_mean(last_6m["magnitude"]),

                "avg_depth_last_6m": safe_mean(last_6m["depth"]),
                "max_depth_last_6m": safe_max(last_6m["depth"]),
            })

            # optional columns if present
            if "sig" in g.columns:
                feature_rows[-1]["max_sig_last_3m"] = safe_max(last_3m["sig"])
            else:
                feature_rows[-1]["max_sig_last_3m"] = 0

            if "mmi" in g.columns:
                feature_rows[-1]["avg_mmi_last_3m"] = safe_mean(last_3m["mmi"])
            else:
                feature_rows[-1]["avg_mmi_last_3m"] = 0

            if "cdi" in g.columns:
                feature_rows[-1]["avg_cdi_last_3m"] = safe_mean(last_3m["cdi"])
            else:
                feature_rows[-1]["avg_cdi_last_3m"] = 0

            feature_rows[-1]["months_since_last_quake"] = months_since_last_quake
            feature_rows[-1]["trend_3m_minus_6m_avg"] = trend_3m_minus_6m_avg
            feature_rows[-1]["max_mag_last_12m"] = max_mag_last_12m

    feat = pd.DataFrame(feature_rows).fillna(0)
    feat.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE, "| rows:", len(feat))


if __name__ == "__main__":
    main()