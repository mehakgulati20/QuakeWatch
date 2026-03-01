from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


IN_FILE = Path("data/processed/earthquakes_clean_monthly.csv")
OUT_FILE = Path("data/processed/earthquakes_gridded.csv")
BIN = 0.5


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}. Run 01_preprocess.py first.")

    df = pd.read_csv(IN_FILE, parse_dates=["month_date"])

    df["grid_lat"] = np.floor(df["latitude"] / BIN) * BIN
    df["grid_lon"] = np.floor(df["longitude"] / BIN) * BIN

    # ✅ ONE consistent format everywhere:
    df["cell_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    df.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE, "| rows:", len(df))


if __name__ == "__main__":
    main()