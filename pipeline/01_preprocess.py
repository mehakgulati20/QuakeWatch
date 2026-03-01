from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/earthquake_data_tsunami.csv")
OUT_FILE = Path("data/processed/earthquakes_clean_monthly.csv")


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing {RAW_FILE}. Run 00_download_data.py first.")

    df = pd.read_csv(RAW_FILE)

    # Defensive column checks
    required = ["Year", "Month", "latitude", "longitude", "magnitude", "depth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df["month_date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01",
        errors="coerce",
    )

    cols = ["month_date", "latitude", "longitude", "magnitude", "depth"]
    # Optional columns if present
    for c in ["sig", "mmi", "cdi"]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].dropna(subset=["month_date", "latitude", "longitude", "magnitude", "depth"]).copy()
    out.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE, "| rows:", len(out))


if __name__ == "__main__":
    main()