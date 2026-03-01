from __future__ import annotations

from pathlib import Path
import pandas as pd


IN_FILE = Path("data/processed/earthquakes_gridded.csv")
OUT_FILE = Path("data/processed/labels.csv")


def assign_class(mag: float, threshold: float) -> int:
    if pd.isna(mag) or mag < threshold:
        return -1
    if threshold <= mag <= 6.9:
        return 0
    if 7.0 <= mag <= 7.9:
        return 1
    return 2


def main(threshold: float = 7.0) -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing {IN_FILE}. Run 02_make_grid.py first.")

    df = pd.read_csv(IN_FILE, parse_dates=["month_date"])
    df["Year"] = df["month_date"].dt.year
    df["Month"] = df["month_date"].dt.month

    monthly = df.groupby(["cell_id", "Year", "Month"])["magnitude"].max().reset_index()
    monthly = monthly.sort_values(["cell_id", "Year", "Month"])

    monthly["next_month_max_mag"] = monthly.groupby("cell_id")["magnitude"].shift(-1)
    monthly["y_prob"] = (monthly["next_month_max_mag"] >= threshold).astype(int)
    monthly["y_class"] = monthly["next_month_max_mag"].apply(lambda m: assign_class(m, threshold))

    labels = monthly.dropna(subset=["next_month_max_mag"]).copy()

    labels["month_date"] = pd.to_datetime(dict(year=labels["Year"], month=labels["Month"], day=1))
    labels = labels[["cell_id", "month_date", "y_prob", "y_class", "next_month_max_mag"]]

    labels.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE, "| rows:", len(labels), "| threshold:", threshold)


if __name__ == "__main__":
    main()