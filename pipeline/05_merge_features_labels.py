from __future__ import annotations

from pathlib import Path
import pandas as pd


FEATURES = Path("data/processed/features.csv")
LABELS = Path("data/processed/labels.csv")
OUT_FILE = Path("data/processed/ml_final_dataset.csv")


def main() -> None:
    if not FEATURES.exists():
        raise FileNotFoundError("Missing features.csv. Run 03_build_features.py first.")
    if not LABELS.exists():
        raise FileNotFoundError("Missing labels.csv. Run 04_build_labels.py first.")

    features = pd.read_csv(FEATURES, parse_dates=["month_date"])
    labels = pd.read_csv(LABELS, parse_dates=["month_date"])

    final_df = features.merge(labels, on=["cell_id", "month_date"], how="inner")

    # fill features only
    feature_cols = [c for c in final_df.columns if c not in ["cell_id", "month_date", "y_prob", "y_class", "next_month_max_mag"]]
    final_df[feature_cols] = final_df[feature_cols].fillna(0)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE, "| rows:", len(final_df))


if __name__ == "__main__":
    main()