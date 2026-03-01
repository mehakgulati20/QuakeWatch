from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


DATA = Path("data/processed/ml_final_dataset.csv")
PROB_MODEL = Path("models/xgb_prob.pkl")
FEAT_INFO = Path("models/feature_info.pkl")
CLASS_MODEL = Path("models/xgb_class.pkl")
CLASS_INFO = Path("models/class_info.pkl")

OUT = Path("outputs/predictions_latest_month.csv")


def main(threshold: float = 0.3) -> None:
    for p in [DATA, PROB_MODEL, FEAT_INFO]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run pipeline up to training first.")

    df = pd.read_csv(DATA, parse_dates=["month_date"])
    latest_month = df["month_date"].max()
    latest = df[df["month_date"] == latest_month].copy()

    prob_model = joblib.load(PROB_MODEL)
    feat_info = joblib.load(FEAT_INFO)
    feature_cols = feat_info["feature_cols"]

    X = latest[feature_cols].fillna(0)
    risk_prob = 1 - prob_model.predict_proba(X)[:, 1]

    latest["risk_prob"] = risk_prob
    latest["predicted_quake"] = (latest["risk_prob"] >= threshold).astype(int)

    # magnitude class only for predicted_quake==1 (optional)
    latest["predicted_class"] = -1
    if CLASS_MODEL.exists() and CLASS_INFO.exists():
        class_model = joblib.load(CLASS_MODEL)
        class_info = joblib.load(CLASS_INFO)
        class_feature_cols = class_info["feature_cols"]

        high = latest[latest["predicted_quake"] == 1].copy()
        if len(high) > 0:
            Xh = high[class_feature_cols].fillna(0)
            pred_idx = class_model.predict(Xh)
            idx_to_class = class_info["idx_to_class"]
            pred_original = [idx_to_class.get(int(i), idx_to_class.get(str(int(i)), -1)) for i in pred_idx]
            latest.loc[high.index, "predicted_class"] = pred_original

    out = latest[["cell_id", "month_date", "risk_prob", "predicted_quake", "predicted_class"]].copy()

    out["month_date"] = pd.to_datetime(out["month_date"]).dt.strftime("%Y-%m-%d")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("Saved:", OUT, "| latest_month:", latest_month.date(), "| rows:", len(out))


if __name__ == "__main__":
    main()