from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score


DATA = Path("data/processed/ml_final_dataset.csv")
PROB_MODEL = Path("models/xgb_prob.pkl")
FEAT_INFO = Path("models/feature_info.pkl")


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["month_date"]).sort_values("month_date").reset_index(drop=True)

    model = joblib.load(PROB_MODEL)
    feature_cols = joblib.load(FEAT_INFO)["feature_cols"]

    split = int(len(df) * 0.8)
    test = df.iloc[split:]
    X_test = test[feature_cols].fillna(0)
    y_test = test["y_prob"]

    probs = model.predict_proba(X_test)[:, 1]
    print("Prob model ROC-AUC:", roc_auc_score(y_test, probs))


if __name__ == "__main__":
    main()