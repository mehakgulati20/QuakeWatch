from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np


DATA = Path("data/processed/ml_final_dataset.csv")
MODEL_OUT = Path("models/xgb_prob.pkl")
INFO_OUT = Path("models/feature_info.pkl")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Missing ml_final_dataset.csv. Run 05_merge_features_labels.py first.")

    df = pd.read_csv(DATA, parse_dates=["month_date"]).sort_values("month_date").reset_index(drop=True)

    exclude = {"cell_id", "month_date", "y_prob", "y_class", "next_month_max_mag"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].fillna(0)
    y = df["y_prob"]

    cutoff = df["month_date"].quantile(0.8)
    train_mask = df["month_date"] < cutoff
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Cutoff date:", cutoff)
    print("Train rows:", len(X_train), "| Test rows:", len(X_test))

    print("\n===== DATASET SANITY CHECKS =====")
    print("Total rows:", len(df))
    print("Date range:", df["month_date"].min(), "→", df["month_date"].max())

    print("\nTrain label distribution (y_prob):")
    print(y_train.value_counts(dropna=False))
    print("\nTest label distribution (y_prob):")
    print(y_test.value_counts(dropna=False))

    # How imbalanced is it?
    train_pos = y_train.mean()
    test_pos = y_test.mean()
    print(f"\nTrain positive rate: {train_pos:.6f}")
    print(f"Test positive rate : {test_pos:.6f}")

    # Ensure no NaNs in X
    print("\nAny NaNs in X_train?", X_train.isna().any().any())
    print("Any NaNs in X_test ?", X_test.isna().any().any())

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    print("\n===== BASELINES =====")
    # baseline: always predict the mean probability
    baseline_probs = np.full(shape=len(y_test), fill_value=y_train.mean())
    print("Baseline ROC-AUC (constant mean):", roc_auc_score(y_test, baseline_probs) if len(y_test.unique()) > 1 else "N/A")

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    probs = 1 - probs
    preds = (probs >= 0.5).astype(int)

    from sklearn.metrics import confusion_matrix, classification_report

    print("\n===== MODEL OUTPUT CHECKS =====")
    print("Predicted prob mean:", probs.mean())
    print("Predicted prob min :", probs.min())
    print("Predicted prob max :", probs.max())

    auc = roc_auc_score(y_test, probs) if len(y_test.unique()) > 1 else float("nan")
    print("\nROC-AUC:", auc)

    # Check if model is inverted
    flipped_auc = roc_auc_score(y_test, 1 - probs) if len(y_test.unique()) > 1 else float("nan")
    print("Flipped ROC-AUC (1 - probs):", flipped_auc)

    # Confusion matrix at a few thresholds
    for t in [0.3, 0.5, 0.7]:
        pred = (probs >= t).astype(int)
        print(f"\n--- Threshold {t} ---")
        print("alerts:", int(pred.sum()))
        print("confusion matrix:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, digits=3, zero_division=0))


    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    info = {"feature_cols": feature_cols}
    joblib.dump(info, INFO_OUT)

    print("Saved:", MODEL_OUT, "and", INFO_OUT)


if __name__ == "__main__":
    main()