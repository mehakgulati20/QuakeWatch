from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


DATA = Path("data/processed/ml_final_dataset.csv")
MODEL_OUT = Path("models/xgb_class.pkl")
INFO_OUT = Path("models/class_info.pkl")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Missing ml_final_dataset.csv. Run 05_merge_features_labels.py first.")

    df = pd.read_csv(DATA, parse_dates=["month_date"])

    # keep valid magnitude classes only
    df = df[df["y_class"] != -1].sort_values("month_date").reset_index(drop=True)

    exclude = {"cell_id", "month_date", "y_prob", "y_class", "next_month_max_mag"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].fillna(0)
    y = df["y_class"].astype(int)

    # time split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ✅ Remap classes based on what's actually in TRAIN
    train_classes = sorted(y_train.unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    y_train_mapped = y_train.map(class_to_idx)
    y_test_mapped = y_test.map(class_to_idx)

    # If test contains a class not seen in train, drop those rows (rare but possible)
    valid_test_mask = y_test_mapped.notna()
    X_test = X_test.loc[valid_test_mask].copy()
    y_test_mapped = y_test_mapped.loc[valid_test_mask].astype(int)

    num_classes = len(train_classes)

    # If only 1 class exists in train, training a classifier is meaningless
    if num_classes < 2:
        raise ValueError(
            f"Not enough classes in training split. Found only: {train_classes}. "
            "Try lowering threshold or using a non-time-based split for this model."
        )

    # ✅ Set objective based on number of classes
    if num_classes == 2:
        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
            num_class=num_classes,
        )

    model.fit(X_train, y_train_mapped)

    pred_mapped = model.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test_mapped, pred_mapped))
    print(classification_report(y_test_mapped, pred_mapped, digits=3))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    info = {
        "feature_cols": feature_cols,
        "class_to_idx": class_to_idx,   # e.g., {1:0, 2:1}
        "idx_to_class": idx_to_class,   # e.g., {0:1, 1:2}
        "class_mapping_human": {0: "6.0–6.9", 1: "7.0–7.9", 2: "8.0+"},
    }
    joblib.dump(info, INFO_OUT)

    print("Saved:", MODEL_OUT, "and", INFO_OUT)
    print("Classes in training split:", train_classes)
    print("Mapping used:", class_to_idx)


if __name__ == "__main__":
    main()