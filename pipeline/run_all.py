from __future__ import annotations

import subprocess
import sys

STEPS = [
    "pipeline/00_download_data.py",
    "pipeline/01_preprocess.py",
    "pipeline/02_make_grid.py",
    "pipeline/03_build_features.py",
    "pipeline/04_build_labels.py",
    "pipeline/05_merge_features_labels.py",
    "pipeline/06_train_xgb_prob_model.py",
    "pipeline/07_train_xgb_class_model.py",
    "pipeline/08_predict_latest_month.py",
    "pipeline/09_evaluate_models.py",
]


def main() -> None:
    for step in STEPS:
        print("\n==============================")
        print("RUNNING:", step)
        print("==============================")
        subprocess.check_call([sys.executable, step])


if __name__ == "__main__":
    main()