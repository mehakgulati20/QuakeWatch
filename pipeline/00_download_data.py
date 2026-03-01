from __future__ import annotations

import os
from pathlib import Path
import kagglehub


DATASET = "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
TARGET_FILENAME = "earthquake_data_tsunami.csv"


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(DATASET)
    path = Path(path)

    # Find the CSV inside the downloaded folder
    csv_candidates = list(path.rglob(TARGET_FILENAME))
    if not csv_candidates:
        # fallback: any csv
        csv_candidates = list(path.rglob("*.csv"))

    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in downloaded dataset folder: {path}")

    src = csv_candidates[0]
    dst = raw_dir / TARGET_FILENAME
    dst.write_bytes(src.read_bytes())

    print("Downloaded dataset to:", dst)


if __name__ == "__main__":
    main()