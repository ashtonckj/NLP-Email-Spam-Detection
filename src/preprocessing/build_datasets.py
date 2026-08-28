import sys
from pathlib import Path

import pandas as pd

# Navigate up 2 levels from testing.py to find the project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import FEATURES, TARGET, run_cleaning_pipeline

KEEP_COLS = FEATURES + TARGET
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILES = ["CEAS_08.csv", "Enron.csv", "Nazario.csv"]
OUT_DIR = ROOT_DIR / "data" / "processed"


def build_datasets():
    frames = []
    for filename in RAW_FILES:
        df = pd.read_csv(RAW_DIR / filename)
        df = df.drop(columns="Unnamed: 0", errors="ignore")
        df = df[KEEP_COLS]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined raw rows: {len(combined)}")

    spam_df = run_cleaning_pipeline(combined)
    spam_df.to_csv(OUT_DIR / "spam.csv", index=False)
    print(f"Saved spam.csv, final rows: {len(spam_df)}")


if __name__ == "__main__":
    build_datasets()
