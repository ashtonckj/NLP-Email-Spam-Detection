import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import FEATURES, TARGET, run_cleaning_pipeline

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILES = ["CEAS_08.csv", "Enron.csv", "Nazario.csv"]
SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
SAVE_DIR = ROOT_DIR / "src" / "saved_models"
OUT_DIR = ROOT_DIR / "output"
KEEP_COLS = FEATURES + TARGET


# Spam vs ham rate by message length
def fig_spam_ham_rate_by_length():
    df_spam = pd.read_csv(SPAM_CSV)
    df_spam["WordCount"] = df_spam["Message"].str.split().str.len()

    buckets = [(0, 5), (6, 15), (16, 50), (51, 100), (101, 300), (300, np.inf)]
    labels = ["0–5", "6–15", "16–50", "51–100", "101–300", "300+"]

    spam_rates, ham_rates, counts = [], [], []
    for low, high in buckets:
        s = df_spam[(df_spam.WordCount >= low) & (df_spam.WordCount <= high)]
        n = len(s)
        spam_rate = s.Category.mean() * 100 if n else 0
        spam_rates.append(spam_rate)
        ham_rates.append(100 - spam_rate if n else 0)
        counts.append(n)

    x = np.arange(len(labels))
    width = 0.5

    _fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, ham_rates, width, label="Ham", color="#4aa331", edgecolor="black", linewidth=0.6)
    ax.bar(x, spam_rates, width, bottom=ham_rates, label="Spam", color="#be1c10", edgecolor="black", linewidth=0.6)

    for i, (h, s) in enumerate(zip(ham_rates, spam_rates)):
        if h > 4:
            ax.text(i, h / 2, f"{h:.1f}%", ha="center", va="center", fontsize=8, color="white")
        if s > 4:
            ax.text(i, h + s / 2, f"{s:.1f}%", ha="center", va="center", fontsize=8, color="white")

    for i, n in enumerate(counts):
        ax.text(i, 103, f"n={n:,}", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Message length (words)")
    ax.set_ylabel("Proportion of messages (%)")
    ax.set_title("Spam and Ham Proportion by Message Length")
    ax.set_ylim(0, 122)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "spam_ham_message_length.png", dpi=200)


# Dataset merging effect on spam percentage with message length 
def fig_merge_effect():
    df_ceas = pd.read_csv(RAW_DIR / "CEAS_08.csv")
    df_ceas = run_cleaning_pipeline(df_ceas)
    df_ceas["WordCount"] = df_ceas["Message"].str.split().str.len()
    ceas_short = df_ceas[df_ceas["WordCount"] <= 15]
    ceas_rate = ceas_short["Category"].mean() if len(ceas_short) else float("nan")
    
    df_spam = pd.read_csv(SPAM_CSV)
    df_spam["WordCount"] = df_spam["Message"].str.split().str.len()
    spam_short = df_spam[df_spam["WordCount"] <= 15]
    spam_rate = spam_short["Category"].mean() if len(spam_short) else float("nan")
    
    _fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["CEAS_08 only", "CEAS_08 + Enron + Nazario"], [ceas_rate, spam_rate], color=["#c44e52", "#55a868"], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, [97.5, 76.5]):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Short messages (≤15 words) labelled spam (%)")
    ax.set_title("Effect of merging Enron on length confound")
    ax.set_ylim(0, 110)
    plt.tight_layout(); plt.savefig(OUT_DIR / "fig_merge_effect.png", dpi=200)


if __name__ == "__main__":
    fig_merge_effect()