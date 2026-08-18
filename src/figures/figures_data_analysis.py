import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import run_cleaning_pipeline

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILES = ["CEAS_08.csv", "Enron.csv", "Nazario.csv"]
SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
OUT_DIR = ROOT_DIR / "output"


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
    plt.savefig(OUT_DIR / "fig_spam_ham_message_length.png", dpi=200)


# Dataset merging effect on spam percentage with message length 
def fig_merge_effect():
    df_ceas = pd.read_csv(RAW_DIR / "CEAS_08.csv")
    df_ceas = run_cleaning_pipeline(df_ceas)
    df_ceas["WordCount"] = df_ceas["Message"].str.split().str.len()
    ceas_short = df_ceas[df_ceas["WordCount"] <= 15]
    ceas_rate = ceas_short["Category"].mean() * 100 if len(ceas_short) else float("nan")

    df_spam = pd.read_csv(SPAM_CSV)
    df_spam["WordCount"] = df_spam["Message"].str.split().str.len()
    spam_short = df_spam[df_spam["WordCount"] <= 15]
    spam_rate = spam_short["Category"].mean() * 100 if len(spam_short) else float("nan")

    rates = [ceas_rate, spam_rate]
    _fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["CEAS_08 only", "CEAS_08 + Enron + Nazario"], rates, color=["#c44e52", "#55a868"], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Short messages (≤15 words) labelled spam (%)")
    ax.set_title("Effect of merging Enron + Nazario")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_effect_of_merging.png", dpi=200)


# Spam vs ham counts in spam.csv
def fig_class_distribution():
    df = pd.read_csv(SPAM_CSV)
    counts = df["Category"].value_counts().sort_index()
    labels = {0: "Ham", 1: "Spam"}
    total = counts.sum()
    
    _fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar([labels[i] for i in counts.index], counts.values, color=["#55a868", "#c44e52"])
    ax.set_ylim(0, counts.values.max() * 1.15)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,}\n({h / total:.1%})", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom")
    ax.set_ylabel("Number of messages")
    ax.set_title("Class Distribution (Merged Dataset)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_spam_ham_counts.png")


# Dataset Composition
def fig_dataset_composition():
    per_file = {}
    for filename in RAW_FILES:
        df = pd.read_csv(RAW_DIR / filename)
        df = df.drop(columns="Unnamed: 0", errors="ignore")
        df = run_cleaning_pipeline(df)
        per_file[filename] = len(df)

    df_spam = pd.read_csv(SPAM_CSV)

    n_total = len(df_spam)
    n_spam = int((df_spam["Category"] == 1).sum())
    n_ham = n_total - n_spam

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(per_file.keys(), per_file.values(), color="#4C72B0", edgecolor="black", linewidth=0.6)
    axes[0].set_title("Records per Source File")
    axes[0].set_ylabel("Number of Emails")
    axes[0].tick_params(axis="x", rotation=20)
    for i, (k, v) in enumerate(per_file.items()):
        axes[0].text(i, v + max(per_file.values()) * 0.01, f"{v:,}", ha="center", fontsize=8)

    axes[1].bar(["Ham", "Spam"], [n_ham, n_spam], color=["#55a868", "#c44e52"], edgecolor="black", linewidth=0.6)
    axes[1].set_title("Class Distribution (Merged Dataset)")
    axes[1].set_ylabel("Number of Emails")
    for i, v in enumerate([n_ham, n_spam]):
        axes[1].text(i, v + max(n_ham, n_spam) * 0.01, f"{v:,}", ha="center", fontsize=8)

    fig.tight_layout()
    plt.savefig(OUT_DIR / "fig_dataset_composition.png", dpi=300)


# Vocabulary association with spam and ham
VOCAB_PHRASES = ["click here", "guarantee", "limited time", "free", "invoice", "schedule", "meeting"]
def fig_vocab_association():
    df_spam = pd.read_csv(SPAM_CSV)
    overall_rate = df_spam["Category"].mean()
    msg = df_spam["Message"].astype(str)

    rows = []
    for phrase in VOCAB_PHRASES:
        mask = msg.str.contains(phrase, regex=False)
        if mask.sum() == 0:
            continue
        rows.append((phrase, df_spam.loc[mask, "Category"].mean(), int(mask.sum())))

    assoc = pd.DataFrame(rows, columns=["phrase", "spam_rate", "count"]).sort_values("spam_rate")

    _fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#c44e52" if v > overall_rate else "#55a868" for v in assoc["spam_rate"]]
    ax.barh(assoc["phrase"], assoc["spam_rate"], color=colors)
    ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1, label=f"Overall spam rate ({overall_rate:.1%})")
    ax.set_xlabel("Spam rate among messages containing phrase")
    ax.set_xlim(0, 1)
    ax.set_title("Spam Association of Selected Phrases")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_vocab_association.png")
    print(assoc)


# Confusion Matrics
def fig_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    _fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")

    threshold = cm.max() * 0.6  # cells darker than this get white text

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    ax.set_xticks([1, 0])
    ax.set_yticks([1, 0])
    ax.set_xticklabels(["Spam", "Ham"])
    ax.set_yticklabels(["Spam", "Ham"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name.upper()}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fig_confusion_{name}.png", dpi=300)


if __name__ == "__main__":
    fig_vocab_association()