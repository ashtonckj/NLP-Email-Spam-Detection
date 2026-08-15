# Generates every figure/table the report (BMCS2074 assignment) needs for
# Chapter 3 (dataset description & preliminary analysis) and Chapter 4 (results).
#
# Run this AFTER:
#   1. build_datasets.py      -> produces data/processed/spam.csv
#   2. lstm_cnn.py             -> trains LSTM+CNN, saves src/saved_models/lstm_cnn_metrics.json
#   3. nb_lr_rf.py              -> trains NB/LR/RF + ensemble, saves nb_lr_rf_metrics.json
#   4. nb_svm.py                -> trains NB/SVM + NB-SVM hybrid, saves nb_svm_metrics.json
#
# Everything is saved as PNG (plus two small CSVs) into <project_root>/output/.
# Console output also prints the exact numbers to paste into the report tables/text
# wherever the document has a "[PLACEHOLDER]" marker.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[0]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

RAW_DIR = root_dir / "data" / "raw"
PROCESSED_DIR = root_dir / "data" / "processed"
SAVE_DIR = root_dir / "src" / "saved_models"
OUT_DIR = root_dir / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 150

# Final combined model per family -- these are the rows that belong in Table 4.1 /
# Figure 4.1 (baseline components like standalone "Naive Bayes" or "SVM" are excluded
# from that headline comparison, but still get their own confusion matrix figure).
FINAL_MODEL_KEYS = ["NB-SVM Hybrid", "LSTM + CNN", "NB + LR + RF Hybrid"]

# Edit this list to whichever spam-indicative / ham-indicative phrases you want to
# feature in Figure 3.2.6.3 -- these are the ones referenced in the current report text.
VOCAB_PHRASES = [
    "click here", "guarantee", "limited time", "free", "winner",
    "conference call", "schedule", "meeting", "invoice", "urgent",
]


def word_count(text):
    return len(str(text).split())


def load_raw(filename):
    df = pd.read_csv(RAW_DIR / filename)
    return df.drop(columns="Unnamed: 0", errors="ignore")


# ---------------------------------------------------------------------------
# Chapter 3 -- dataset description & preliminary analysis
# ---------------------------------------------------------------------------

def fig_class_distribution():
    """Figure/Table for Section 3.2.2 -- spam vs ham counts in the final merged,
    cleaned dataset (data/processed/spam.csv)."""
    df = pd.read_csv(PROCESSED_DIR / "spam.csv")
    counts = df["Category"].value_counts().sort_index()
    labels = {0: "Ham", 1: "Spam"}
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([labels[i] for i in counts.index], counts.values,
                  color=["#2e7d32", "#e65100"])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,}\n({h / total:.1%})", (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom")
    ax.set_ylabel("Number of messages")
    ax.set_title("Class Distribution (CEAS_08 + Enron + Nazario, merged)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_2_2_class_distribution.png")
    plt.close(fig)

    print("\n[Section 3.2.2 / 3.2.3 -- paste into the dataset description table]")
    print(f"  Total records:  {total:,}")
    for i in (1, 0):
        c = counts.get(i, 0)
        print(f"  {labels[i]:5s}: {c:,} ({c / total:.2%})")


def fig_length_stats():
    """Numbers for Section 3.2.3 -- average/median word count per class."""
    df = pd.read_csv(PROCESSED_DIR / "spam.csv")
    df["word_count"] = df["Message"].apply(word_count)
    stats = df.groupby("Category")["word_count"].agg(["mean", "median"]).rename(
        index={0: "Ham", 1: "Spam"})

    print("\n[Section 3.2.3 -- message length by class, paste into the paragraph]")
    print(stats.round(1))


def fig_spam_rate_by_length():
    """Figure 3.2.6.1 -- spam rate binned by message length, on the merged dataset."""
    df = pd.read_csv(PROCESSED_DIR / "spam.csv")
    df["word_count"] = df["Message"].apply(word_count)

    bins = [0, 5, 15, 50, 100, 250, 500, np.inf]
    bin_labels = ["<5", "5-15", "15-50", "50-100", "100-250", "250-500", "500+"]
    df["length_bin"] = pd.cut(df["word_count"], bins=bins, labels=bin_labels)
    rate = df.groupby("length_bin", observed=True)["Category"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rate.index.astype(str), rate.values, color="#c62828")
    ax.set_xlabel("Message length (words)")
    ax.set_ylabel("Spam rate")
    ax.set_ylim(0, 1)
    ax.set_title("Spam Rate by Message Length (merged dataset)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_2_6_1_spam_rate_by_length.png")
    plt.close(fig)


def fig_merge_effect():
    """Figure 3.2.6.2 -- spam rate among short messages, CEAS_08 alone vs the full
    merged (CEAS_08 + Enron + Nazario) dataset."""
    ceas = load_raw("CEAS_08.csv")[["subject", "body", "label"]].dropna(subset=["label"]).copy()
    ceas["subject"] = ceas["subject"].fillna("")
    ceas["body"] = ceas["body"].fillna("")
    ceas["word_count"] = (ceas["subject"] + " " + ceas["body"]).apply(word_count)
    ceas_short = ceas[ceas["word_count"] <= 15]
    ceas_rate = ceas_short["label"].mean() if len(ceas_short) else float("nan")

    merged = pd.read_csv(PROCESSED_DIR / "spam.csv")
    merged["word_count"] = merged["Message"].apply(word_count)
    merged_short = merged[merged["word_count"] <= 15]
    merged_rate = merged_short["Category"].mean() if len(merged_short) else float("nan")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(["CEAS_08 only", "+ Enron + Nazario"], [ceas_rate, merged_rate],
           color=["#455a64", "#00838f"])
    ax.set_ylabel("Spam rate among short messages (<=15 words)")
    ax.set_ylim(0, 1)
    ax.set_title("Effect of Merging Additional Corpora")
    for i, v in enumerate([ceas_rate, merged_rate]):
        ax.annotate(f"{v:.1%}", (i, v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_2_6_2_merge_effect.png")
    plt.close(fig)

    print("\n[Section 3.2.6 -- short-message (<=15 words) spam rate, paste into the paragraph]")
    print(f"  CEAS_08 alone:        {ceas_rate:.1%}")
    print(f"  CEAS_08+Enron+Nazario: {merged_rate:.1%}")


def fig_vocab_association():
    """Figure 3.2.6.3 -- spam rate among messages containing each phrase, vs the
    dataset's overall spam rate. Edit VOCAB_PHRASES above to change the word list."""
    df = pd.read_csv(PROCESSED_DIR / "spam.csv")
    overall_rate = df["Category"].mean()
    msg = df["Message"].astype(str)

    rows = []
    for phrase in VOCAB_PHRASES:
        mask = msg.str.contains(phrase, regex=False)
        if mask.sum() == 0:
            continue
        rows.append((phrase, df.loc[mask, "Category"].mean(), int(mask.sum())))

    assoc = pd.DataFrame(rows, columns=["phrase", "spam_rate", "count"]).sort_values("spam_rate")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#c62828" if v > overall_rate else "#1565c0" for v in assoc["spam_rate"]]
    ax.barh(assoc["phrase"], assoc["spam_rate"], color=colors)
    ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1,
               label=f"Overall spam rate ({overall_rate:.1%})")
    ax.set_xlabel("Spam rate among messages containing phrase")
    ax.set_xlim(0, 1)
    ax.set_title("Spam Association of Selected Phrases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_2_6_3_vocab_association.png")
    plt.close(fig)

    print("\n[Section 3.2.6 -- vocabulary association, paste into the paragraph]")
    print(assoc.to_string(index=False))


# ---------------------------------------------------------------------------
# Chapter 4 -- model comparison
# ---------------------------------------------------------------------------

def load_all_metrics():
    """Merge every model's metrics from the three saved *_metrics.json files."""
    entries = {}
    for path in ["lstm_cnn_metrics.json", "nb_lr_rf_metrics.json", "nb_svm_metrics.json"]:
        p = SAVE_DIR / path
        if p.exists():
            with open(p) as f:
                entries.update(json.load(f))
        else:
            print(f"  (missing {p} -- run the matching training script first)")
    return entries


def fig_model_comparison():
    """Table 4.1 + Figure 4.1 -- headline comparison of the three final models."""
    entries = load_all_metrics()
    final = {k: v for k, v in entries.items() if k in FINAL_MODEL_KEYS}
    if not final:
        print("\n[Section 4.1] No metrics found yet -- train the models first.")
        return

    df = pd.DataFrame(final).T[["accuracy", "precision", "recall", "f1"]]
    df = df.reindex([k for k in FINAL_MODEL_KEYS if k in df.index])

    print("\n[Table 4.1 -- paste these into the results table]")
    print((df * 100).round(2))
    df.to_csv(OUT_DIR / "table_4_1_model_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.2
    for i, metric in enumerate(df.columns):
        ax.bar(x + i * width, df[metric], width, label=metric.capitalize())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of Model Performance on the Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_1_model_comparison.png")
    plt.close(fig)


def fig_confusion_matrices():
    """One confusion-matrix heatmap per model that has one saved (baselines included)."""
    entries = load_all_metrics()
    for name, m in entries.items():
        cm = m.get("confusion_matrix")
        if cm is None:
            continue
        cm = np.array(cm)

        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        im = ax.imshow(cm, cmap="Greens")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Ham", "Spam"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Ham", "Spam"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        fig.savefig(OUT_DIR / f"fig_confusion_matrix_{safe_name}.png")
        plt.close(fig)


if __name__ == "__main__":
    print("Generating Chapter 3 dataset figures...")
    fig_class_distribution()
    fig_length_stats()
    fig_spam_rate_by_length()
    fig_merge_effect()
    fig_vocab_association()

    print("\nGenerating Chapter 4 results figures...")
    fig_model_comparison()
    fig_confusion_matrices()

    print(f"\nAll figures/tables saved to: {OUT_DIR}")