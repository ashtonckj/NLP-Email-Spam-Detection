"""
Generate word clouds per Category (e.g. spam/ham), for the two distinct
text representations used across your training scripts:

  1. light-clean text  -> what lstm_cnn.py trains on (Message column,
     already light-cleaned by preprocessing.clean_text_light)
  2. heavy-clean text   -> what nb_lr_rf.py and nb_svm.py both train on
     (Message column run through preprocessing.clean_text_heavy:
     stemmed, stopwords removed)

Images are saved to output/ instead of shown on screen, since plt.show()
blocks and doesn't produce a file you can keep or share.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed -- we're only saving files
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import clean_text_heavy

SPAM_CSV = ROOT_DIR / "data" / "processed" / "spam.csv"
OUTPUT_DIR = ROOT_DIR / "output"


def generate_and_save_wordclouds(df, text_col, output_dir, variant_label):
    """
    For each unique value in df['Category'], build a word cloud from the
    text in `text_col` and save it as a PNG in output_dir.

    Returns the list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for category in df["Category"].unique():
        filtered_df = df[df["Category"] == category]
        text = " ".join(filtered_df[text_col].astype(str))

        if not text.strip():
            print(f"[{variant_label}] Skipping category {category!r} -- no text found")
            continue

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Word Cloud for Category: {category} ({variant_label})")
        plt.axis("off")

        filename = f"{variant_label}_category_{category}_wordcloud.png"
        out_path = output_dir / filename
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()  # free the figure -- without this, figures pile up in memory across the loop

        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

    return saved_paths


def main():
    df = pd.read_csv(SPAM_CSV)
    df[["subject", "body", "Message"]] = df[["subject", "body", "Message"]].fillna("")

    # --- Variant 1: light-clean text -- same input the LSTM+CNN model sees ---
    generate_and_save_wordclouds(
        df,
        text_col="Message",
        output_dir=OUTPUT_DIR,
        variant_label="lstm_cnn_lightclean",
    )

    # --- Variant 2: heavy-clean text -- same input NB+LR+RF and NB+SVM see ---
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    df["Message_heavy"] = df["Message"].apply(lambda t: clean_text_heavy(t, stemmer, stop_words))

    generate_and_save_wordclouds(
        df,
        text_col="Message_heavy",
        output_dir=OUTPUT_DIR,
        variant_label="nb_lr_rf_svm_heavyclean",
    )


if __name__ == "__main__":
    main()