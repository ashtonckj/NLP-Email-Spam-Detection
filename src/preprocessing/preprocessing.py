"""Core data cleaning functions for the spam dataset."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["subject", "body"]
TARGET = ["label"]


# Dedup, fill NaNs, drop rows with no label -- shared by every dataset
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df.dropna(subset=TARGET).reset_index(drop=True)
    return df


# Light clean -- safe for BOTH tfidf and lstm branches, keeps sentence structure intact
def clean_text_light(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)     # urls
    text = re.sub(r"\S+@\S+", " EMAIL ", text)            # emails
    text = re.sub(r"<.*?>", " ", text)                    # html tags
    text = re.sub(r"\b\d+\b", " NUM ", text)              # numbers -> placeholder
    text = re.sub(r"[^a-z\s]", " ", text)                 # remaining punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Heavy clean -- extra step ON TOP of light clean, only used by the tfidf/classical branch
# requires: nltk.download("stopwords") once, and PorterStemmer/stopwords passed in
def clean_text_heavy(light_text: str, stemmer, stop_words) -> str:
    tokens = light_text.split()
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


# Combine subject + body into Message, label into Category
def combine_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Message"] = (df["subject"] + " " + df["body"]).str.strip()
    df["Category"] = df[TARGET]
    return df


def validate_labels(df: pd.DataFrame) -> None:
    print("\nLabel distribution")
    print(df["Category"].value_counts())


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    for col in FEATURES:
        df[col] = df[col].apply(clean_text_light)
    df = combine_text(df)
    df["Message"] = df["Message"].replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=["Message"])
    validate_labels(df)
    print(df.info())
    return df[["subject", "body", "Message", "Category"]]


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    df[["subject", "body", "Message"]] = df[["subject", "body", "Message"]].fillna("")
    X = df[["subject", "body", "Message"]]
    y = df["Category"]
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)