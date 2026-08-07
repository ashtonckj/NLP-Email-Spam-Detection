"""Core data cleaning functions for the spam dataset."""
import re

import pandas as pd


def load_data(filepath):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(filepath)


def count_duplicates(df):
    """Return the number of duplicate rows in the DataFrame."""
    return int(df.duplicated().sum())


def remove_duplicates(df):
    """Drop duplicate rows and return a new DataFrame."""
    return df.drop_duplicates().reset_index(drop=True)


def lowercase_column(df, column):
    """Convert all values in a text column to lowercase."""
    df[column] = df[column].str.lower()
    return df

def handle_missing_value(df):
    """Fill missing subject/body with empty strings."""
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df.dropna(subset=["label"])
    return df

def combine_text(df):
    """Combine subject and body into a single text column."""
    #df["text"] = df["subject"] + " " + df["body"]
    df = df.copy()
    df["Message"] = df["subject"] + " " + df["body"]
    df["Category"] = df["label"]
    return df

def clean_text(text):
    """
    Clean email text.
    """
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_column(df):
    """Apply text cleaning to the combined text column."""
    #df["text"] = df["text"].apply(clean_text)
    df = df.copy()
    df["Message"] = df["Message"].apply(clean_text)
    return df


def validate_labels(df):
    """Display class distribution."""
    # print("\nLabel Distribution")
    # print(df["label"].value_counts())
    print("\nLabel Distribution")
    print(df["Category"].value_counts())
    return df

def run_cleaning_pipeline(filepath, text_column="Message"):
    """
    Generator that runs each cleaning step in order and yields
    (step_id, message, preview_data) after every step, so a caller
    (like a GUI) can display progress live instead of waiting for
    everything to finish at once.
    """
    yield "load", f"Loading '{filepath}'...", None
    df = load_data(filepath)
    yield "load_done", f"Loaded {len(df)} rows, {len(df.columns)} columns.", df.head()

    yield "check_dupes", "Scanning for duplicate rows...", None
    dupes = count_duplicates(df)
    yield "check_dupes_done", f"Found {dupes} duplicate row(s).", None

    yield "drop_dupes", "Removing duplicate rows...", None
    df = remove_duplicates(df)
    yield "drop_dupes_done", f"{len(df)} rows remain.", None

    yield "missing", "Handling missing values...", None
    df = handle_missing_value(df)
    yield "missing_done", "Missing values handled.", None

    yield "combine", "Combining subject and body...", None
    df = combine_text(df)
    yield "combine_done", "Subject and body combined.", df[["Message", "Category"]].head()

    yield "clean", "Cleaning email text...", None
    df = clean_text_column(df)
    yield "clean_done", "Text cleaned.", df[["Message", "Category"]].head()

    yield "labels", "Checking class distribution...", None
    validate_labels(df)
    yield "labels_done", "Label validation completed.", None


    # yield "lowercase", f"Lowercasing '{text_column}' column...", None
    # df = lowercase_column(df, text_column)
    # yield "lowercase_done", "Text normalized to lowercase.", df.head()

    yield "done", "Cleaning complete!", df