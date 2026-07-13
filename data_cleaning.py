"""Core data cleaning functions for the spam dataset."""
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

    yield "lowercase", f"Lowercasing '{text_column}' column...", None
    df = lowercase_column(df, text_column)
    yield "lowercase_done", "Text normalized to lowercase.", df.head()

    yield "done", "Cleaning complete!", df