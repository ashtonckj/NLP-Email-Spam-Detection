"""
SpamMe
======
A small Tkinter GUI for testing spam/ham classifier "hybrid" models on the
classic Category/Message spam dataset.

Setup
-----
    pip install pandas scikit-learn emoji

Run
---
    python spamme.py

The app expects a CSV file at:  data/spam.csv   (relative to this script)
with two columns: Category (spam/ham), Message (the email/sms text).

Currently only "Hybrid 1: SVM + LR (Stacking)" is implemented. Hybrid 2 and
Hybrid 3 are left as `None` placeholders in MODEL_SLOTS below -- plug your
own trained pipelines in there later and everything else (the metrics
table, the single-email tester, etc.) will pick them up automatically.
"""

import os
import random
import re
import string
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import emoji
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Chat-word / slang normalization dictionary
# (starter list -- extend this whenever you find more shorthand in your data)
# ---------------------------------------------------------------------------
CHAT_WORDS = {
    "u": "you", "ur": "your", "r": "are", "y": "why",
    "pls": "please", "plz": "please",
    "thx": "thanks", "tks": "thanks", "thnx": "thanks",
    "lol": "laughing out loud", "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing", "omg": "oh my god",
    "omw": "on my way", "btw": "by the way", "asap": "as soon as possible",
    "idk": "i do not know", "imo": "in my opinion", "imho": "in my honest opinion",
    "fyi": "for your information", "brb": "be right back", "b4": "before",
    "gr8": "great", "l8r": "later", "2day": "today", "2nite": "tonight",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "cuz": "because", "coz": "because", "bc": "because", "b/c": "because",
    "congrats": "congratulations", "msg": "message", "txt": "text",
    "urs": "yours", "yr": "your", "np": "no problem", "ppl": "people",
    "sry": "sorry", "tho": "though", "thru": "through",
    "w/": "with", "w/o": "without", "info": "information", "abt": "about",
    "bday": "birthday", "xoxo": "hugs and kisses", "ttyl": "talk to you later",
    "afaik": "as far as i know", "irl": "in real life", "smh": "shaking my head",
    # contracted forms (apostrophes are still present at this point in cleaning)
    "im": "i am", "i'm": "i am", "ive": "i have", "i've": "i have",
    "dont": "do not", "don't": "do not", "cant": "cannot", "can't": "cannot",
    "wont": "will not", "won't": "will not",
    "isnt": "is not", "isn't": "is not", "didnt": "did not", "didn't": "did not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "you're": "you are", "you'll": "you will", "let's": "let us",
}

# Words/slots for the hybrid models. Hybrid 1 is implemented below.
# Hybrid 2 / Hybrid 3 are placeholders for models you'll add later.
HYBRID_NAMES = [
    "Hybrid 1: SVM + LR (Stacking)",
    "Hybrid 2",
    "Hybrid 3",
]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
class TextPreprocessor:
    """Cleans raw email/message text before TF-IDF vectorization."""

    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<.*?>")
    NUMBER_PATTERN = re.compile(r"\d+")
    NON_ALNUM_PATTERN = re.compile(r"[^a-z\s]")
    MULTI_SPACE_PATTERN = re.compile(r"\s+")
    PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    def to_lowercase(self, text: str) -> str:
        return text.lower()

    def remove_html_and_links(self, text: str) -> str:
        text = self.HTML_PATTERN.sub(" ", text)
        text = self.URL_PATTERN.sub(" ", text)
        return text

    def handle_emojis(self, text: str) -> str:
        # Turn emoji glyphs into readable words, e.g. "😀" -> " grinning face "
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace("_", " ").replace(":", " ")
        return text

    def handle_chat_words(self, text: str) -> str:
        tokens = text.split()
        expanded = [CHAT_WORDS.get(tok, tok) for tok in tokens]
        return " ".join(expanded)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(self.PUNCT_TABLE)

    def remove_numbers(self, text: str) -> str:
        return self.NUMBER_PATTERN.sub(" ", text)

    def remove_non_alphanumeric(self, text: str) -> str:
        # Catches any leftover symbols/accents/special characters
        return self.NON_ALNUM_PATTERN.sub(" ", text)

    def remove_extra_spaces(self, text: str) -> str:
        return self.MULTI_SPACE_PATTERN.sub(" ", text).strip()

    def remove_stopwords(self, text: str) -> str:
        tokens = [tok for tok in text.split() if tok not in ENGLISH_STOP_WORDS]
        return " ".join(tokens)

    def clean(self, text) -> str:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = self.to_lowercase(text)
        text = self.remove_html_and_links(text)
        text = self.handle_emojis(text)
        text = self.handle_chat_words(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_non_alphanumeric(text)
        text = self.remove_extra_spaces(text)
        text = self.remove_stopwords(text)
        text = self.remove_extra_spaces(text)
        return text


# ---------------------------------------------------------------------------
# Model training (kept independent of Tkinter so it can be tested/reused)
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_hybrid1(X_train_tfidf, y_train, X_test_tfidf, y_test, seed: int):
    """
    Hybrid 1 = SVM + Logistic Regression combined via STACKING:
    both base models are trained, their (cross-validated) prediction
    probabilities become the input features for a Logistic Regression
    meta-model, which makes the final call.
    """
    svm = SVC(kernel="linear", probability=True, random_state=seed)
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    meta = LogisticRegression(max_iter=1000, random_state=seed)
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    stacked_model = StackingClassifier(
        estimators=[("svm", svm), ("lr", lr)],
        final_estimator=meta,
        cv=cv_splitter,
        stack_method="predict_proba",
    )
    stacked_model.fit(X_train_tfidf, y_train)
    y_pred = stacked_model.predict(X_test_tfidf)
    metrics = compute_metrics(y_test, y_pred)
    return stacked_model, metrics


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class SpamMeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SpamMe")
        self.geometry("880x800")
        self.minsize(760, 700)

        self.preprocessor = TextPreprocessor()
        self.df = None                 # full raw dataset (Category, Message)
        self.vectorizer = None
        self.models = {name: None for name in HYBRID_NAMES}
        self.metrics = {name: None for name in HYBRID_NAMES}
        self.test_df = None            # held-out 20% (original text + label), set after training
        self.current_email_text = None
        self.current_true_label = None

        self._load_dataset()
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def _load_dataset(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spam.csv")
        if not os.path.exists(path):
            messagebox.showerror(
                "Dataset not found",
                f"Could not find:\n{path}\n\nPlace your spam.csv file at data/spam.csv "
                f"(relative to this script) and restart the app.",
            )
            self.df = None
            return

        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception as e:
            messagebox.showerror("Error loading dataset", str(e))
            self.df = None
            return

        if "Category" not in df.columns or "Message" not in df.columns:
            messagebox.showerror(
                "Unexpected columns",
                f"Expected columns 'Category' and 'Message' but found: {list(df.columns)}",
            )
            self.df = None
            return

        df = df[["Category", "Message"]].dropna()
        df["Category"] = df["Category"].astype(str).str.strip().str.lower()
        df = df[df["Category"].isin(["spam", "ham"])].reset_index(drop=True)

        if df.empty:
            messagebox.showerror("Empty dataset", "No valid spam/ham rows found in the CSV.")
            self.df = None
            return

        self.df = df

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        title_label = tk.Label(self, text="SpamMe", font=("Segoe UI", 22, "bold"))
        title_label.pack(pady=(14, 4))

        subtitle = tk.Label(
            self, text="Train and test spam/ham hybrid classifiers", font=("Segoe UI", 10)
        )
        subtitle.pack(pady=(0, 10))

        # --- Seed + Run Models row ------------------------------------------------
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", **pad)

        tk.Label(top_frame, text="Random Seed:").pack(side="left")
        self.seed_var = tk.StringVar(value="42")
        seed_entry = tk.Entry(top_frame, textvariable=self.seed_var, width=10)
        seed_entry.pack(side="left", padx=(6, 6))

        randomize_btn = tk.Button(top_frame, text="🎲 Randomize Seed", command=self._randomize_seed)
        randomize_btn.pack(side="left", padx=(0, 20))

        self.run_btn = tk.Button(
            top_frame,
            text="▶  Run Models  (80/20 split + train)",
            font=("Segoe UI", 10, "bold"),
            command=self.run_models,
        )
        self.run_btn.pack(side="left")

        # --- Metrics table ---------------------------------------------------------
        tk.Label(self, text="Model Performance", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10, pady=(14, 2)
        )

        metrics_cols = ("Model", "Accuracy", "Precision", "Recall", "F1 Score")
        self.metrics_tree = ttk.Treeview(self, columns=metrics_cols, show="headings", height=4)
        for col in metrics_cols:
            self.metrics_tree.heading(col, text=col)
            width = 260 if col == "Model" else 110
            self.metrics_tree.column(col, width=width, anchor="center")
        self.metrics_tree.pack(fill="x", padx=10)
        self._refresh_metrics_table()

        # --- Email tester ------------------------------------------------------------
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=14, padx=10)
        tk.Label(self, text="Test a Single Email", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10
        )

        random_frame = tk.Frame(self)
        random_frame.pack(fill="x", padx=10, pady=(6, 4))
        tk.Button(
            random_frame, text="🎲 Random SPAM email (from test set)",
            command=lambda: self.load_random_email("spam"),
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            random_frame, text="🎲 Random HAM email (from test set)",
            command=lambda: self.load_random_email("ham"),
        ).pack(side="left")

        tk.Label(self, text="(Or just type/paste your own email text below)").pack(
            anchor="w", padx=10
        )
        self.email_text = scrolledtext.ScrolledText(self, height=6, wrap="word")
        self.email_text.pack(fill="x", padx=10, pady=(4, 4))
        # Any manual edit means we no longer know the "true" label for this text.
        self.email_text.bind("<<Modified>>", self._on_email_text_modified)

        self.check_btn = tk.Button(
            self, text="🔍  Check Email  (confidence + accuracy per model)",
            font=("Segoe UI", 10, "bold"), command=self.check_email,
        )
        self.check_btn.pack(pady=(4, 10))

        pred_cols = ("Model", "Prediction", "Confidence", "Result vs true label")
        self.pred_tree = ttk.Treeview(self, columns=pred_cols, show="headings", height=4)
        for col in pred_cols:
            self.pred_tree.heading(col, text=col)
            width = 260 if col == "Model" else 150
            self.pred_tree.column(col, width=width, anchor="center")
        self.pred_tree.pack(fill="x", padx=10)

        # --- Explainability ------------------------------------------------------------
        tk.Label(
            self, text="Why? Top influential words (from Hybrid 1's Logistic Regression)",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=10, pady=(14, 2))

        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.word_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, font=("Consolas", 10)
        )
        self.word_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.word_listbox.yview)

        # --- Status bar ------------------------------------------------------------------
        self.status_var = tk.StringVar(value="Ready." if self.df is not None else "No dataset loaded.")
        status_bar = tk.Label(
            self, textvariable=self.status_var, bd=1, relief="sunken", anchor="w"
        )
        status_bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _randomize_seed(self):
        self.seed_var.set(str(random.randint(0, 999_999)))

    def _on_email_text_modified(self, event):
        # Tkinter's <<Modified>> flag needs manual resetting each time
        self.email_text.edit_modified(False)
        current_text = self.email_text.get("1.0", tk.END).strip()
        if current_text != self.current_email_text:
            self.current_true_label = None  # user has typed something new; no ground truth

    def run_models(self):
        if self.df is None:
            messagebox.showerror("No dataset", "No dataset is loaded. Add data/spam.csv and restart.")
            return

        seed_text = self.seed_var.get().strip()
        try:
            seed = int(seed_text)
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be a whole number, e.g. 42.")
            return

        self.run_btn.config(state="disabled", text="Training... please wait")
        self.status_var.set("Cleaning text and training models -- this may take a little while...")
        self.update_idletasks()

        try:
            cleaned = self.df["Message"].apply(self.preprocessor.clean)
            labels = self.df["Category"]

            train_idx, test_idx = train_test_split(
                self.df.index, test_size=0.2, random_state=seed, stratify=labels
            )
            X_train_text = cleaned.loc[train_idx]
            X_test_text = cleaned.loc[test_idx]
            y_train = labels.loc[train_idx]
            y_test = labels.loc[test_idx]

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train_text)
            X_test_tfidf = vectorizer.transform(X_test_text)

            hybrid1_model, hybrid1_metrics = train_hybrid1(
                X_train_tfidf, y_train, X_test_tfidf, y_test, seed
            )

            self.vectorizer = vectorizer
            self.models["Hybrid 1: SVM + LR (Stacking)"] = hybrid1_model
            self.metrics["Hybrid 1: SVM + LR (Stacking)"] = hybrid1_metrics
            # Hybrid 2 / Hybrid 3 stay None until you add real models
            self.models["Hybrid 2"] = None
            self.metrics["Hybrid 2"] = None
            self.models["Hybrid 3"] = None
            self.metrics["Hybrid 3"] = None

            # Keep the held-out test rows (original, uncleaned text) for the
            # "random email" buttons below.
            self.test_df = self.df.loc[test_idx].reset_index(drop=True)

            self._refresh_metrics_table()
            self.status_var.set(f"Training complete (seed={seed}). Test set size: {len(test_idx)}.")
            messagebox.showinfo("Done", "Models trained successfully.")
        except Exception as e:
            messagebox.showerror("Training failed", str(e))
            self.status_var.set("Training failed. See error message.")
        finally:
            self.run_btn.config(state="normal", text="▶  Run Models  (80/20 split + train)")

    def load_random_email(self, category: str):
        if self.test_df is None or self.test_df.empty:
            messagebox.showwarning("No test set yet", "Run the models first (this builds the held-out test set).")
            return
        subset = self.test_df[self.test_df["Category"] == category]
        if subset.empty:
            messagebox.showwarning("None found", f"No '{category}' examples in the test set.")
            return
        row = subset.sample(n=1).iloc[0]

        self.email_text.delete("1.0", tk.END)
        self.email_text.insert(tk.END, row["Message"])
        self.email_text.edit_modified(False)
        self.current_email_text = row["Message"].strip()
        self.current_true_label = row["Category"]

    def check_email(self):
        if self.vectorizer is None or all(m is None for m in self.models.values()):
            messagebox.showwarning("Not trained yet", "Run the models first.")
            return

        raw_text = self.email_text.get("1.0", tk.END).strip()
        if not raw_text:
            messagebox.showwarning("Empty email", "Type or load an email first.")
            return

        # We only know the ground-truth label if this text is exactly what a
        # "random email" button loaded (i.e. the user hasn't edited it).
        true_label = self.current_true_label if raw_text == self.current_email_text else None

        cleaned = self.preprocessor.clean(raw_text)
        vec = self.vectorizer.transform([cleaned])

        for row in self.pred_tree.get_children():
            self.pred_tree.delete(row)

        for name in HYBRID_NAMES:
            model = self.models.get(name)
            if model is None:
                self.pred_tree.insert("", tk.END, values=(name, "—", "—", "—"))
                continue
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            classes = list(model.classes_)
            confidence = proba[classes.index(pred)]
            if true_label is not None:
                result = "✅ Correct" if pred == true_label else "❌ Incorrect"
            else:
                result = "N/A (no ground truth)"
            self.pred_tree.insert(
                "", tk.END, values=(name, pred, f"{confidence * 100:.2f}%", result)
            )

        self._update_explainability(vec)
        self.status_var.set(
            "Checked email against true label."
            if true_label is not None
            else "Checked email (typed/edited text -- no ground truth to compare against)."
        )

    # ------------------------------------------------------------------ #
    # Table / explainability rendering helpers
    # ------------------------------------------------------------------ #
    def _refresh_metrics_table(self):
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for name in HYBRID_NAMES:
            m = self.metrics.get(name)
            if m is None:
                self.metrics_tree.insert("", tk.END, values=(name, "—", "—", "—", "—"))
            else:
                self.metrics_tree.insert(
                    "",
                    tk.END,
                    values=(
                        name,
                        f"{m['Accuracy'] * 100:.2f}%",
                        f"{m['Precision'] * 100:.2f}%",
                        f"{m['Recall'] * 100:.2f}%",
                        f"{m['F1 Score'] * 100:.2f}%",
                    ),
                )

    def _update_explainability(self, vec):
        self.word_listbox.delete(0, tk.END)

        hybrid1 = self.models.get("Hybrid 1: SVM + LR (Stacking)")
        if hybrid1 is None:
            self.word_listbox.insert(tk.END, "(Train the models first to see word explanations.)")
            return

        lr_fitted = hybrid1.named_estimators_["lr"]
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = lr_fitted.coef_[0]
        classes = lr_fitted.classes_  # e.g. ['ham', 'spam'], alphabetical
        negative_class, positive_class = classes[0], classes[1]

        nonzero_indices = vec.nonzero()[1]
        if len(nonzero_indices) == 0:
            self.word_listbox.insert(tk.END, "(No recognizable words left after cleaning.)")
            return

        contributions = []
        for idx in nonzero_indices:
            weight = coefs[idx] * vec[0, idx]
            contributions.append((feature_names[idx], weight))
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)

        for word, weight in contributions[:15]:
            direction = positive_class if weight > 0 else negative_class
            self.word_listbox.insert(
                tk.END, f"{word:<20} -> leans '{direction}'   (weight {weight:+.3f})"
            )


if __name__ == "__main__":
    app = SpamMeApp()
    app.mainloop()