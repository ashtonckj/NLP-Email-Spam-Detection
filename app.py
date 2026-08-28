import json
import pickle
import random
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np
import scipy.sparse as sp
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocessing import (
    FEATURES,
    TARGET,
    basic_clean,
    clean_text_heavy,
    clean_text_light,
    combine_text,
)

SAVED_MODELS_DIR = ROOT_DIR / "src" / "saved_models"

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_FILES = ["CEAS_08.csv", "Enron.csv", "Nazario.csv"]

# Hardcoded registry of trained models -- add/remove an entry here whenever a
# model is added, renamed, or retired. "kind" controls which _predict_* method
# is used to score a given model type:
#   - "keras"          -> lstm_cnn.py:  .keras model + its own tokenizer.pkl
#   - "sklearn_proba"  -> nb_lr_rf.py:  joblib bundle, classifier exposes predict_proba
#   - "nb_svm"         -> nb_svm.py:    joblib bundle, LinearSVC (no predict_proba),
#                          needs the NB log-count-ratio reweighting before scoring
MODEL_REGISTRY = [
    {
        "name": "LSTM + CNN Hybrid",
        "kind": "keras",
        "model_file": "lstm_cnn_model.keras",
        "tokenizer_file": "lstm_cnn_tokenizer.pkl",
        "metrics_file": "lstm_cnn_metrics.json",
    },
    {
        "name": "NB + LR + RF Hybrid",
        "kind": "sklearn_proba",
        "model_file": "nb_lr_rf_model.joblib",
        "metrics_file": "nb_lr_rf_metrics.json",
        "bundle_model_key": "hybrid",
        "bundle_vectorizer_key": "vectorizer",
    },
    {
        "name": "NB + SVM Hybrid",
        "kind": "nb_svm",
        "model_file": "nb_svm_model.joblib",
        "metrics_file": "nb_svm_metrics.json",
    },
]

HIGHLIGHT_MODEL_FILES = ("nb_svm_model.joblib", "nb_lr_rf_model.joblib")  # linear models pooled for per-word spam scoring

# Palette
BG      = "#1e2229"
CARD    = "#272c35"
FG      = "#e6e9ef"
MUTED   = "#8b93a7"
ACCENT  = "#5b8def"
OK      = "#4cc38a"
ERR     = "#e5645e"

STEP_DELAY_MS = 800

class SpamMe(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SpamMe Application")
        self.geometry("900x700")
        self.minsize(800, 600)
        self.configure(bg=BG)

        # Application State
        self.cleaned_df = None
        self.pipeline = None
        self.training_result = None
        self.trained_model = None  # Placeholder for loaded model in testing

        # Lazily-loaded NLTK assets (stopwords / stemmer), shared by every
        # TF-IDF based model (nb_lr_rf, nb_svm). Loaded on first Analyze click
        # so the app still opens instantly even without the nltk data cached.
        self._stop_words = None
        self._stemmer = None

        # Lazily-loaded pool of real test-set emails (see _get_test_samples),
        # used by the "Randomise Email" button. Loaded on first click so the
        # app still opens instantly.
        self._test_samples = None
        self._raw_split = None

        # Lazily-loaded list of linear models (NB-SVM, NB, LR), reused purely
        # to score per-word spam contribution for whatever email is currently
        # in the box (see _get_highlight_bundle / _get_word_contributions)
        self._highlight_bundle = None

        self._setup_style()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_testing = ttk.Frame(self.notebook, style="Bg.TFrame")
        self.notebook.add(self.tab_testing, text="Model Testing")
        self._build_testing_tab()

    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Card.TFrame", background=CARD, relief="flat")
        s.configure("Bg.TFrame", background=BG)

        # Notebook (Tabs) styling
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=CARD, foreground=MUTED, padding=(15, 8), font=("Segoe UI Semibold", 10))
        s.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])

        s.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI Semibold", 18))
        s.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        s.configure("Warning.TLabel", background=CARD, foreground="#f97979", font=("Segoe UI", 10))
        s.configure("Card.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI Semibold", 9))
        s.configure("KeywordValue.TLabel", background=CARD, foreground=FG, font=("Segoe UI", 10), wraplength=760)

        s.configure("TEntry", fieldbackground="#1a1e24", foreground=FG, bordercolor="#3a4150", insertcolor=FG, padding=6)
        s.configure("TCombobox", fieldbackground="#1a1e24", background="#1a1e24", foreground=FG, arrowcolor=MUTED, padding=5)
        s.map("TCombobox", fieldbackground=[("readonly", "#1a1e24")], foreground=[("readonly", FG)])

        s.configure("Randomise.TButton", background="#9C8F07", foreground=FG, borderwidth=0, padding=(14, 7))
        s.map("Randomise.TButton", background=[("active", "#83710A")])
        s.configure("Ghost.TButton", background="#343b47", foreground=FG, borderwidth=0, padding=(14, 7))
        s.map("Ghost.TButton", background=[("active", "#3f4757")])

        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, font=("Segoe UI Semibold", 10), padding=(22, 9))
        s.map("Accent.TButton", background=[("active", "#6f9bf2"), ("disabled", "#3a4150")], foreground=[("disabled", MUTED)])

        s.configure("Bar.Horizontal.TProgressbar", background=ACCENT, troughcolor=CARD, borderwidth=0, thickness=3)

        s.configure("Treeview", background="#1a1e24", foreground=FG, rowheight=25, fieldbackground="#1a1e24", borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])
        s.configure("Treeview.Heading", background=CARD, foreground=MUTED, font=("Segoe UI Semibold", 9), borderwidth=0)


    def _build_testing_tab(self):
        head = ttk.Frame(self.tab_testing, style="Bg.TFrame", padding=(20, 18, 20, 14))
        head.pack(fill="x")
        ttk.Label(head, text="Test Email Content", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Compare predictions across all your trained models.", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        # Input Area
        input_frame = ttk.Frame(self.tab_testing, style="Card.TFrame", padding=(16, 12))
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        ttk.Label(input_frame, text="EMAIL CONTENT", style="Card.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(input_frame, text="⚠ Messages less than 20 words may cause inaccurate results!", style="Warning.TLabel").place(relx=1.0, y=0, anchor="ne")

        self.test_input = tk.Text(
            input_frame,
            height=8,
            bd=0,
            bg="#1a1e24",
            fg=FG,
            insertbackground=ACCENT,   # cursor color -> now visible
            insertwidth=2,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            font=("Segoe UI", 11),
            padx=10,
            pady=10,
            wrap="word",
        )
        self.test_input.pack(fill="both", expand=True, pady=(0, 10))
        self.test_input.bind("<Control-a>", self._select_all_text)
        self.test_input.bind("<Control-A>", self._select_all_text)
        # Tk's Text widget uses Emacs-style bindings by default on every platorm --
        # that's *why* Ctrl-A needed to be overridden above (it's normally bound to
        # "move to line start"). Word-wise deletion, however, isn't bound to anything
        # by default, so Ctrl+Backspace / Ctrl+Delete need to be added explicitly too.
        self.test_input.bind("<Control-BackSpace>", self._delete_word_before)
        self.test_input.bind("<Control-Delete>", self._delete_word_after)

        # Tag used to visually highlight detected spam-trigger words/phrases
        self.test_input.tag_configure(
            "kw_highlight", background="#f5c451", foreground="#1a1e24", font=("Segoe UI", 11, "bold")
        )

        btn_row = ttk.Frame(input_frame, style="Card.TFrame")
        btn_row.pack(fill="x")

        self.randomise_btn = ttk.Button(
            btn_row, text="🎲 Randomise Email", style="Randomise.TButton", command=self._randomise_email
        )
        self.randomise_btn.pack(side="left")

        self.test_btn = ttk.Button(btn_row, text="Analyze Email", style="Accent.TButton", command=self._run_test)
        self.test_btn.pack(side="right")

        # Result Area (Multi-Model Table)
        self.result_frame = ttk.Frame(self.tab_testing, style="Card.TFrame", padding=(16, 12))
        self.result_frame.pack(fill="x", padx=20, pady=(0, 20))
        ttk.Label(self.result_frame, text="MODEL PREDICTIONS", style="Card.TLabel").pack(anchor="w", pady=(0, 10))

        # Table for results — height bumped to 6 so all current model types
        # (LSTM+CNN, NB+LR+RF hybrid, NB+SVM hybrid, ...) fit without scrolling
        self.result_tree = ttk.Treeview(self.result_frame, columns=("Model", "Accuracy", "Confidence", "Prediction"), show="headings", height=6)
        self.result_tree.pack(fill="x")

        self.result_tree.heading("Model", text="Trained Model")
        self.result_tree.heading("Accuracy", text="Overall Accuracy")
        self.result_tree.heading("Confidence", text="Confidence Score")
        self.result_tree.heading("Prediction", text="Prediction")

        self.result_tree.column("Model", width=160, anchor="center")
        self.result_tree.column("Accuracy", width=120, anchor="center")
        self.result_tree.column("Confidence", width=120, anchor="center")
        self.result_tree.column("Prediction", width=150, anchor="center")

        # Flagged spam-trigger words (heuristic, independent of the ML models)
        ttk.Label(self.result_frame, text="SPAM TRIGGER WORDS", style="Card.TLabel").pack(anchor="w", pady=(14, 5))
        self.keyword_label = ttk.Label(
            self.result_frame,
            text="Click \"Analyze Email\" to scan for spam trigger words.",
            style="KeywordValue.TLabel",
        )
        self.keyword_label.pack(anchor="w")

    def _load_raw_split(self):
        if self._raw_split is not None:
            return self._raw_split

        missing = [f for f in RAW_DATA_FILES if not (RAW_DATA_DIR / f).exists()]
        if missing:
            self._raw_split = ()
            return self._raw_split

        frames = []
        for filename in RAW_DATA_FILES:
            df = pd.read_csv(RAW_DATA_DIR / filename)
            df = df.drop(columns="Unnamed: 0", errors="ignore")
            df = df[FEATURES + TARGET]
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)

        combined = basic_clean(combined)          # same dedup/NaN handling as training
        combined = combine_text(combined)         # raw Message + Category (no clean_text_light)

        X = combined[["subject", "body", "Message"]]
        y = combined["Category"]
        self._raw_split = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
        )
        return self._raw_split

    def _get_test_samples(self):
        if self._test_samples is not None:
            return self._test_samples

        split = self._load_raw_split()
        if not split:
            self._test_samples = []
            return self._test_samples
        _, X_test, _, y_test = split

        test_df = X_test.copy()
        test_df["Category"] = y_test

        # Skip near-empty or very long emails so the demo text stays readable
        word_count = test_df["Message"].str.split().str.len()
        test_df = test_df[(word_count >= 8) & (word_count <= 80)]

        # Take a fixed-size, evenly mixed sample so both spam and ham show up,
        # with a fixed seed so the *pool* is reproducible (the actual pick in
        # _randomise_email below is still random each click).
        spam_pool = test_df.loc[test_df["Category"] == 1, "Message"]
        ham_pool = test_df.loc[test_df["Category"] == 0, "Message"]

        spam_sample = spam_pool.sample(min(len(spam_pool), 25), random_state=42)
        ham_sample = ham_pool.sample(min(len(ham_pool), 25), random_state=42)

        self._test_samples = list(spam_sample) + list(ham_sample)
        return self._test_samples

    def _get_highlight_bundle(self):
        """Lazily load every linear model we have, purely to score per-word
        spam contribution for whatever email is in the box. Nothing new is
        exported to disk -- these are the same joblib bundles already used
        for real predictions elsewhere in the app:

          - NB-SVM's LinearSVC, over (TF-IDF x NB log-count-ratio) features
          - the NB+LR+RF bundle's NB and LR sub-models, pulled straight out
            of the saved VotingClassifier's own fitted clones (no retraining)

        Random Forest is skipped -- it has no per-word linear weight to read.
        """
        if self._highlight_bundle is not None:
            return self._highlight_bundle

        sources = []

        nb_svm_path = SAVED_MODELS_DIR / "nb_svm_model.joblib"
        if nb_svm_path.exists():
            bundle = joblib.load(nb_svm_path)
            sources.append(("svm", bundle["vectorizer"], bundle["nb_log_count_ratio"], bundle["svm"]))

        nb_lr_rf_path = SAVED_MODELS_DIR / "nb_lr_rf_model.joblib"
        if nb_lr_rf_path.exists():
            bundle = joblib.load(nb_lr_rf_path)
            hybrid, vectorizer = bundle["hybrid"], bundle["vectorizer"]
            nb = hybrid.named_estimators_.get("nb")
            lr = hybrid.named_estimators_.get("lr")
            if nb is not None:
                sources.append(("nb", vectorizer, None, nb))
            if lr is not None:
                sources.append(("lr", vectorizer, None, lr))

        self._highlight_bundle = sources
        return self._highlight_bundle

    def _get_word_contributions(self, heavy_clean_text):
        """Return {word: contribution} pooled across every available linear
        model, for THIS email only. Each model's own term is exact math off
        its real learned weights -- summing them just means a word that
        several models independently lean on adds up, so a spam email with
        multiple recognizable signals highlights more of them, while a
        message none of the models find spammy nets out near/below zero.
        """
        sources = self._get_highlight_bundle()
        if not sources:
            return None

        combined = {}
        for kind, vectorizer, r, model in sources:
            vec = sp.csr_matrix(vectorizer.transform([heavy_clean_text]))
            feature_names = vectorizer.get_feature_names_out()

            if kind == "svm":
                reweighted = vec.multiply(r)  # same NB log-count-ratio reweighting used at training time
                contributions = np.asarray(reweighted.todense()).ravel() * model.coef_.ravel()
            elif kind == "lr":
                contributions = np.asarray(vec.multiply(model.coef_.ravel()).todense()).ravel()
            elif kind == "nb":
                # log P(word | spam) - log P(word | ham), scaled by this email's TF-IDF weight
                log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                contributions = np.asarray(vec.multiply(log_ratio).todense()).ravel()
            else:
                continue

            for i in vec.nonzero()[1]:
                word = feature_names[i]
                combined[word] = combined.get(word, 0.0) + contributions[i]

        return combined

    @staticmethod
    def _word_to_feature(word, stemmer, stop_words):
        """Map a raw word from the textbox to the stemmed form the TF-IDF
        vocabulary uses -- the same rule clean_text_heavy applies (drop
        words length <=2, drop stopwords, then stem), just one word at a time
        so a highlighted span in the textbox can be traced back to a feature.
        """
        w = word.lower()
        if len(w) <= 2 or w in stop_words:
            return None
        return stemmer.stem(w)

    def _randomise_email(self):
        samples = self._get_test_samples()
        if not samples:
            missing_list = "\n".join(f"  - {f}" for f in RAW_DATA_FILES)
            messagebox.showwarning(
                "Raw Test Data Not Found",
                f"Couldn't find the raw dataset CSVs in:\n{RAW_DATA_DIR}\n\n"
                f"Expected files:\n{missing_list}",
            )
            return

        sample = random.choice(samples)
        self.test_input.delete("1.0", "end")
        self.test_input.insert("1.0", sample)
        self.test_input.tag_remove("kw_highlight", "1.0", "end")

        self.result_tree.delete(*self.result_tree.get_children())
        self.keyword_label.configure(text="Click \"Analyze Email\" to scan for spam trigger words.")

    def _highlight_keywords(self, email_text):
        widget = self.test_input
        widget.tag_remove("kw_highlight", "1.0", "end")

        stop_words, stemmer = self._get_nltk_assets()
        light_clean = clean_text_light(email_text)
        heavy_clean = clean_text_heavy(light_clean, stemmer, stop_words)

        word_scores = self._get_word_contributions(heavy_clean)
        if word_scores is None:
            self.keyword_label.configure(
                text="No trained classical models found -- can't compute trigger words."
            )
            return

        found = []
        for match in re.finditer(r"[a-zA-Z]+", email_text):
            token = match.group()
            feature = self._word_to_feature(token, stemmer, stop_words)
            score = word_scores.get(feature) if feature else None
            if score is not None and score > 0:
                widget.tag_add("kw_highlight", f"1.0+{match.start()}c", f"1.0+{match.end()}c")
                found.append((token, score))

        if found:
            # Strongest contributors to THIS email's spam score first;
            # de-dupe by word while keeping first-seen casing.
            found.sort(key=lambda pair: pair[1], reverse=True)
            seen = set()
            ordered = []
            for token, _ in found:
                key = token.lower()
                if key not in seen:
                    seen.add(key)
                    ordered.append(token)
            self.keyword_label.configure(text="⚠ " + ", ".join(ordered))
        else:
            self.keyword_label.configure(text="No words in this email pushed it toward spam.")

    @staticmethod
    def _select_all_text(event):
        widget = event.widget
        widget.tag_add("sel", "1.0", "end-1c")
        widget.mark_set("insert", "end-1c")
        widget.see("insert")
        return "break"  # stop Tk's default "move to line start" binding from also running

    @staticmethod
    def _delete_word_before(event):
        widget = event.widget
        widget.delete("insert -1c wordstart", "insert")
        return "break"

    @staticmethod
    def _delete_word_after(event):
        widget = event.widget
        widget.delete("insert", "insert wordend")
        return "break"

    def _get_nltk_assets(self):
        """Lazily load stopwords + stemmer, needed by the TF-IDF (nb_lr_rf, nb_svm) models."""
        if self._stop_words is None or self._stemmer is None:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer

            try:
                self._stop_words = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords")
                self._stop_words = set(stopwords.words("english"))
            self._stemmer = PorterStemmer()
        return self._stop_words, self._stemmer

    def _run_test(self):
        email_text = self.test_input.get("1.0", "end-1c").strip()
        if not email_text:
            messagebox.showwarning("Empty Input", "Please paste some email text to analyze.")
            return

        self.test_btn.configure(state="disabled")

        self._highlight_keywords(email_text)

        # Clear previous results
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_tree.insert("", "end", values=("Scanning models...", "-", "-", "-"))

        # Small delay just lets Tk repaint the "Scanning..." row before the
        # (potentially slow) model loading below blocks the main thread.
        self.after(100, lambda: self._display_test_result(email_text))

    def _display_test_result(self, email_text):
        self.result_tree.delete(*self.result_tree.get_children())

        if not SAVED_MODELS_DIR.is_dir():
            self.result_tree.insert("", "end", values=("No saved_models folder found", "-", "-", "-"))
            self.test_btn.configure(state="normal")
            return

        # Same light clean used to build the training data (see preprocessing.py)
        light_clean = clean_text_light(email_text)
        heavy_clean = None  # only computed if a TF-IDF-based model is actually found

        any_model_found = False

        for entry in MODEL_REGISTRY:
            model_path = SAVED_MODELS_DIR / entry["model_file"]
            if not model_path.exists():
                continue
            any_model_found = True

            try:
                if entry["kind"] == "keras":
                    pred_score = self._predict_keras(entry, model_path, light_clean)
                else:
                    if heavy_clean is None:
                        stop_words, stemmer = self._get_nltk_assets()
                        heavy_clean = clean_text_heavy(light_clean, stemmer, stop_words)

                    if entry["kind"] == "sklearn_proba":
                        pred_score = self._predict_sklearn_proba(entry, model_path, heavy_clean)
                    elif entry["kind"] == "nb_svm":
                        pred_score = self._predict_nb_svm(model_path, heavy_clean)
                    else:
                        raise ValueError(f"Unknown model kind: {entry['kind']!r}")

                self._insert_prediction(entry, pred_score)
            except Exception as exc:
                self.result_tree.insert("", "end", values=(entry["name"], "N/A", "N/A", f"Error: {exc}"))

        if not any_model_found:
            self.result_tree.insert("", "end", values=("No trained models found", "-", "-", "-"))

        self.test_btn.configure(state="normal")

    @staticmethod
    def _predict_keras(entry, model_path, light_clean):
        tokenizer_path = SAVED_MODELS_DIR / entry["tokenizer_file"]
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"{entry['tokenizer_file']} missing")

        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)

        # NOTE: must be wrapped in a list -- TextVectorization expects a
        # batch of strings, not a single scalar string.
        seq = tokenizer(np.array([light_clean])).numpy()
        padded_text = pad_sequences(seq, maxlen=200, padding="post")

        model = load_model(model_path)
        return float(model.predict(padded_text, verbose=0)[0][0])

    @staticmethod
    def _predict_sklearn_proba(entry, model_path, heavy_clean):
        bundle = joblib.load(model_path)
        clf = bundle[entry["bundle_model_key"]]
        vectorizer = bundle[entry["bundle_vectorizer_key"]]

        vec = vectorizer.transform([heavy_clean])
        classes = list(clf.classes_)
        spam_idx = classes.index(1) if 1 in classes else 1
        return float(clf.predict_proba(vec)[0][spam_idx])

    @staticmethod
    def _predict_nb_svm(model_path, heavy_clean):
        bundle = joblib.load(model_path)
        vectorizer = bundle["vectorizer"]
        r = bundle["nb_log_count_ratio"]
        svm = bundle["svm"]

        vec = vectorizer.transform([heavy_clean])
        r_sparse = sp.diags(r)
        vec_nb = sp.csr_matrix(vec.dot(r_sparse))

        # LinearSVC has no predict_proba -- squash the raw decision margin
        # through a sigmoid so it can be shown as a 0-1 confidence score.
        # This is NOT a calibrated probability, just a monotonic stand-in
        # for display purposes (same >0.5 = spam threshold as the others).
        margin = svm.decision_function(vec_nb)[0]
        return float(1.0 / (1.0 + np.exp(-margin)))

    def _insert_prediction(self, entry, pred_score):
        if pred_score > 0.5:
            prediction_label = "🚨 SPAM"
            confidence = pred_score * 100
        else:
            prediction_label = "✅ HAM"
            confidence = (1.0 - pred_score) * 100

        display_accuracy = self._get_model_accuracy(entry)

        self.result_tree.insert(
            "", "end",
            values=(entry["name"], display_accuracy, f"{confidence:.2f}%", prediction_label),
        )

    @staticmethod
    def _get_model_accuracy(entry):
        metrics_path = SAVED_MODELS_DIR / entry["metrics_file"]
        if not metrics_path.exists():
            return "N/A"

        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        # Every metrics file now has the same shape (see evaluation.export_model_metrics):
        # {"Some Model Name": {"accuracy": ..., "precision": ..., ...}, ...}
        # Pull whichever entry is the final hybrid/ensemble result rather than
        # an intermediate sub-model (e.g. skip "Naive Bayes" inside nb_lr_rf_metrics.json).
        hybrid_entry = next((v for k, v in metrics_data.items() if "hybrid" in k.lower()), None)
        acc = hybrid_entry.get("accuracy", "N/A") if hybrid_entry else "N/A"

        if isinstance(acc, (int, float)):
            return f"{acc * 100:.2f}%"
        return acc


if __name__ == "__main__":
    app = SpamMe()
    app.mainloop()