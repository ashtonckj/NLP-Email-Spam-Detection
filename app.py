import json
import os
import pickle
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --- Make sure "src.preprocessing.preprocessing" is importable regardless of
#     where this script is launched from (assumes app.py lives at the project root,
#     same level as the "src" folder) ---
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import clean_text_heavy, clean_text_light

# Palette
BG      = "#1e2229"
CARD    = "#272c35"
FG      = "#e6e9ef"
MUTED   = "#8b93a7"
ACCENT  = "#5b8def"
OK      = "#4cc38a"
ERR     = "#e5645e"

DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "Ceas08_Enron.csv")
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

        self._setup_style()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_testing = ttk.Frame(self.notebook, style="Bg.TFrame")
        self.notebook.add(self.tab_testing, text="3. Model Testing")
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
        s.configure("Card.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI Semibold", 9))

        s.configure("TEntry", fieldbackground="#1a1e24", foreground=FG, bordercolor="#3a4150", insertcolor=FG, padding=6)
        s.configure("TCombobox", fieldbackground="#1a1e24", background="#1a1e24", foreground=FG, arrowcolor=MUTED, padding=5)
        s.map("TCombobox", fieldbackground=[("readonly", "#1a1e24")], foreground=[("readonly", FG)])

        s.configure("Ghost.TButton", background="#343b47", foreground=FG, borderwidth=0, padding=(14, 7))
        s.map("Ghost.TButton", background=[("active", "#3f4757")])

        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, font=("Segoe UI Semibold", 10), padding=(22, 9))
        s.map("Accent.TButton", background=[("active", "#6f9bf2"), ("disabled", "#3a4150")], foreground=[("disabled", MUTED)])

        s.configure("Bar.Horizontal.TProgressbar", background=ACCENT, troughcolor=CARD, borderwidth=0, thickness=3)

        s.configure("Treeview", background="#1a1e24", foreground=FG, rowheight=25, fieldbackground="#1a1e24", borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])
        s.configure("Treeview.Heading", background=CARD, foreground=MUTED, font=("Segoe UI Semibold", 9), borderwidth=0)

    # ==========================================
    # TAB 3: MODEL TESTING (PREDICTION)
    # ==========================================
    def _build_testing_tab(self):
        head = ttk.Frame(self.tab_testing, style="Bg.TFrame", padding=(20, 18, 20, 14))
        head.pack(fill="x")
        ttk.Label(head, text="Test Email Content", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Compare predictions across all your trained models.", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        # Input Area
        input_frame = ttk.Frame(self.tab_testing, style="Card.TFrame", padding=(16, 12))
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)

        ttk.Label(input_frame, text="EMAIL CONTENT", style="Card.TLabel").pack(anchor="w", pady=(0, 5))

        # --- FIX 1: cursor was invisible because insertbackground was never set,
        #     so Tk defaulted to a black caret on a near-black background. ---
        # --- FIX 2: selection colors added so Ctrl+A / click-drag selection is
        #     actually visible against the dark theme. ---
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

        # --- FIX 3: Tk's Text widget binds Ctrl+A to "move cursor to start of
        #     line" (an emacs-style default), not "select all". That's why
        #     nothing was actually selected, and Delete/Backspace had nothing
        #     to remove. Rebind it to a real select-all and swallow the event
        #     (return "break") so the old behavior doesn't also fire. ---
        self.test_input.bind("<Control-a>", self._select_all_text)
        self.test_input.bind("<Control-A>", self._select_all_text)

        self.test_btn = ttk.Button(input_frame, text="Analyze Email", style="Accent.TButton", command=self._run_test)
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

    @staticmethod
    def _select_all_text(event):
        widget = event.widget
        widget.tag_add("sel", "1.0", "end-1c")
        widget.mark_set("insert", "end-1c")
        widget.see("insert")
        return "break"  # stop Tk's default "move to line start" binding from also running

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

        # Clear previous results
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_tree.insert("", "end", values=("Scanning models...", "-", "-", "-"))

        # Small delay just lets Tk repaint the "Scanning..." row before the
        # (potentially slow) model loading below blocks the main thread.
        self.after(100, lambda: self._display_test_result(email_text))

    def _display_test_result(self, email_text):
        self.result_tree.delete(*self.result_tree.get_children())

        saved_dir = "src/saved_models"
        if not os.path.isdir(saved_dir):
            self.result_tree.insert("", "end", values=("No saved_models folder found", "-", "-", "-"))
            self.test_btn.configure(state="normal")
            return

        files = os.listdir(saved_dir)
        keras_files = sorted(f for f in files if f.endswith("_model.keras"))
        hybrid_files = sorted(f for f in files if f.endswith("__hybrid.joblib"))

        if not keras_files and not hybrid_files:
            self.result_tree.insert("", "end", values=("No trained models found", "-", "-", "-"))
            self.test_btn.configure(state="normal")
            return

        # Same light clean used to build the training data (see preprocessing.py)
        light_clean = clean_text_light(email_text)

        # --- Keras models (e.g. LSTM+CNN) ---
        if keras_files:
            tokenizer_path = os.path.join(saved_dir, "tokenizer.pkl")
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, "rb") as handle:
                    tokenizer = pickle.load(handle)

                # NOTE: must be wrapped in a list -- TextVectorization expects a
                # batch of strings, not a single scalar string.
                seq = tokenizer(np.array([light_clean])).numpy()
                padded_text = pad_sequences(seq, maxlen=200, padding="post")

                for file in keras_files:
                    base_name = file.replace("_model.keras", "")
                    try:
                        model = load_model(os.path.join(saved_dir, file))
                        pred_score = float(model.predict(padded_text, verbose=0)[0][0])
                        self._insert_prediction(saved_dir, base_name, "keras", pred_score)
                    except Exception as exc:
                        self.result_tree.insert("", "end", values=(base_name.upper(), "N/A", "N/A", f"Error: {exc}"))
            else:
                self.result_tree.insert("", "end", values=("tokenizer.pickle missing", "-", "-", "-"))

        # --- Joblib hybrid models (e.g. NB+LR+RF, NB+SVM) ---
        if hybrid_files:
            try:
                stop_words, stemmer = self._get_nltk_assets()
                heavy_clean = clean_text_heavy(light_clean, stemmer, stop_words)

                for file in hybrid_files:
                    base_name = file.replace("__hybrid.joblib", "")
                    vec_path = os.path.join(saved_dir, f"{base_name}__tfidf_vectorizer.joblib")
                    if not os.path.exists(vec_path):
                        self.result_tree.insert("", "end", values=(base_name.upper(), "N/A", "N/A", "Vectorizer missing"))
                        continue
                    try:
                        hybrid_model = joblib.load(os.path.join(saved_dir, file))
                        vectorizer = joblib.load(vec_path)
                        vec = vectorizer.transform([heavy_clean])

                        classes = list(hybrid_model.classes_)
                        spam_idx = classes.index(1) if 1 in classes else 1
                        pred_score = float(hybrid_model.predict_proba(vec)[0][spam_idx])

                        self._insert_prediction(saved_dir, base_name, "joblib", pred_score)
                    except Exception as exc:
                        self.result_tree.insert("", "end", values=(base_name.upper(), "N/A", "N/A", f"Error: {exc}"))
            except Exception as exc:
                messagebox.showerror("NLTK Error", f"Could not load stopwords/stemmer: {exc}")

        self.test_btn.configure(state="normal")

    def _insert_prediction(self, saved_dir, base_name, model_type, pred_score):
        if pred_score > 0.5:
            prediction_label = "🚨 SPAM"
            confidence = pred_score * 100
        else:
            prediction_label = "✅ HAM"
            confidence = (1.0 - pred_score) * 100

        display_accuracy = self._get_model_accuracy(saved_dir, base_name, model_type)
        display_name = base_name.upper().replace("_", " ")

        self.result_tree.insert(
            "", "end",
            values=(display_name, display_accuracy, f"{confidence:.2f}%", prediction_label),
        )

    @staticmethod
    def _get_model_accuracy(saved_dir, base_name, model_type):
        json_path = os.path.join(saved_dir, f"{base_name}_metrics.json")
        if not os.path.exists(json_path):
            return "N/A"

        with open(json_path, "r") as f:
            metrics_data = json.load(f)

        if model_type == "keras":
            # lstm_cnn_metrics.json -> {"accuracy": "97.50%"}
            acc = metrics_data.get("accuracy", "N/A")
        else:
            # nb_lr_rf_metrics.json / nb_svm_metrics.json ->
            # {"Naive Bayes": {...}, "NB + LR + RF Hybrid": {"accuracy": 0.975, ...}, ...}
            hybrid_entry = next((v for k, v in metrics_data.items() if "hybrid" in k.lower()), None)
            acc = hybrid_entry.get("accuracy", "N/A") if hybrid_entry else "N/A"

        if isinstance(acc, (int, float)):
            return f"{acc * 100:.2f}%"
        return acc


if __name__ == "__main__":
    app = SpamMe()
    app.mainloop()