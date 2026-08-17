import json
import os
import pickle
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Import your external functions here
from data_cleaning import clean_text, run_cleaning_pipeline
from lstm_cnn import train_lstm_cnn

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
        self.trained_model = None # Placeholder for loaded model in testing
        
        self._setup_style()
        self._build_tabs()

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

        s.configure("Treeview", background="#1a1e24", foreground=FG,  rowheight=25, fieldbackground="#1a1e24", borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT)],  foreground=[("selected", "#ffffff")])
        s.configure("Treeview.Heading", background=CARD,  foreground=MUTED, font=("Segoe UI Semibold", 9), borderwidth=0)

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create Frames for each tab
        self.tab_cleaning = ttk.Frame(self.notebook, style="Bg.TFrame")
        self.tab_training = ttk.Frame(self.notebook, style="Bg.TFrame")
        self.tab_testing = ttk.Frame(self.notebook, style="Bg.TFrame")

        self.notebook.add(self.tab_cleaning, text="1. Data Cleaning")
        self.notebook.add(self.tab_training, text="2. Model Training")
        self.notebook.add(self.tab_testing, text="3. Model Testing")

        self._build_cleaning_tab()
        self._build_training_tab()
        self._build_testing_tab()

    # ==========================================
    # TAB 1: DATA CLEANING
    # ==========================================
    def _build_cleaning_tab(self):
        head = ttk.Frame(self.tab_cleaning, style="Bg.TFrame", padding=(20, 18, 20, 14))
        head.pack(fill="x")
        ttk.Label(head, text="Data Preprocessing", style="Title.TLabel").pack(anchor="w")
        
        row = ttk.Frame(self.tab_cleaning, style="Card.TFrame", padding=(16, 12))
        row.pack(fill="x", padx=20, pady=(0, 12))
        
        self.clean_path_var = tk.StringVar(value="data/spam.csv")
        ttk.Entry(row, textvariable=self.clean_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", style="Ghost.TButton", command=self._browse_clean).pack(side="left", padx=(10, 10))
        self.start_clean_btn = ttk.Button(row, text="Start Cleaning", style="Accent.TButton", command=self._start_cleaning)
        self.start_clean_btn.pack(side="right")

        # Log & Preview Area
        wrap = ttk.Frame(self.tab_cleaning, style="Bg.TFrame")
        wrap.pack(fill="both", expand=True, padx=20, pady=5)
        
        self.clean_log = tk.Text(wrap, height=8, state="disabled", bg="#161a20", fg="#c8cede", bd=0)
        self.clean_log.pack(fill="x", pady=(0, 10))

        self.preview = ttk.Treeview(wrap, show="headings", height=6)
        self.preview.pack(fill="both", expand=True)

        self.save_clean_btn = ttk.Button(self.tab_cleaning, text="Save Cleaned CSV", style="Ghost.TButton", command=self._save_cleaned, state="disabled")
        self.save_clean_btn.pack(pady=15)

    def _browse_clean(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path: self.clean_path_var.set(path)

    def _log_clean(self, text):
        self.clean_log.configure(state="normal")
        self.clean_log.insert("end", text + "\n")
        self.clean_log.see("end")
        self.clean_log.configure(state="disabled")

    def _show_preview(self, df):
        self.preview.delete(*self.preview.get_children())
        self.preview["columns"] = list(df.columns)
        for col in df.columns:
            self.preview.heading(col, text=col)
            self.preview.column(col, width=180)
        for _, row in df.iterrows():
            self.preview.insert("", "end", values=list(row))

    def _start_cleaning(self):
        self.start_clean_btn.configure(state="disabled")
        self.save_clean_btn.configure(state="disabled")
        self.clean_log.configure(state="normal")
        self.clean_log.delete("1.0", "end")
        self.clean_log.configure(state="disabled")
        try:
            self.pipeline = run_cleaning_pipeline(self.clean_path_var.get())
            self._step_cleaning()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.start_clean_btn.configure(state="normal")

    def _step_cleaning(self):
        try:
            step, message, data = next(self.pipeline)
            self._log_clean(f"[{step}] {message}")
            if data is not None:
                if step == "done":
                    self.cleaned_df = data
                    self.save_clean_btn.configure(state="normal")
                self._show_preview(data)
            self.after(STEP_DELAY_MS, self._step_cleaning)
        except StopIteration:
            self.start_clean_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.start_clean_btn.configure(state="normal")

    def _save_cleaned(self):
        if self.cleaned_df is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.cleaned_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Cleaned data saved to {path}")
            # Auto-populate the training tab with the new cleaned file
            self.train_path_var.set(path)

    # ==========================================
    # TAB 2: MODEL TRAINING
    # ==========================================
    def _build_training_tab(self):
        head = ttk.Frame(self.tab_training, style="Bg.TFrame", padding=(20, 18, 20, 14))
        head.pack(fill="x")
        ttk.Label(head, text="Train Classification Model", style="Title.TLabel").pack(anchor="w")

        # Dataset & Model Selection
        row1 = ttk.Frame(self.tab_training, style="Card.TFrame", padding=(16, 12))
        row1.pack(fill="x", padx=20, pady=(0, 12))
        ttk.Label(row1, text="CLEANED DATASET", style="Card.TLabel").pack(anchor="w")
        
        inner_row = ttk.Frame(row1, style="Card.TFrame")
        inner_row.pack(fill="x", pady=(5, 10))
        self.train_path_var = tk.StringVar(value=DEFAULT_DATA)
        ttk.Entry(inner_row, textvariable=self.train_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(inner_row, text="Browse", style="Ghost.TButton", command=self._browse_train).pack(side="left", padx=(10, 0))

        ttk.Label(row1, text="SELECT ALGORITHM", style="Card.TLabel").pack(anchor="w")
        inner_row2 = ttk.Frame(row1, style="Card.TFrame")
        inner_row2.pack(fill="x", pady=(5, 0))
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(inner_row2, textvariable=self.model_var, state="readonly", width=26, values=["NB + SVM", "NB + LR + RF", "LSTM + CNN"])
        self.model_dropdown.current(2)
        self.model_dropdown.pack(side="left")
        self.train_btn = ttk.Button(inner_row2, text="Start Training", style="Accent.TButton", command=self._start_training)
        self.train_btn.pack(side="right")

        self.progress = ttk.Progressbar(self.tab_training, mode="determinate", maximum=100, style="Bar.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=20, pady=(5, 10))

        # Output Log
        logwrap = ttk.Frame(self.tab_training, style="Card.TFrame", padding=(14, 12))
        logwrap.pack(fill="both", expand=True, padx=20, pady=5)
        self.train_log = tk.Text(logwrap, state="disabled", wrap="none", bd=0, bg="#161a20", fg="#c8cede", font=("Cascadia Mono", 9))
        self.train_log.pack(side="left", fill="both", expand=True)

    def _browse_train(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path: self.train_path_var.set(path)

    def _log_train(self, text):
        self.train_log.configure(state="normal")
        self.train_log.insert("end", text + "\n")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")

    def _update_progress(self, current_epoch):
        total_epochs = 10 
        value = (current_epoch / total_epochs) * 100
        
        self.after(0, lambda: self.progress.configure(value=value))

    def _start_training(self):
        filepath = self.train_path_var.get()
        selected_model = self.model_var.get()
        self.train_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", "end")
        self.train_log.configure(state="disabled")
        self._log_train(f"Starting training for {selected_model}...")

        threading.Thread(target=self._run_training_thread, args=(filepath, selected_model), daemon=True).start()

    def _run_training_thread(self, filepath, selected_model):
        try:
            if selected_model == "LSTM + CNN":
               result = train_lstm_cnn(filepath, self._log_train, self._update_progress)
            else:
                self._log_train("Model not yet implemented.")
                result = None
            self.after(0, lambda: self._training_done(result))
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, lambda: (self._log_train(tb), self.train_btn.configure(state="normal")))

    def _training_done(self, result):
        self.progress["value"] = 100
        self.train_btn.configure(state="normal")
        if result:
            self.training_result = result
            self._log_train(f"\nFinal Accuracy: {result.get('accuracy', 0):.4f}")
            self._log_train(result.get("report", ""))

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
        self.test_input = tk.Text(input_frame, height=8, bd=0, bg="#1a1e24", fg=FG, font=("Segoe UI", 11), padx=10, pady=10)
        self.test_input.pack(fill="both", expand=True, pady=(0, 10))

        self.test_btn = ttk.Button(input_frame, text="Analyze Email", style="Accent.TButton", command=self._run_test)
        self.test_btn.pack(side="right")

        # Result Area (Multi-Model Table)
        self.result_frame = ttk.Frame(self.tab_testing, style="Card.TFrame", padding=(16, 12))
        self.result_frame.pack(fill="x", padx=20, pady=(0, 20))
        ttk.Label(self.result_frame, text="MODEL PREDICTIONS", style="Card.TLabel").pack(anchor="w", pady=(0, 10))

        # 4 Columns table for result
        self.result_tree = ttk.Treeview(self.result_frame, columns=("Model", "Accuracy", "Confidence", "Prediction"), show="headings", height=4)
        self.result_tree.pack(fill="x")
        
        self.result_tree.heading("Model", text="Trained Model")
        self.result_tree.heading("Accuracy", text="Overall Accuracy")
        self.result_tree.heading("Confidence", text="Confidence Score")
        self.result_tree.heading("Prediction", text="Prediction")
        
        self.result_tree.column("Model", width=160, anchor="center")
        self.result_tree.column("Accuracy", width=120, anchor="center")
        self.result_tree.column("Confidence", width=120, anchor="center")
        self.result_tree.column("Prediction", width=150, anchor="center")

    def _run_test(self):
        email_text = self.test_input.get("1.0", "end").strip()
        if not email_text:
            messagebox.showwarning("Empty Input", "Please paste some email text to analyze.")
            return
            
        self.test_btn.configure(state="disabled")
        
        # Clear previous results
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_tree.insert("", "end", values=("Scanning folder...", "-", "-", "-"))
        
        # Simulate loading from a folder and predicting
        self.after(1500, lambda: self._display_test_result(email_text))

    def _display_test_result(self, email_text):
        self.test_btn.configure(state="normal")
        self.result_tree.delete(*self.result_tree.get_children())
    
        # 1. Clean the user's input text (email_text) just like you did in training
        cleaned_text = clean_text(email_text)

        # 2. Look inside the folder
        saved_dir = "saved_models"
        if os.path.exists(saved_dir):
            
            # 3. Load the tokenizer you saved during training to convert the text to numbers
            with open(f"{saved_dir}/tokenizer.pkl", 'rb') as handle:
                tokenizer = pickle.load(handle)
                
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded_text = pad_sequences(seq, maxlen=200, padding="post")

            # 4. Find all the saved models and test them one by one
            for file in os.listdir(saved_dir):
                if file.endswith(".keras"):
                    
                    # 1. Load the Model to predict
                    model_path = os.path.join(saved_dir, file)
                    model = load_model(model_path)
                    
                    # Get raw prediction score
                    pred_score = float(model.predict(padded_text)[0][0])
                    
                    # Calculate Confidence Score
                    if pred_score > 0.5:
                        prediction_label = "🚨 SPAM"
                        confidence = pred_score * 100 
                    else:
                        prediction_label = "✅ HAM"
                        confidence = (1.0 - pred_score) * 100 
                        
                    display_confidence = f"{confidence:.2f}%"
                    
                    
                    # --- 2. Read Overall Accuracy from JSON ---
                    base_name = file.replace("_model.keras", "")
                    json_file = f"{base_name}_metrics.json"
                    json_path = os.path.join(saved_dir, json_file)
                    
                    display_accuracy = "N/A"
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            metrics_data = json.load(f)
                            display_accuracy = metrics_data.get("accuracy", "N/A")
                    
                    display_name = base_name.upper()
                    
                    # --- 3. Insert all 4 values into the table ---
                    self.result_tree.insert("", "end", values=(display_name, display_accuracy, display_confidence, prediction_label))
        
if __name__ == "__main__":
    app = SpamMe()
    app.mainloop()