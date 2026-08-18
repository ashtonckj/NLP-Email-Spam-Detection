# """
# GUI for selecting and training spam detection models.
# """
# import os
# import tkinter as tk
# import threading
# from tkinter import ttk, filedialog, messagebox
# import traceback

# # Import model functions
# from lstm_cnn import train_lstm_cnn


# def train_nb_svm(filepath):
#     raise NotImplementedError(
#         "NB + SVM model is not connected yet."
#     )


# def train_ensemble(filepath):
#     raise NotImplementedError(
#         "NB + LR + RF model is not connected yet."
#     )

# # Palette
# BG      = "#1e2229"
# CARD    = "#272c35"
# FG      = "#e6e9ef"
# MUTED   = "#8b93a7"
# ACCENT  = "#5b8def"
# OK      = "#4cc38a"
# ERR     = "#e5645e"

# DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "Ceas08_Enron.csv")


# class ModelApp(tk.Tk):

#     def __init__(self):
#         super().__init__()
#         self.title("Spam Email Detection — Model Trainer")
#         self.geometry("860x640")
#         self.minsize(720, 520)
#         self.configure(bg=BG)
#         self.result = None
#         self._setup_style()
#         self.build_widgets()

#     def _setup_style(self):
#         s = ttk.Style(self)
#         s.theme_use("clam")

#         s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
#         s.configure("Card.TFrame", background=CARD, relief="flat")
#         s.configure("Bg.TFrame", background=BG)

#         s.configure("Title.TLabel", background=BG, foreground=FG,
#                     font=("Segoe UI Semibold", 18))
#         s.configure("Sub.TLabel", background=BG, foreground=MUTED,
#                     font=("Segoe UI", 10))
#         s.configure("Card.TLabel", background=CARD, foreground=MUTED,
#                     font=("Segoe UI Semibold", 9))
#         s.configure("Status.TLabel", background=BG, foreground=MUTED,
#                     font=("Segoe UI", 9))
#         s.configure("Result.TLabel", background=BG, foreground=OK,
#                     font=("Segoe UI Semibold", 13))

#         s.configure("TEntry", fieldbackground="#1a1e24", foreground=FG,
#                     bordercolor="#3a4150", insertcolor=FG, padding=6)
#         s.configure("TCombobox", fieldbackground="#1a1e24", background="#1a1e24",
#                     foreground=FG, arrowcolor=MUTED, padding=5)
#         s.map("TCombobox", fieldbackground=[("readonly", "#1a1e24")],
#               foreground=[("readonly", FG)])

#         s.configure("Ghost.TButton", background="#343b47", foreground=FG,
#                     borderwidth=0, padding=(14, 7))
#         s.map("Ghost.TButton", background=[("active", "#3f4757")])

#         s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
#                     borderwidth=0, font=("Segoe UI Semibold", 10), padding=(22, 9))
#         s.map("Accent.TButton",
#               background=[("active", "#6f9bf2"), ("disabled", "#3a4150")],
#               foreground=[("disabled", MUTED)])

#         s.configure("Bar.Horizontal.TProgressbar", background=ACCENT,
#                     troughcolor=CARD, borderwidth=0, thickness=3)

#     def _card(self, parent, label):
#         wrap = ttk.Frame(parent, style="Card.TFrame", padding=(16, 12))
#         wrap.pack(fill="x", padx=20, pady=(0, 12))
#         ttk.Label(wrap, text=label.upper(), style="Card.TLabel").pack(anchor="w")
#         row = ttk.Frame(wrap, style="Card.TFrame")
#         row.pack(fill="x", pady=(8, 0))
#         return row

#     def build_widgets(self):
#         # ---- Header ----
#         head = ttk.Frame(self, style="Bg.TFrame", padding=(20, 18, 20, 14))
#         head.pack(fill="x")
#         ttk.Label(head, text="Spam Email Detection", style="Title.TLabel").pack(anchor="w")
#         ttk.Label(head, text="Select a cleaned dataset and a model, then train.",
#                   style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

#         # ---- Dataset card ----
#         row = self._card(self, "Cleaned dataset")
#         self.path_var = tk.StringVar(value=DEFAULT_DATA)
#         ttk.Entry(row, textvariable=self.path_var).pack(
#             side="left", fill="x", expand=True)
#         ttk.Button(row, text="Browse", style="Ghost.TButton",
#                    command=self.browse_file).pack(side="left", padx=(10, 0))

#         # ---- Model card ----
#         row = self._card(self, "Model")
#         self.model_var = tk.StringVar()
#         self.model_dropdown = ttk.Combobox(
#             row, textvariable=self.model_var, state="readonly", width=26,
#             values=["NB + SVM", "NB + LR + RF", "LSTM + CNN"])
#         self.model_dropdown.current(0)
#         self.model_dropdown.pack(side="left")
#         self.train_btn = ttk.Button(row, text="Train Model", style="Accent.TButton",
#                                     command=self.train_model)
#         self.train_btn.pack(side="right")

#         self.progress = ttk.Progressbar(
#             self,
#             mode="determinate",
#             maximum=100,
#             style="Bar.Horizontal.TProgressbar"
#         )

#         self.progress.pack(
#             fill="x",
#             padx=20,
#             pady=(5, 10)
#         )

#         # ---- Log ----
#         logwrap = ttk.Frame(self, style="Card.TFrame", padding=(14, 12))
#         logwrap.pack(fill="both", expand=True, padx=20, pady=12)
#         ttk.Label(logwrap, text="OUTPUT", style="Card.TLabel").pack(anchor="w")

#         inner = ttk.Frame(logwrap, style="Card.TFrame")
#         inner.pack(fill="both", expand=True, pady=(8, 0))

#         self.log = tk.Text(inner, state="disabled", wrap="none", bd=0,
#                            bg="#161a20", fg="#c8cede", insertbackground=FG,
#                            font=("Cascadia Mono", 9), padx=12, pady=10,
#                            relief="flat", highlightthickness=0)
#         self.log.pack(side="left", fill="both", expand=True)
#         sb = ttk.Scrollbar(inner, orient="vertical", command=self.log.yview)
#         sb.pack(side="right", fill="y")
#         self.log.configure(yscrollcommand=sb.set)

#         self.log.tag_configure("info", foreground="#c8cede")
#         self.log.tag_configure("ok",   foreground=OK)
#         self.log.tag_configure("err",  foreground=ERR)
#         self.log.tag_configure("dim",  foreground=MUTED)

#         # ---- Footer ----
#         foot = ttk.Frame(self, style="Bg.TFrame", padding=(20, 0, 20, 16))
#         foot.pack(fill="x")
#         self.result_label = ttk.Label(foot, text="No result yet", style="Status.TLabel")
#         self.result_label.pack(side="left")
#         self.status = ttk.Label(foot, text="Ready", style="Status.TLabel")
#         self.status.pack(side="right")

#     def browse_file(self):
#         filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
#         if filepath:
#             self.path_var.set(filepath)
#             self.write_log(f"Dataset set to {filepath}", "dim")

#     def set_busy(self, busy):
#         self.train_btn.configure(
#             state="disabled" if busy else "normal"
#         )

#         self.status.configure(
#             text="Training…" if busy else "Ready"
#         )

#         if busy:
#             self.progress["value"] = 0
#         else:
#             self.progress["value"] = 100


#     def browse_file(self):
#         filepath = filedialog.askopenfilename(
#             filetypes=[
#                 ("CSV Files","*.csv")
#             ]
#         )
#         if filepath:
#             self.path_var.set(filepath)

#     def write_log(self, text):
#         self.log.configure(
#             state="normal"
#         )
#         self.log.insert(
#             "end",
#             text + "\n"
#         )
#         self.log.see(
#             "end"
#         )
#         self.log.configure(
#             state="disabled"
#         )

#     def train_model(self):
#         filepath = self.path_var.get()
#         selected_model = self.model_var.get()

#         self.set_busy(True)

#         self.write_log(
#             f"Selected Model: {selected_model}"
#         )

#         threading.Thread(
#             target=self._run_training,
#             args=(filepath, selected_model),
#             daemon=True
#         ).start()

#     def _run_training(self, filepath, selected_model):
#         try:
#             if selected_model == "LSTM + CNN":
#                 result = train_lstm_cnn(filepath, self.write_log, self.update_progress)
#             elif selected_model == "NB + SVM":
#                 result = train_nb_svm(filepath)
#             else:
#                 result = train_ensemble(filepath)
#             self.after(0, lambda: self._done(result))
#         except Exception as e:
#             tb = traceback.format_exc()
#             self.after(0, lambda err=e, t=tb: (self.write_log(t), messagebox.showerror("Error", str(err))))

#     def _done(self, result):
#         self.progress["value"] = 100
#         self.set_busy(False)

#         self.result = result
#         self.show_result()

#     def show_result(self):

#         if self.result is None:
#             return

#         accuracy = self.result.get("accuracy", "N/A")
#         report = self.result.get("report", "")

#         self.write_log("")
#         self.write_log("===================================")
#         self.write_log("Training Completed!")
#         self.write_log("===================================")
#         self.write_log(f"Accuracy : {accuracy:.4f}")
#         self.write_log("")
#         self.write_log("Classification Report")
#         self.write_log(report)

#         self.result_label.config(
#             text=f"Accuracy: {accuracy:.4f}"
#         )
        

#     def update_progress(self, epoch):
#         total_epoch = 10
#         value = (epoch / total_epoch) * 100

#         self.after(
#             0,
#             lambda: self.progress.configure(value=value)
#         )

# if __name__ == "__main__":

#     ModelApp().mainloop()