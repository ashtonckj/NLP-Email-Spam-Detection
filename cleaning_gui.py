# """GUI: click a button, watch the data cleaning pipeline run step by step."""
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from data_cleaning import run_cleaning_pipeline

# STEP_DELAY_MS = 800  # pause between steps so progress is visible

 
# class CleaningApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Spam Data Cleaner")
#         self.geometry("680x520")
#         self.cleaned_df = None
#         self.pipeline = None
#         self._build_widgets()

#     def _build_widgets(self):
#         """Lay out the file picker, buttons, log console, and preview table."""
#         top = ttk.Frame(self, padding=10)
#         top.pack(fill="x")

#         ttk.Label(top, text="CSV file:").pack(side="left")
#         self.path_var = tk.StringVar(value="data/spam.csv")
#         ttk.Entry(top, textvariable=self.path_var, width=40).pack(side="left", padx=5)
#         ttk.Button(top, text="Browse", command=self._browse).pack(side="left")

#         self.start_btn = ttk.Button(top, text="Start Cleaning", command=self._start)
#         self.start_btn.pack(side="right")

#         self.log = tk.Text(self, height=14, state="disabled", bg="#111111", fg="#33ff33")
#         self.log.pack(fill="both", expand=True, padx=10, pady=5)

#         self.preview = ttk.Treeview(self, show="headings", height=6)
#         self.preview.pack(fill="both", expand=True, padx=10, pady=5)

#         self.save_btn = ttk.Button(self, text="Save Cleaned CSV", command=self._save, state="disabled")
#         self.save_btn.pack(pady=5)

#     def _browse(self):
#         """Open a file dialog so the user can pick a CSV."""
#         path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#         if path:
#             self.path_var.set(path)

#     def _log(self, text):
#         """Append a line to the on-screen console."""
#         self.log.configure(state="normal")
#         self.log.insert("end", text + "\n")
#         self.log.see("end")
#         self.log.configure(state="disabled")

#     def _show_preview(self, df):
#         """Render a DataFrame in the Treeview table."""
#         self.preview.delete(*self.preview.get_children())
#         self.preview["columns"] = list(df.columns)
#         for col in df.columns:
#             self.preview.heading(col, text=col)
#             self.preview.column(col, width=180)
#         for _, row in df.iterrows():
#             self.preview.insert("", "end", values=list(row))

#     def _start(self):
#         """Kick off the pipeline generator and disable controls while running."""
#         self.start_btn.configure(state="disabled")
#         self.save_btn.configure(state="disabled")
#         self.log.configure(state="normal")
#         self.log.delete("1.0", "end")
#         self.log.configure(state="disabled")
#         try:
#             self.pipeline = run_cleaning_pipeline(self.path_var.get())
#         except Exception as e:
#             messagebox.showerror("Error", str(e))
#             self.start_btn.configure(state="normal")
#             return
#         self._step()

#     def _step(self):
#         """Pull one step from the generator, display it, then schedule the next."""
#         try:
#             step, message, data = next(self.pipeline)
#         except StopIteration:
#             self.start_btn.configure(state="normal")
#             return
#         except Exception as e:
#             messagebox.showerror("Error", str(e))
#             self.start_btn.configure(state="normal")
#             return

#         self._log(f"[{step}] {message}")
#         if data is not None:
#             if step == "done":
#                 self.cleaned_df = data
#                 self.save_btn.configure(state="normal")
#             self._show_preview(data)

#         self.after(STEP_DELAY_MS, self._step)

#     def _save(self):
#         """Export the cleaned DataFrame to a CSV the user chooses."""
#         if self.cleaned_df is None:
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
#         if path:
#             self.cleaned_df.to_csv(path, index=False)
#             messagebox.showinfo("Saved", f"Cleaned data saved to {path}")


# if __name__ == "__main__":
#     CleaningApp().mainloop()