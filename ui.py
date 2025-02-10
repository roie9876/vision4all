import json
import tkinter as tk
from tkinter import ttk

def load_results(json_path="results.json"):
    with open(json_path) as f:
        return json.load(f)

def update_ui():
    results = load_results()
    root = tk.Tk()
    root.title("YOLO Results")
    # ...existing UI layout code...
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    row = 0
    for key, value in results.items():
        ttk.Label(frm, text=f"{key}: {value}").grid(column=0, row=row)
        row += 1
    # If a note was added by GPT-4 integration, display it.
    if "note" in results:
        ttk.Label(frm, text=f"Note: {results['note']}").grid(column=0, row=row)
    root.mainloop()

if __name__ == "__main__":
    update_ui()
