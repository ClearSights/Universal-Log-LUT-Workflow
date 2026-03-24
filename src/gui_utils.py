"""Shared utilities for the GUI: stdout redirection, resource paths, threaded execution."""

import os
import sys
import tkinter as tk
import threading
from tkinter import messagebox


class RedirectText:
    """Redirect stdout/stderr to a text widget"""

    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, string):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", string, self.tag)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")
        self.text_widget.update_idletasks()

    def flush(self):
        pass


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)


def run_in_thread(func, console_widget, status_var, tr):
    """Run a function in a separate thread and redirect output to console"""

    def wrapper():
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = RedirectText(console_widget, "stdout")
        sys.stderr = RedirectText(console_widget, "stderr")

        try:
            func()
            status_var.set(tr("status_completed"))
        except Exception as e:
            print(f"\nError: {e}")
            status_var.set(tr("status_error").format(e))
            messagebox.showerror(tr("error_title"), str(e))
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    # Clear console
    console_widget.configure(state="normal")
    console_widget.delete(1.0, tk.END)
    console_widget.configure(state="disabled")

    status_var.set(tr("status_processing"))
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
