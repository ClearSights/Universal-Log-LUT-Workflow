"""Resize LUT tab."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

from resize_lut import resize_lut
from gui_utils import run_in_thread


class ResizeTab:
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=app.tr("tab_resize"))
        self._build_ui()

    def _build_ui(self):
        app = self.app
        main_frame = ttk.Frame(self.frame, padding="15")
        main_frame.pack(fill="both", expand=True)

        # Input file
        input_frame = app._t(ttk.LabelFrame, main_frame, "input_lut", padding="8")
        input_frame.pack(fill="x", pady=(0, 8))

        self.input = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, input_frame, "browse", command=self.browse_input
        ).pack(side="left", padx=5)

        # Target size
        size_frame = app._t(ttk.LabelFrame, main_frame, "target_size", padding="8")
        size_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, size_frame, "new_grid_size").pack(side="left", padx=5)
        self.size = ttk.Combobox(
            size_frame, values=[17, 33, 65, 129], width=10, state="readonly"
        )
        self.size.pack(side="left", padx=5)
        self.size.set(33)

        # Output file
        output_frame = app._t(ttk.LabelFrame, main_frame, "output", padding="8")
        output_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, output_frame, "output_file").pack(side="left", padx=5)
        self.output = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, output_frame, "browse", command=self.browse_output
        ).pack(side="left", padx=5)

        app._t(ttk.Label, output_frame, "auto_generate_hint", foreground="gray").pack(
            side="left", padx=5
        )

        # Resize button
        app._t(
            ttk.Button, main_frame, "btn_resize_lut",
            command=self.resize_lut_action, style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = app._t(ttk.LabelFrame, main_frame, "output_log", padding="8")
        console_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.console = scrolledtext.ScrolledText(
            console_frame, height=20, state="disabled", bg="white", fg="black"
        )
        self.console.pack(fill="both", expand=True)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("CUBE files", "*.cube")])
        if path:
            self.input.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".cube", filetypes=[("CUBE files", "*.cube")]
        )
        if path:
            self.output.set(path)

    def resize_lut_action(self):
        app = self.app

        def task():
            input_path = os.path.abspath(self.input.get())
            output_path = self.output.get()
            target_size = int(self.size.get())

            if not input_path:
                raise ValueError(app.tr("error_specify_input"))

            if not output_path:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}_{target_size}{ext}"
            else:
                output_path = os.path.abspath(output_path)

            resize_lut(input_path, output_path, target_size)

        run_in_thread(task, self.console, app.status_var, app.tr)
