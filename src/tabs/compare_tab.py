"""Compare Images tab."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

from compare_images import compare_px_diff, compare_image_dirs
from gui_utils import run_in_thread


class CompareTab:
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=app.tr("tab_compare"))
        self._build_ui()

    def _build_ui(self):
        app = self.app
        main_frame = ttk.Frame(self.frame, padding="15")
        main_frame.pack(fill="both", expand=True)

        # Mode selection
        mode_frame = app._t(ttk.LabelFrame, main_frame, "mode", padding="8")
        mode_frame.pack(fill="x", pady=(0, 8))

        self.mode = tk.StringVar(value="single")
        app._t(
            ttk.Radiobutton, mode_frame, "single_image_pair",
            variable=self.mode, value="single",
        ).pack(side="left", padx=10)
        app._t(
            ttk.Radiobutton, mode_frame, "directory_comparison",
            variable=self.mode, value="batch",
        ).pack(side="left", padx=10)

        # Image 1 / Directory 1
        input1_frame = app._t(ttk.LabelFrame, main_frame, "first_input", padding="8")
        input1_frame.pack(fill="x", pady=(0, 8))

        self.input1 = tk.StringVar()
        ttk.Entry(input1_frame, textvariable=self.input1, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, input1_frame, "browse",
            command=lambda: self.browse_input(1),
        ).pack(side="left", padx=5)

        # Image 2 / Directory 2
        input2_frame = app._t(ttk.LabelFrame, main_frame, "second_input", padding="8")
        input2_frame.pack(fill="x", pady=(0, 8))

        self.input2 = tk.StringVar()
        ttk.Entry(input2_frame, textvariable=self.input2, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, input2_frame, "browse",
            command=lambda: self.browse_input(2),
        ).pack(side="left", padx=5)

        # Options
        options_frame = app._t(ttk.LabelFrame, main_frame, "options", padding="8")
        options_frame.pack(fill="x", pady=(0, 8))

        self.visualize = tk.BooleanVar(value=True)
        app._t(
            ttk.Checkbutton, options_frame, "generate_visualization",
            variable=self.visualize,
        ).pack(side="left", padx=10)

        app._t(ttk.Label, options_frame, "amplification").pack(side="left", padx=5)
        self.amplification = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            options_frame, from_=0.1, to=100.0, increment=0.5,
            textvariable=self.amplification, width=10,
        ).pack(side="left", padx=5)

        app._t(ttk.Label, options_frame, "workers").pack(side="left", padx=5)
        self.workers = tk.IntVar(value=4)
        ttk.Spinbox(
            options_frame, from_=1, to=16,
            textvariable=self.workers, width=10,
        ).pack(side="left", padx=5)

        # Output
        output_frame = app._t(ttk.LabelFrame, main_frame, "output", padding="8")
        output_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, output_frame, "output_path").pack(side="left", padx=5)
        self.output = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, output_frame, "browse", command=self.browse_output
        ).pack(side="left", padx=5)

        # Compare button
        app._t(
            ttk.Button, main_frame, "btn_compare_images",
            command=self.compare_images, style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = app._t(ttk.LabelFrame, main_frame, "output_log", padding="8")
        console_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.console.pack(fill="both", expand=True)

    def browse_input(self, input_num):
        path_var = self.input1 if input_num == 1 else self.input2

        if self.mode.get() == "single":
            path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.tif *.tiff *.png *.jpg")]
            )
        else:
            path = filedialog.askdirectory()

        if path:
            path_var.set(path)

    def browse_output(self):
        if self.mode.get() == "single":
            path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG files", "*.png")]
            )
        else:
            path = filedialog.askdirectory()

        if path:
            self.output.set(path)

    def compare_images(self):
        app = self.app

        def task():
            input1 = os.path.abspath(self.input1.get())
            input2 = os.path.abspath(self.input2.get())
            output = (
                os.path.abspath(self.output.get()) if self.output.get() else None
            )
            visualize = self.visualize.get()
            amplification = self.amplification.get()
            workers = self.workers.get()

            if not input1 or not input2:
                raise ValueError(app.tr("error_specify_both_inputs"))

            if self.mode.get() == "single":
                compare_px_diff(
                    input1, input2,
                    visualize=visualize, output_path=output,
                    amplification=amplification,
                )
            else:
                compare_image_dirs(
                    input1, input2,
                    visualize=visualize, output_dir=output,
                    amplification=amplification, workers=workers,
                )

        run_in_thread(task, self.console, app.status_var, app.tr)
