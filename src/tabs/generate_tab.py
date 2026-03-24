"""Generate Log-to-Log LUT tab."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

from generate_log2log_lut import LOG_CONFIGS, generate_log_to_log_lut, generate_multiple_luts
from gui_utils import run_in_thread


class GenerateTab:
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=app.tr("tab_generate"))
        self._build_ui()

    def _build_ui(self):
        app = self.app
        main_frame = ttk.Frame(self.frame, padding="15")
        main_frame.pack(fill="both", expand=True)

        # Mode selection
        mode_frame = app._t(ttk.LabelFrame, main_frame, "mode", padding="8")
        mode_frame.pack(fill="x", pady=(0, 8))

        self.gen_mode = tk.StringVar(value="single")
        app._t(
            ttk.Radiobutton, mode_frame, "single_conversion",
            variable=self.gen_mode, value="single",
        ).pack(side="left", padx=10)
        app._t(
            ttk.Radiobutton, mode_frame, "batch_conversion",
            variable=self.gen_mode, value="batch",
        ).pack(side="left", padx=10)

        # Source log selection
        source_frame = app._t(ttk.LabelFrame, main_frame, "source_log_format", padding="8")
        source_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, source_frame, "source").pack(side="left", padx=5)
        self.gen_source = ttk.Combobox(
            source_frame, values=list(LOG_CONFIGS.keys()), width=30, state="readonly"
        )
        self.gen_source.pack(side="left", padx=5, fill="x", expand=True)
        self.gen_source.current(0)

        # Target log selection (for single mode)
        target_frame = app._t(ttk.LabelFrame, main_frame, "target_log_format", padding="8")
        target_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, target_frame, "target").pack(side="left", padx=5)
        self.gen_target = ttk.Combobox(
            target_frame, values=list(LOG_CONFIGS.keys()), width=30, state="readonly"
        )
        self.gen_target.pack(side="left", padx=5, fill="x", expand=True)
        self.gen_target.current(1)

        # LUT size
        size_frame = app._t(ttk.LabelFrame, main_frame, "lut_size", padding="8")
        size_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, size_frame, "grid_size").pack(side="left", padx=5)
        self.gen_size = ttk.Combobox(
            size_frame, values=[17, 33, 65, 129], width=10, state="readonly"
        )
        self.gen_size.pack(side="left", padx=5)
        self.gen_size.set(65)

        # Output settings
        output_frame = app._t(ttk.LabelFrame, main_frame, "output", padding="8")
        output_frame.pack(fill="x", pady=(0, 8))

        app._t(ttk.Label, output_frame, "output_directory").pack(side="left", padx=5)
        self.gen_output_dir = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.gen_output_dir, width=40).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(ttk.Button, output_frame, "browse", command=self.browse_output).pack(
            side="left", padx=5
        )

        # Generate button
        app._t(
            ttk.Button, main_frame, "btn_generate_lut",
            command=self.generate_lut, style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = app._t(ttk.LabelFrame, main_frame, "output_log", padding="8")
        console_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.console.pack(fill="both", expand=True)

    def browse_output(self):
        directory = filedialog.askdirectory(initialdir=self.gen_output_dir.get())
        if directory:
            self.gen_output_dir.set(directory)

    def generate_lut(self):
        app = self.app

        def task():
            source = self.gen_source.get()
            size = int(self.gen_size.get())
            output_dir = os.path.abspath(self.gen_output_dir.get())

            if self.gen_mode.get() == "single":
                target = self.gen_target.get()
                if source == target:
                    raise ValueError(app.tr("error_same_source_target"))

                source_name = source.replace(" ", "_").replace(".", "")
                target_name = target.replace(" ", "_").replace(".", "")
                out_filename = f"{source_name}_to_{target_name}_{size}.cube"
                out_path = os.path.join(output_dir, out_filename)

                generate_log_to_log_lut(
                    source_log=source, target_log=target,
                    lut_size=size, out_path=out_path,
                )
            else:
                generate_multiple_luts(
                    source_log=source, target_logs=None,
                    lut_size=size, output_dir=output_dir,
                )

        run_in_thread(task, self.console, app.status_var, app.tr)
