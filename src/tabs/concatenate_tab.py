"""Concatenate LUTs tab."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

from concatenate_luts import process_luts
from gui_utils import run_in_thread


class ConcatenateTab:
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=app.tr("tab_concatenate"))
        self._build_ui()

    def _build_ui(self):
        app = self.app
        main_frame = ttk.Frame(self.frame, padding="15")
        main_frame.pack(fill="both", expand=True)

        # Input 1
        input1_frame = app._t(
            ttk.LabelFrame, main_frame, "first_input_applied_first", padding="8"
        )
        input1_frame.pack(fill="x", pady=(0, 8))

        self.input1_type = tk.StringVar(value="file")
        app._t(
            ttk.Radiobutton, input1_frame, "file",
            variable=self.input1_type, value="file",
        ).pack(side="left", padx=5)
        app._t(
            ttk.Radiobutton, input1_frame, "directory",
            variable=self.input1_type, value="dir",
        ).pack(side="left", padx=5)

        self.input1 = tk.StringVar()
        ttk.Entry(input1_frame, textvariable=self.input1, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, input1_frame, "browse",
            command=lambda: self.browse_input(1),
        ).pack(side="left", padx=5)

        # Input 2
        input2_frame = app._t(
            ttk.LabelFrame, main_frame, "second_input_applied_second", padding="8"
        )
        input2_frame.pack(fill="x", pady=(0, 8))

        self.input2_type = tk.StringVar(value="file")
        app._t(
            ttk.Radiobutton, input2_frame, "file",
            variable=self.input2_type, value="file",
        ).pack(side="left", padx=5)
        app._t(
            ttk.Radiobutton, input2_frame, "directory",
            variable=self.input2_type, value="dir",
        ).pack(side="left", padx=5)

        self.input2 = tk.StringVar()
        ttk.Entry(input2_frame, textvariable=self.input2, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        app._t(
            ttk.Button, input2_frame, "browse",
            command=lambda: self.browse_input(2),
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

        # Workers
        workers_frame = ttk.Frame(main_frame)
        workers_frame.pack(fill="x", pady=(0, 4))

        app._t(ttk.Label, workers_frame, "parallel_workers").pack(side="left", padx=5)
        self.workers = tk.IntVar(value=4)
        ttk.Spinbox(
            workers_frame, from_=1, to=16, textvariable=self.workers, width=10
        ).pack(side="left", padx=5)

        # Concatenate button
        app._t(
            ttk.Button, main_frame, "btn_concatenate_luts",
            command=self.concatenate_luts, style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = app._t(ttk.LabelFrame, main_frame, "output_log", padding="8")
        console_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.console.pack(fill="both", expand=True)

        # Results
        results_frame = app._t(ttk.LabelFrame, main_frame, "results", padding="8")
        results_frame.pack(fill="both", expand=True, pady=(0, 4))

        columns = ("name", "status", "clipped", "clip_ratio", "output")
        self.results_tree = ttk.Treeview(
            results_frame, columns=columns, show="headings", height=8
        )
        self.results_tree.pack(fill="both", expand=True)

        self.results_tree.heading("name", text=app.tr("col_name"))
        self.results_tree.heading("status", text=app.tr("col_status"))
        self.results_tree.heading("clipped", text=app.tr("col_clipped"))
        self.results_tree.heading("clip_ratio", text=app.tr("col_clip_ratio"))
        self.results_tree.heading("output", text=app.tr("col_output"))

        self.results_tree.column("name", width=200)
        self.results_tree.column("status", width=80, anchor="center")
        self.results_tree.column("clipped", width=80, anchor="center")
        self.results_tree.column("clip_ratio", width=80, anchor="center")
        self.results_tree.column("output", width=400)

    def browse_input(self, input_num):
        if input_num == 1:
            type_var = self.input1_type
            path_var = self.input1
        else:
            type_var = self.input2_type
            path_var = self.input2

        if type_var.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("CUBE files", "*.cube")])
        else:
            path = filedialog.askdirectory()

        if path:
            path_var.set(path)

    def browse_output(self):
        is_batch = self.input1_type.get() == "dir" or self.input2_type.get() == "dir"

        if is_batch:
            path = filedialog.askdirectory()
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".cube", filetypes=[("CUBE files", "*.cube")]
            )

        if path:
            self.output.set(path)

    def update_results(self, results):
        tree = self.results_tree

        for row in tree.get_children():
            tree.delete(row)

        for item in results:
            clipped = item.get("clipped", False)
            status = item.get("status", "unknown")
            tag = "error" if status != "ok" else ("clipped" if clipped else "ok")

            tree.insert(
                "", tk.END,
                values=(
                    item.get("name", ""),
                    status.upper(),
                    "YES" if clipped else "NO",
                    f"{item.get('clip_ratio', 0.0) * 100:.2f}%",
                    item.get("output", ""),
                ),
                tags=(tag,),
            )

        tree.tag_configure("ok", background="#e8f5e9")
        tree.tag_configure("clipped", background="#fff8e1")
        tree.tag_configure("error", background="#ffebee")

    def concatenate_luts(self):
        app = self.app

        def task():
            input1 = os.path.abspath(self.input1.get())
            input2 = os.path.abspath(self.input2.get())
            output = os.path.abspath(self.output.get())
            workers = self.workers.get()

            if not input1 or not input2:
                raise ValueError(app.tr("error_specify_both_inputs"))
            if not output:
                raise ValueError(app.tr("error_specify_output"))

            results = process_luts(input1, input2, output, max_workers=workers)
            app.root.after(0, lambda: self.update_results(results))

        run_in_thread(task, self.console, app.status_var, app.tr)

    def refresh_language(self, tr):
        """Update non-widget text (treeview headings)."""
        heading_map = {
            "name": "col_name",
            "status": "col_status",
            "clipped": "col_clipped",
            "clip_ratio": "col_clip_ratio",
            "output": "col_output",
        }
        for col, key in heading_map.items():
            self.results_tree.heading(col, text=tr(key))
