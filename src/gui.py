"""
GUI for Universal Log LUT Workflow
Integrates all LUT processing tools in a user-friendly interface
"""

import os
import sys
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue

from i18n import TRANSLATIONS, SUPPORTED_LANGUAGES, detect_language

# Import the backend modules
try:
    from generate_log2log_lut import (
        LOG_CONFIGS,
        generate_log_to_log_lut,
        generate_multiple_luts,
    )
    from concatenate_luts import process_luts
    from compare_images import compare_px_diff, compare_image_dirs
    from resize_lut import resize_lut
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required scripts are in the same directory")
    sys.exit(1)


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


class LUTWorkflowGUI:
    def __init__(self, root):
        self.root = root
        self.current_lang = detect_language()
        self._i18n_map = []
        self.root.title(self.tr("window_title"))
        self.root.geometry("900x800")

        # Set window icon
        self._set_window_icon()

        # Configure style
        self.style = ttk.Style()

        # Select default theme based on OS
        default_theme = "clam"  # Default for Linux/Mac
        if platform.system() == "Windows":
            available_themes = self.style.theme_names()
            if "vista" in available_themes:
                default_theme = "vista"
            elif "winnative" in available_themes:
                default_theme = "winnative"

        self.style.theme_use(default_theme)

        # Theme and language selection frame at the top
        top_frame = ttk.Frame(root)
        top_frame.pack(fill="x", padx=5, pady=5)

        self._t(ttk.Label, top_frame, "theme_label").pack(side="left", padx=5)
        self.theme_var = tk.StringVar(value=self.style.theme_use())
        theme_selector = ttk.Combobox(
            top_frame,
            textvariable=self.theme_var,
            values=sorted(self.style.theme_names()),
            state="readonly",
            width=15,
        )
        theme_selector.pack(side="left", padx=5)
        theme_selector.bind("<<ComboboxSelected>>", self.change_theme)

        ttk.Separator(top_frame, orient="vertical").pack(side="left", padx=10, fill="y")

        self._t(ttk.Label, top_frame, "language_label").pack(side="left", padx=5)
        self.lang_var = tk.StringVar(value=SUPPORTED_LANGUAGES[self.current_lang])
        lang_selector = ttk.Combobox(
            top_frame,
            textvariable=self.lang_var,
            values=list(SUPPORTED_LANGUAGES.values()),
            state="readonly",
            width=10,
        )
        lang_selector.pack(side="left", padx=5)
        lang_selector.bind("<<ComboboxSelected>>", self.change_language)

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.create_generate_tab()
        self.create_concatenate_tab()
        self.create_compare_tab()
        self.create_resize_tab()

        # Status bar
        self.status_var = tk.StringVar(value=self.tr("status_ready"))
        status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def tr(self, key):
        """Get translated string for current language"""
        return TRANSLATIONS.get(self.current_lang, TRANSLATIONS["en"]).get(key, key)

    def _t(self, widget_class, parent, key, **kwargs):
        """Create a widget with translated text and register for language updates"""
        w = widget_class(parent, text=self.tr(key), **kwargs)
        self._i18n_map.append((w, key))
        return w

    def create_generate_tab(self):
        """Tab 1: Generate Log-to-Log LUT"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.tr("tab_generate"))

        # Main frame with padding
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Mode selection
        mode_frame = self._t(ttk.LabelFrame, main_frame, "mode", padding="5")
        mode_frame.pack(fill="x", pady=5)

        self.gen_mode = tk.StringVar(value="single")
        self._t(
            ttk.Radiobutton,
            mode_frame,
            "single_conversion",
            variable=self.gen_mode,
            value="single",
        ).pack(side="left", padx=10)
        self._t(
            ttk.Radiobutton,
            mode_frame,
            "batch_conversion",
            variable=self.gen_mode,
            value="batch",
        ).pack(side="left", padx=10)

        # Source log selection
        source_frame = self._t(
            ttk.LabelFrame, main_frame, "source_log_format", padding="5"
        )
        source_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, source_frame, "source").pack(side="left", padx=5)
        self.gen_source = ttk.Combobox(
            source_frame, values=list(LOG_CONFIGS.keys()), width=30, state="readonly"
        )
        self.gen_source.pack(side="left", padx=5, fill="x", expand=True)
        self.gen_source.current(0)

        # Target log selection (for single mode)
        target_frame = self._t(
            ttk.LabelFrame, main_frame, "target_log_format", padding="5"
        )
        target_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, target_frame, "target").pack(side="left", padx=5)
        self.gen_target = ttk.Combobox(
            target_frame, values=list(LOG_CONFIGS.keys()), width=30, state="readonly"
        )
        self.gen_target.pack(side="left", padx=5, fill="x", expand=True)
        self.gen_target.current(1)

        # LUT size
        size_frame = self._t(ttk.LabelFrame, main_frame, "lut_size", padding="5")
        size_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, size_frame, "grid_size").pack(side="left", padx=5)
        self.gen_size = ttk.Combobox(
            size_frame, values=[17, 33, 65, 129], width=10, state="readonly"
        )
        self.gen_size.pack(side="left", padx=5)
        self.gen_size.set(65)

        # Output settings
        output_frame = self._t(ttk.LabelFrame, main_frame, "output", padding="5")
        output_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, output_frame, "output_directory").pack(side="left", padx=5)
        self.gen_output_dir = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.gen_output_dir, width=40).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button, output_frame, "browse", command=self.browse_gen_output
        ).pack(side="left", padx=5)

        # Generate button
        self._t(
            ttk.Button,
            main_frame,
            "btn_generate_lut",
            command=self.generate_lut,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = self._t(ttk.LabelFrame, main_frame, "output_log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.gen_console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.gen_console.pack(fill="both", expand=True)

    def create_concatenate_tab(self):
        """Tab 2: Concatenate LUTs"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.tr("tab_concatenate"))

        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Input 1
        input1_frame = self._t(
            ttk.LabelFrame, main_frame, "first_input_applied_first", padding="5"
        )
        input1_frame.pack(fill="x", pady=5)

        self.concat_input1_type = tk.StringVar(value="file")
        self._t(
            ttk.Radiobutton,
            input1_frame,
            "file",
            variable=self.concat_input1_type,
            value="file",
        ).pack(side="left", padx=5)
        self._t(
            ttk.Radiobutton,
            input1_frame,
            "directory",
            variable=self.concat_input1_type,
            value="dir",
        ).pack(side="left", padx=5)

        self.concat_input1 = tk.StringVar()
        ttk.Entry(input1_frame, textvariable=self.concat_input1, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button,
            input1_frame,
            "browse",
            command=lambda: self.browse_concat_input(1),
        ).pack(side="left", padx=5)

        # Input 2
        input2_frame = self._t(
            ttk.LabelFrame, main_frame, "second_input_applied_second", padding="5"
        )
        input2_frame.pack(fill="x", pady=5)

        self.concat_input2_type = tk.StringVar(value="file")
        self._t(
            ttk.Radiobutton,
            input2_frame,
            "file",
            variable=self.concat_input2_type,
            value="file",
        ).pack(side="left", padx=5)
        self._t(
            ttk.Radiobutton,
            input2_frame,
            "directory",
            variable=self.concat_input2_type,
            value="dir",
        ).pack(side="left", padx=5)

        self.concat_input2 = tk.StringVar()
        ttk.Entry(input2_frame, textvariable=self.concat_input2, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button,
            input2_frame,
            "browse",
            command=lambda: self.browse_concat_input(2),
        ).pack(side="left", padx=5)

        # Output
        output_frame = self._t(ttk.LabelFrame, main_frame, "output", padding="5")
        output_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, output_frame, "output_path").pack(side="left", padx=5)
        self.concat_output = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.concat_output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button, output_frame, "browse", command=self.browse_concat_output
        ).pack(side="left", padx=5)

        # Workers
        workers_frame = ttk.Frame(main_frame)
        workers_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, workers_frame, "parallel_workers").pack(side="left", padx=5)
        self.concat_workers = tk.IntVar(value=4)
        ttk.Spinbox(
            workers_frame, from_=1, to=16, textvariable=self.concat_workers, width=10
        ).pack(side="left", padx=5)

        # Concatenate button
        self._t(
            ttk.Button,
            main_frame,
            "btn_concatenate_luts",
            command=self.concatenate_luts,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = self._t(ttk.LabelFrame, main_frame, "output_log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.concat_console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.concat_console.pack(fill="both", expand=True)

        # Results
        results_frame = self._t(ttk.LabelFrame, main_frame, "results", padding="5")
        results_frame.pack(fill="both", expand=True, pady=5)

        columns = ("name", "status", "clipped", "clip_ratio", "output")
        self.concat_results_tree = ttk.Treeview(
            results_frame, columns=columns, show="headings", height=8
        )
        self.concat_results_tree.pack(fill="both", expand=True)

        self.concat_results_tree.heading("name", text=self.tr("col_name"))
        self.concat_results_tree.heading("status", text=self.tr("col_status"))
        self.concat_results_tree.heading("clipped", text=self.tr("col_clipped"))
        self.concat_results_tree.heading("clip_ratio", text=self.tr("col_clip_ratio"))
        self.concat_results_tree.heading("output", text=self.tr("col_output"))

        self.concat_results_tree.column("name", width=200)
        self.concat_results_tree.column("status", width=80, anchor="center")
        self.concat_results_tree.column("clipped", width=80, anchor="center")
        self.concat_results_tree.column("clip_ratio", width=80, anchor="center")
        self.concat_results_tree.column("output", width=400)

    def create_compare_tab(self):
        """Tab 3: Compare Images"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.tr("tab_compare"))

        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Mode selection
        mode_frame = self._t(ttk.LabelFrame, main_frame, "mode", padding="5")
        mode_frame.pack(fill="x", pady=5)

        self.compare_mode = tk.StringVar(value="single")
        self._t(
            ttk.Radiobutton,
            mode_frame,
            "single_image_pair",
            variable=self.compare_mode,
            value="single",
        ).pack(side="left", padx=10)
        self._t(
            ttk.Radiobutton,
            mode_frame,
            "directory_comparison",
            variable=self.compare_mode,
            value="batch",
        ).pack(side="left", padx=10)

        # Image 1 / Directory 1
        input1_frame = self._t(ttk.LabelFrame, main_frame, "first_input", padding="5")
        input1_frame.pack(fill="x", pady=5)

        self.compare_input1 = tk.StringVar()
        ttk.Entry(input1_frame, textvariable=self.compare_input1, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button,
            input1_frame,
            "browse",
            command=lambda: self.browse_compare_input(1),
        ).pack(side="left", padx=5)

        # Image 2 / Directory 2
        input2_frame = self._t(ttk.LabelFrame, main_frame, "second_input", padding="5")
        input2_frame.pack(fill="x", pady=5)

        self.compare_input2 = tk.StringVar()
        ttk.Entry(input2_frame, textvariable=self.compare_input2, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button,
            input2_frame,
            "browse",
            command=lambda: self.browse_compare_input(2),
        ).pack(side="left", padx=5)

        # Options
        options_frame = self._t(ttk.LabelFrame, main_frame, "options", padding="5")
        options_frame.pack(fill="x", pady=5)

        self.compare_visualize = tk.BooleanVar(value=True)
        self._t(
            ttk.Checkbutton,
            options_frame,
            "generate_visualization",
            variable=self.compare_visualize,
        ).pack(side="left", padx=10)

        self._t(ttk.Label, options_frame, "amplification").pack(side="left", padx=5)
        self.compare_amplification = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            options_frame,
            from_=0.1,
            to=100.0,
            increment=0.5,
            textvariable=self.compare_amplification,
            width=10,
        ).pack(side="left", padx=5)

        self._t(ttk.Label, options_frame, "workers").pack(side="left", padx=5)
        self.compare_workers = tk.IntVar(value=4)
        ttk.Spinbox(
            options_frame,
            from_=1,
            to=16,
            textvariable=self.compare_workers,
            width=10,
        ).pack(side="left", padx=5)

        # Output
        output_frame = self._t(ttk.LabelFrame, main_frame, "output", padding="5")
        output_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, output_frame, "output_path").pack(side="left", padx=5)
        self.compare_output = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.compare_output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button, output_frame, "browse", command=self.browse_compare_output
        ).pack(side="left", padx=5)

        # Compare button
        self._t(
            ttk.Button,
            main_frame,
            "btn_compare_images",
            command=self.compare_images,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = self._t(ttk.LabelFrame, main_frame, "output_log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.compare_console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.compare_console.pack(fill="both", expand=True)

    def create_resize_tab(self):
        """Tab 4: Resize LUT"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=self.tr("tab_resize"))

        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Input file
        input_frame = self._t(ttk.LabelFrame, main_frame, "input_lut", padding="5")
        input_frame.pack(fill="x", pady=5)

        self.resize_input = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.resize_input, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button, input_frame, "browse", command=self.browse_resize_input
        ).pack(side="left", padx=5)

        # Target size
        size_frame = self._t(ttk.LabelFrame, main_frame, "target_size", padding="5")
        size_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, size_frame, "new_grid_size").pack(side="left", padx=5)
        self.resize_size = ttk.Combobox(
            size_frame, values=[17, 33, 65, 129], width=10, state="readonly"
        )
        self.resize_size.pack(side="left", padx=5)
        self.resize_size.set(33)

        # Output file
        output_frame = self._t(ttk.LabelFrame, main_frame, "output", padding="5")
        output_frame.pack(fill="x", pady=5)

        self._t(ttk.Label, output_frame, "output_file").pack(side="left", padx=5)
        self.resize_output = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.resize_output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        self._t(
            ttk.Button, output_frame, "browse", command=self.browse_resize_output
        ).pack(side="left", padx=5)

        self._t(ttk.Label, output_frame, "auto_generate_hint", foreground="gray").pack(
            side="left", padx=5
        )

        # Resize button
        self._t(
            ttk.Button,
            main_frame,
            "btn_resize_lut",
            command=self.resize_lut_action,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = self._t(ttk.LabelFrame, main_frame, "output_log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.resize_console = scrolledtext.ScrolledText(
            console_frame, height=20, state="disabled", bg="white", fg="black"
        )
        self.resize_console.pack(fill="both", expand=True)

    # Browse methods
    def browse_gen_output(self):
        directory = filedialog.askdirectory(initialdir=self.gen_output_dir.get())
        if directory:
            self.gen_output_dir.set(directory)

    def browse_concat_input(self, input_num):
        if input_num == 1:
            type_var = self.concat_input1_type
            path_var = self.concat_input1
        else:
            type_var = self.concat_input2_type
            path_var = self.concat_input2

        if type_var.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("CUBE files", "*.cube")])
        else:
            path = filedialog.askdirectory()

        if path:
            path_var.set(path)

    def browse_concat_output(self):
        # Check if any input is a directory
        is_batch = (
            self.concat_input1_type.get() == "dir"
            or self.concat_input2_type.get() == "dir"
        )

        if is_batch:
            path = filedialog.askdirectory()
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".cube", filetypes=[("CUBE files", "*.cube")]
            )

        if path:
            self.concat_output.set(path)

    def browse_compare_input(self, input_num):
        path_var = self.compare_input1 if input_num == 1 else self.compare_input2

        if self.compare_mode.get() == "single":
            path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.tif *.tiff *.png *.jpg")]
            )
        else:
            path = filedialog.askdirectory()

        if path:
            path_var.set(path)

    def browse_compare_output(self):
        if self.compare_mode.get() == "single":
            path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG files", "*.png")]
            )
        else:
            path = filedialog.askdirectory()

        if path:
            self.compare_output.set(path)

    def browse_resize_input(self):
        path = filedialog.askopenfilename(filetypes=[("CUBE files", "*.cube")])
        if path:
            self.resize_input.set(path)

    def browse_resize_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".cube", filetypes=[("CUBE files", "*.cube")]
        )
        if path:
            self.resize_output.set(path)

    # Action methods
    def run_in_thread(self, func, console_widget):
        """Run a function in a separate thread and redirect output to console"""

        def wrapper():
            # Redirect stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            sys.stdout = RedirectText(console_widget, "stdout")
            sys.stderr = RedirectText(console_widget, "stderr")

            try:
                func()
                self.status_var.set(self.tr("status_completed"))
            except Exception as e:
                print(f"\nError: {e}")
                self.status_var.set(self.tr("status_error").format(e))
                messagebox.showerror(self.tr("error_title"), str(e))
            finally:
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # Clear console
        console_widget.configure(state="normal")
        console_widget.delete(1.0, tk.END)
        console_widget.configure(state="disabled")

        self.status_var.set(self.tr("status_processing"))
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    def update_concat_results(self, results):
        tree = self.concat_results_tree

        # Clear old rows
        for row in tree.get_children():
            tree.delete(row)

        for item in results:
            clipped = item.get("clipped", False)
            status = item.get("status", "unknown")

            tag = "error" if status != "ok" else ("clipped" if clipped else "ok")

            tree.insert(
                "",
                tk.END,
                values=(
                    item.get("name", ""),
                    status.upper(),
                    "YES" if clipped else "NO",
                    f"{item.get('clip_ratio', 0.0) * 100:.2f}%",
                    item.get("output", ""),
                ),
                tags=(tag,),
            )

        # Row colors
        tree.tag_configure("ok", background="#e8f5e9")
        tree.tag_configure("clipped", background="#fff8e1")
        tree.tag_configure("error", background="#ffebee")

    def generate_lut(self):
        def task():
            source = self.gen_source.get()
            size = int(self.gen_size.get())
            output_dir = os.path.abspath(self.gen_output_dir.get())

            if self.gen_mode.get() == "single":
                target = self.gen_target.get()
                if source == target:
                    raise ValueError(self.tr("error_same_source_target"))

                # Generate filename and place it in output_dir
                source_name = source.replace(" ", "_").replace(".", "")
                target_name = target.replace(" ", "_").replace(".", "")
                out_filename = f"{source_name}_to_{target_name}_{size}.cube"
                out_path = os.path.join(output_dir, out_filename)

                generate_log_to_log_lut(
                    source_log=source,
                    target_log=target,
                    lut_size=size,
                    out_path=out_path,
                )
            else:  # batch mode
                generate_multiple_luts(
                    source_log=source, target_logs=None, lut_size=size, output_dir=output_dir
                )

        self.run_in_thread(task, self.gen_console)

    def concatenate_luts(self):
        def task():
            input1 = os.path.abspath(self.concat_input1.get())
            input2 = os.path.abspath(self.concat_input2.get())
            output = os.path.abspath(self.concat_output.get())
            workers = self.concat_workers.get()

            if not input1 or not input2:
                raise ValueError(self.tr("error_specify_both_inputs"))
            if not output:
                raise ValueError(self.tr("error_specify_output"))

            results = process_luts(input1, input2, output, max_workers=workers)
            self.root.after(0, lambda: self.update_concat_results(results))

        self.run_in_thread(task, self.concat_console)

    def compare_images(self):
        def task():
            input1 = os.path.abspath(self.compare_input1.get())
            input2 = os.path.abspath(self.compare_input2.get())
            output = (
                os.path.abspath(self.compare_output.get())
                if self.compare_output.get()
                else None
            )
            visualize = self.compare_visualize.get()
            amplification = self.compare_amplification.get()
            workers = self.compare_workers.get()

            if not input1 or not input2:
                raise ValueError(self.tr("error_specify_both_inputs"))

            if self.compare_mode.get() == "single":
                compare_px_diff(
                    input1,
                    input2,
                    visualize=visualize,
                    output_path=output,
                    amplification=amplification,
                )
            else:  # batch mode
                compare_image_dirs(
                    input1,
                    input2,
                    visualize=visualize,
                    output_dir=output,
                    amplification=amplification,
                    workers=workers,
                )

        self.run_in_thread(task, self.compare_console)

    def resize_lut_action(self):
        def task():
            input_path = os.path.abspath(self.resize_input.get())
            output_path = self.resize_output.get()
            target_size = int(self.resize_size.get())

            if not input_path:
                raise ValueError(self.tr("error_specify_input"))

            # Auto-generate output filename if not specified
            if not output_path:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}_{target_size}{ext}"
            else:
                output_path = os.path.abspath(output_path)

            resize_lut(input_path, output_path, target_size)

        self.run_in_thread(task, self.resize_console)

    def _set_window_icon(self):
        """Set the window icon for both development and PyInstaller"""
        try:
            if platform.system() == "Windows":
                # Windows uses .ico files
                icon_path = resource_path(os.path.join("static", "logo.ico"))
                if os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
            else:
                # Linux/Mac: Try to use PIL to load .ico as PhotoImage
                try:
                    from PIL import Image, ImageTk

                    icon_path = resource_path(os.path.join("static", "logo.ico"))
                    if os.path.exists(icon_path):
                        img = Image.open(icon_path)
                        photo = ImageTk.PhotoImage(img)
                        self.root.iconphoto(True, photo)
                        # Keep a reference to prevent garbage collection
                        self.root._icon_photo = photo
                except ImportError:
                    print("Warning: PIL not available for icon on Linux/Mac")
                except Exception as e:
                    print(f"Warning: Could not set window icon with PIL: {e}")
        except Exception as e:
            print(f"Warning: Could not set window icon: {e}")

    def change_theme(self, event=None):
        """Change the application theme"""
        selected_theme = self.theme_var.get()
        try:
            self.style.theme_use(selected_theme)
            self.status_var.set(self.tr("theme_changed").format(selected_theme))
        except Exception as e:
            self.status_var.set(self.tr("status_error").format(e))
            messagebox.showerror(
                self.tr("theme_error_title"),
                self.tr("theme_error_msg").format(selected_theme, e),
            )

    def change_language(self, event=None):
        """Change the application language"""
        selected_name = self.lang_var.get()
        for code, name in SUPPORTED_LANGUAGES.items():
            if name == selected_name:
                self.current_lang = code
                break
        self._refresh_language()

    def _refresh_language(self):
        """Update all UI text to current language"""
        self.root.title(self.tr("window_title"))

        # Update all registered widgets
        for widget, key in self._i18n_map:
            widget.configure(text=self.tr(key))

        # Update notebook tabs
        tab_keys = ["tab_generate", "tab_concatenate", "tab_compare", "tab_resize"]
        for i, key in enumerate(tab_keys):
            self.notebook.tab(i, text=self.tr(key))

        # Update treeview headings
        heading_map = {
            "name": "col_name",
            "status": "col_status",
            "clipped": "col_clipped",
            "clip_ratio": "col_clip_ratio",
            "output": "col_output",
        }
        for col, key in heading_map.items():
            self.concat_results_tree.heading(col, text=self.tr(key))

        # Update status bar
        self.status_var.set(self.tr("status_ready"))


def main():
    root = tk.Tk()
    app = LUTWorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
