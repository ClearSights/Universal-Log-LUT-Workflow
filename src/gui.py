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
        self.root.title("Universal Log LUT Workflow")
        self.root.geometry("900x700")

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

        # Theme selection frame at the top
        theme_frame = ttk.Frame(root)
        theme_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(theme_frame, text="Theme:").pack(side="left", padx=5)
        self.theme_var = tk.StringVar(value=self.style.theme_use())
        theme_selector = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=sorted(self.style.theme_names()),
            state="readonly",
            width=15,
        )
        theme_selector.pack(side="left", padx=5)
        theme_selector.bind("<<ComboboxSelected>>", self.change_theme)

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.create_generate_tab()
        self.create_concatenate_tab()
        self.create_compare_tab()
        self.create_resize_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_generate_tab(self):
        """Tab 1: Generate Log-to-Log LUT"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Generate LUT")

        # Main frame with padding
        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Mode", padding="5")
        mode_frame.pack(fill="x", pady=5)

        self.gen_mode = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_frame, text="Single Conversion", variable=self.gen_mode, value="single"
        ).pack(side="left", padx=10)
        ttk.Radiobutton(
            mode_frame, text="Batch Conversion", variable=self.gen_mode, value="batch"
        ).pack(side="left", padx=10)

        # Source log selection
        source_frame = ttk.LabelFrame(main_frame, text="Source Log Format", padding="5")
        source_frame.pack(fill="x", pady=5)

        ttk.Label(source_frame, text="Source:").pack(side="left", padx=5)
        self.gen_source = ttk.Combobox(
            source_frame, values=list(LOG_CONFIGS.keys()), width=30, state="readonly"
        )
        self.gen_source.pack(side="left", padx=5, fill="x", expand=True)
        self.gen_source.current(0)

        # Target log selection (for single mode)
        target_frame = ttk.LabelFrame(main_frame, text="Target Log Format", padding="5")
        target_frame.pack(fill="x", pady=5)

        ttk.Label(target_frame, text="Target:").pack(side="left", padx=5)
        self.gen_target = ttk.Combobox(
            target_frame, values=list(LOG_CONFIGS.keys()), width=30, state="readonly"
        )
        self.gen_target.pack(side="left", padx=5, fill="x", expand=True)
        self.gen_target.current(1)

        # LUT size
        size_frame = ttk.LabelFrame(main_frame, text="LUT Size", padding="5")
        size_frame.pack(fill="x", pady=5)

        ttk.Label(size_frame, text="Grid Size:").pack(side="left", padx=5)
        self.gen_size = ttk.Combobox(
            size_frame, values=[17, 33, 65, 129], width=10, state="readonly"
        )
        self.gen_size.pack(side="left", padx=5)
        self.gen_size.set(65)

        # Output settings
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="Output Directory:").pack(side="left", padx=5)
        self.gen_output_dir = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.gen_output_dir, width=40).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            output_frame, text="Browse", command=self.browse_gen_output
        ).pack(side="left", padx=5)

        # Generate button
        ttk.Button(
            main_frame,
            text="Generate LUT",
            command=self.generate_lut,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.gen_console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.gen_console.pack(fill="both", expand=True)

    def create_concatenate_tab(self):
        """Tab 2: Concatenate LUTs"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Concatenate LUTs")

        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Input 1
        input1_frame = ttk.LabelFrame(
            main_frame, text="First Input (Applied First)", padding="5"
        )
        input1_frame.pack(fill="x", pady=5)

        self.concat_input1_type = tk.StringVar(value="file")
        ttk.Radiobutton(
            input1_frame, text="File", variable=self.concat_input1_type, value="file"
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            input1_frame,
            text="Directory",
            variable=self.concat_input1_type,
            value="dir",
        ).pack(side="left", padx=5)

        self.concat_input1 = tk.StringVar()
        ttk.Entry(input1_frame, textvariable=self.concat_input1, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            input1_frame, text="Browse", command=lambda: self.browse_concat_input(1)
        ).pack(side="left", padx=5)

        # Input 2
        input2_frame = ttk.LabelFrame(
            main_frame, text="Second Input (Applied Second)", padding="5"
        )
        input2_frame.pack(fill="x", pady=5)

        self.concat_input2_type = tk.StringVar(value="file")
        ttk.Radiobutton(
            input2_frame, text="File", variable=self.concat_input2_type, value="file"
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            input2_frame,
            text="Directory",
            variable=self.concat_input2_type,
            value="dir",
        ).pack(side="left", padx=5)

        self.concat_input2 = tk.StringVar()
        ttk.Entry(input2_frame, textvariable=self.concat_input2, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            input2_frame, text="Browse", command=lambda: self.browse_concat_input(2)
        ).pack(side="left", padx=5)

        # Output
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="Output Path:").pack(side="left", padx=5)
        self.concat_output = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.concat_output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            output_frame, text="Browse", command=self.browse_concat_output
        ).pack(side="left", padx=5)

        # Workers
        workers_frame = ttk.Frame(main_frame)
        workers_frame.pack(fill="x", pady=5)

        ttk.Label(workers_frame, text="Parallel Workers:").pack(side="left", padx=5)
        self.concat_workers = tk.IntVar(value=4)
        ttk.Spinbox(
            workers_frame, from_=1, to=16, textvariable=self.concat_workers, width=10
        ).pack(side="left", padx=5)

        # Concatenate button
        ttk.Button(
            main_frame,
            text="Concatenate LUTs",
            command=self.concatenate_luts,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.concat_console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.concat_console.pack(fill="both", expand=True)

    def create_compare_tab(self):
        """Tab 3: Compare Images"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Compare Images")

        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Mode", padding="5")
        mode_frame.pack(fill="x", pady=5)

        self.compare_mode = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_frame,
            text="Single Image Pair",
            variable=self.compare_mode,
            value="single",
        ).pack(side="left", padx=10)
        ttk.Radiobutton(
            mode_frame,
            text="Directory Comparison",
            variable=self.compare_mode,
            value="batch",
        ).pack(side="left", padx=10)

        # Image 1 / Directory 1
        input1_frame = ttk.LabelFrame(main_frame, text="First Input", padding="5")
        input1_frame.pack(fill="x", pady=5)

        self.compare_input1 = tk.StringVar()
        ttk.Entry(input1_frame, textvariable=self.compare_input1, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            input1_frame, text="Browse", command=lambda: self.browse_compare_input(1)
        ).pack(side="left", padx=5)

        # Image 2 / Directory 2
        input2_frame = ttk.LabelFrame(main_frame, text="Second Input", padding="5")
        input2_frame.pack(fill="x", pady=5)

        self.compare_input2 = tk.StringVar()
        ttk.Entry(input2_frame, textvariable=self.compare_input2, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            input2_frame, text="Browse", command=lambda: self.browse_compare_input(2)
        ).pack(side="left", padx=5)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.pack(fill="x", pady=5)

        self.compare_visualize = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, text="Generate Visualization", variable=self.compare_visualize
        ).pack(side="left", padx=10)

        ttk.Label(options_frame, text="Amplification:").pack(side="left", padx=5)
        self.compare_amplification = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            options_frame,
            from_=0.1,
            to=100.0,
            increment=0.5,
            textvariable=self.compare_amplification,
            width=10,
        ).pack(side="left", padx=5)

        ttk.Label(options_frame, text="Workers:").pack(side="left", padx=5)
        self.compare_workers = tk.IntVar(value=4)
        ttk.Spinbox(
            options_frame,
            from_=1,
            to=16,
            textvariable=self.compare_workers,
            width=10,
        ).pack(side="left", padx=5)

        # Output
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="Output Path:").pack(side="left", padx=5)
        self.compare_output = tk.StringVar(value=os.getcwd())
        ttk.Entry(output_frame, textvariable=self.compare_output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            output_frame, text="Browse", command=self.browse_compare_output
        ).pack(side="left", padx=5)

        # Compare button
        ttk.Button(
            main_frame,
            text="Compare Images",
            command=self.compare_images,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="5")
        console_frame.pack(fill="both", expand=True, pady=5)

        self.compare_console = scrolledtext.ScrolledText(
            console_frame, height=15, state="disabled", bg="white", fg="black"
        )
        self.compare_console.pack(fill="both", expand=True)

    def create_resize_tab(self):
        """Tab 4: Resize LUT"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Resize LUT")

        main_frame = ttk.Frame(tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Input file
        input_frame = ttk.LabelFrame(main_frame, text="Input LUT", padding="5")
        input_frame.pack(fill="x", pady=5)

        self.resize_input = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.resize_input, width=60).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_resize_input).pack(
            side="left", padx=5
        )

        # Target size
        size_frame = ttk.LabelFrame(main_frame, text="Target Size", padding="5")
        size_frame.pack(fill="x", pady=5)

        ttk.Label(size_frame, text="New Grid Size:").pack(side="left", padx=5)
        self.resize_size = ttk.Combobox(
            size_frame, values=[17, 33, 65, 129], width=10, state="readonly"
        )
        self.resize_size.pack(side="left", padx=5)
        self.resize_size.set(33)

        # Output file
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="Output File:").pack(side="left", padx=5)
        self.resize_output = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.resize_output, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(
            output_frame, text="Browse", command=self.browse_resize_output
        ).pack(side="left", padx=5)

        ttk.Label(
            output_frame, text="(Leave empty to auto-generate)", foreground="gray"
        ).pack(side="left", padx=5)

        # Resize button
        ttk.Button(
            main_frame,
            text="Resize LUT",
            command=self.resize_lut_action,
            style="Accent.TButton",
        ).pack(pady=10)

        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="5")
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
                self.status_var.set("Operation completed successfully")
            except Exception as e:
                print(f"\nError: {e}")
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # Clear console
        console_widget.configure(state="normal")
        console_widget.delete(1.0, tk.END)
        console_widget.configure(state="disabled")

        self.status_var.set("Processing...")
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    def generate_lut(self):
        def task():
            source = self.gen_source.get()
            size = int(self.gen_size.get())
            output_dir = os.path.abspath(self.gen_output_dir.get())

            if self.gen_mode.get() == "single":
                target = self.gen_target.get()
                if source == target:
                    raise ValueError("Source and target cannot be the same")

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
                raise ValueError("Please specify both inputs")
            if not output:
                raise ValueError("Please specify output path")

            process_luts(input1, input2, output, max_workers=workers)

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
                raise ValueError("Please specify both inputs")

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
                raise ValueError("Please specify input file")

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
            self.status_var.set(f"Theme changed to: {selected_theme}")
        except Exception as e:
            self.status_var.set(f"Failed to change theme: {e}")
            messagebox.showerror(
                "Theme Error", f"Could not apply theme '{selected_theme}': {e}"
            )


def main():
    root = tk.Tk()
    app = LUTWorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
