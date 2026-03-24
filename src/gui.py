"""
GUI for Universal Log LUT Workflow
Integrates all LUT processing tools in a user-friendly interface
"""

import os
import sys
import platform
import tkinter as tk
from tkinter import ttk, messagebox

from i18n import TRANSLATIONS, SUPPORTED_LANGUAGES, detect_language
from gui_utils import resource_path
from tabs import GenerateTab, ConcatenateTab, CompareTab, ResizeTab


class LUTWorkflowGUI:
    def __init__(self, root):
        self.root = root
        self.current_lang = detect_language()
        self._i18n_map = []
        self.root.title(self.tr("window_title"))

        # Size window to 80% of screen and center it
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(max(int(screen_w * 0.55), 1024), 1400)
        win_h = min(max(int(screen_h * 0.75), 768), 1050)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

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

        # Add padding to tab headers and widgets for better spacing
        self.style.configure("TNotebook.Tab", padding=[16, 8])
        self.style.configure("TLabelframe", padding=10)
        self.style.configure("TLabelframe.Label", padding=[4, 2])

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
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # Status bar (must exist before tabs, since tabs reference status_var)
        self.status_var = tk.StringVar(value=self.tr("status_ready"))
        status_bar = ttk.Label(
            root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create tabs
        self.tabs = [
            GenerateTab(self.notebook, self),
            ConcatenateTab(self.notebook, self),
            CompareTab(self.notebook, self),
            ResizeTab(self.notebook, self),
        ]

    def tr(self, key):
        """Get translated string for current language"""
        return TRANSLATIONS.get(self.current_lang, TRANSLATIONS["en"]).get(key, key)

    def _t(self, widget_class, parent, key, **kwargs):
        """Create a widget with translated text and register for language updates"""
        w = widget_class(parent, text=self.tr(key), **kwargs)
        self._i18n_map.append((w, key))
        return w

    def _set_window_icon(self):
        """Set the window icon for both development and PyInstaller"""
        try:
            if platform.system() == "Windows":
                icon_path = resource_path(os.path.join("static", "logo.ico"))
                if os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
            else:
                try:
                    from PIL import Image, ImageTk

                    icon_path = resource_path(os.path.join("static", "logo.ico"))
                    if os.path.exists(icon_path):
                        img = Image.open(icon_path)
                        photo = ImageTk.PhotoImage(img)
                        self.root.iconphoto(True, photo)
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

        # Delegate tab-specific updates
        for tab in self.tabs:
            if hasattr(tab, "refresh_language"):
                tab.refresh_language(self.tr)

        # Update status bar
        self.status_var.set(self.tr("status_ready"))


def main():
    # Enable DPI awareness for sharp rendering on high-DPI displays
    if platform.system() == "Windows":
        import ctypes

        try:
            # Per-monitor DPI aware (Windows 8.1+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # System DPI aware (fallback)
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    root = tk.Tk()

    # Scale UI elements based on actual screen DPI
    try:
        dpi = root.winfo_fpixels("1i")
        scale_factor = dpi / 72.0
        root.tk.call("tk", "scaling", scale_factor)
    except Exception:
        pass

    app = LUTWorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
