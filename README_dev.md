# Developer Guide

## Environment Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python src/gui.py
   ```

## Code Structure

```
src/
├── gui.py                    # Main entry point – Tkinter application window, tab loading, theming
├── gui_utils.py              # GUI helpers: stdout redirection to text widget, resource path
│                             #   resolution (dev vs packaged), background thread runner
├── i18n.py                   # Internationalization – English/Chinese translations and auto-detection
│
├── generate_log2log_lut.py   # Core: generates "bridge" 3D LUTs between log color spaces
├── concatenate_luts.py       # Core: composes two LUTs (LUT2 ∘ LUT1) with batch & multiprocessing
├── compare_images.py         # Core: per-pixel ΔE comparison of 16-bit TIFFs with heatmap output
├── resize_lut.py             # Core: resamples .cube LUT grids via scipy interpolation
│
└── tabs/
    ├── __init__.py           # Re-exports all tab classes
    ├── generate_tab.py       # UI tab for generate_log2log_lut (single & batch modes)
    ├── concatenate_tab.py    # UI tab for concatenate_luts (file/directory inputs, worker config)
    ├── compare_tab.py        # UI tab for compare_images (single & batch modes, amplification)
    └── resize_tab.py         # UI tab for resize_lut (target grid size selection)

static/
├── logo.ico                  # Application icon
└── *.png                     # Screenshots used in README documentation

gui.spec                      # PyInstaller spec file for building a standalone executable
```

Each tab in `src/tabs/` is a thin UI wrapper around the corresponding core module in `src/`. The core modules are self-contained and can also be used independently from the command line or as library imports.

## Building the Executable

Install PyInstaller if you haven't already:

```bash
pip install pyinstaller
```

Build from the project root:

```bash
pyinstaller gui.spec
```

The standalone executable `UniversalLogLUT` will be output to the `dist/` directory. The spec file bundles `static/logo.ico`, sets `console=False` (no terminal window), and includes `PIL` as a hidden import.
