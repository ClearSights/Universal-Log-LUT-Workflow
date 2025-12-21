# given two 16-bit tiff images, compute and print per-pixel absolute and relative differences
import argparse
import numpy as np
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from PIL import Image
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt


try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


@jit(nopython=True, parallel=True, cache=True)
def fast_rgb_to_lab(rgb_norm):
    """
    Fast RGB to LAB conversion optimized with numba.
    Implements simplified sRGB to CIELAB conversion.

    Args:
        rgb_norm: Normalized RGB array (0-1 range), shape (H, W, 3)

    Returns:
        LAB array, shape (H, W, 3)
    """
    h, w = rgb_norm.shape[0], rgb_norm.shape[1]
    lab = np.empty_like(rgb_norm)

    for i in prange(h):
        for j in range(w):
            # sRGB to linear RGB
            r, g, b = rgb_norm[i, j, 0], rgb_norm[i, j, 1], rgb_norm[i, j, 2]

            # Gamma correction
            if r > 0.04045:
                r = ((r + 0.055) / 1.055) ** 2.4
            else:
                r = r / 12.92
            if g > 0.04045:
                g = ((g + 0.055) / 1.055) ** 2.4
            else:
                g = g / 12.92
            if b > 0.04045:
                b = ((b + 0.055) / 1.055) ** 2.4
            else:
                b = b / 12.92

            # Linear RGB to XYZ (D65 illuminant)
            x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
            y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
            z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

            # XYZ to LAB
            x = x / 0.95047
            y = y / 1.00000
            z = z / 1.08883

            if x > 0.008856:
                x = x ** (1.0 / 3.0)
            else:
                x = (7.787 * x) + (16.0 / 116.0)
            if y > 0.008856:
                y = y ** (1.0 / 3.0)
            else:
                y = (7.787 * y) + (16.0 / 116.0)
            if z > 0.008856:
                z = z ** (1.0 / 3.0)
            else:
                z = (7.787 * z) + (16.0 / 116.0)

            lab[i, j, 0] = (116.0 * y) - 16.0  # L
            lab[i, j, 1] = 500.0 * (x - y)  # a
            lab[i, j, 2] = 200.0 * (y - z)  # b

    return lab


@jit(nopython=True, parallel=True, cache=True)
def fast_delta_e(lab1, lab2):
    """
    Fast Delta E calculation optimized with numba.

    Args:
        lab1: First LAB image array
        lab2: Second LAB image array

    Returns:
        Delta E array
    """
    h, w = lab1.shape[0], lab1.shape[1]
    delta_e = np.empty((h, w), dtype=np.float32)

    for i in prange(h):
        for j in range(w):
            dl = lab1[i, j, 0] - lab2[i, j, 0]
            da = lab1[i, j, 1] - lab2[i, j, 1]
            db = lab1[i, j, 2] - lab2[i, j, 2]
            delta_e[i, j] = np.sqrt(dl * dl + da * da + db * db)

    return delta_e


def analyze_perceptual_difference(arr1, arr2, abs_diff, bit_depth=16):
    """
    Analyze perceptual significance of differences.

    Args:
        arr1: First image array
        arr2: Second image array
        abs_diff: Absolute difference array
        bit_depth: Bit depth of input images (default 16)

    Returns:
        Dictionary with perceptual metrics
    """
    max_value = 2**bit_depth - 1

    # Calculate per-pixel difference magnitude
    if len(abs_diff.shape) == 3:
        diff_magnitude = np.mean(abs_diff, axis=2)
    else:
        diff_magnitude = abs_diff

    # Convert to 8-bit equivalent for reference
    diff_8bit_equiv = (diff_magnitude / max_value) * 255

    # Calculate percentage difference
    diff_percentage = (diff_magnitude / max_value) * 100

    # Calculate Delta E in CIELAB color space (more perceptually uniform)
    delta_e = None
    if len(arr1.shape) == 3 and arr1.shape[2] == 3:  # RGB images
        # Normalize to 0-1 range for color conversion
        img1_norm = (arr1 / max_value).astype(np.float32)
        img2_norm = (arr2 / max_value).astype(np.float32)

        # Convert RGB to LAB using optimized function if available
        if NUMBA_AVAILABLE:
            lab1 = fast_rgb_to_lab(img1_norm)
            lab2 = fast_rgb_to_lab(img2_norm)
            delta_e = fast_delta_e(lab1, lab2)
        else:
            from skimage import color

            lab1 = color.rgb2lab(img1_norm)
            lab2 = color.rgb2lab(img2_norm)
            delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    # Perceptibility assessment
    # JND (Just Noticeable Difference) thresholds:
    # - Delta E < 1.0: Imperceptible
    # - Delta E 1.0-2.3: Perceptible with close observation
    # - Delta E 2.3-10: Perceptible at a glance
    # - Delta E > 10: Obvious difference

    # For grayscale/raw differences: ~1-2% is typically JND
    jnd_threshold_percent = 1.0
    pixels_above_jnd = np.sum(diff_percentage > jnd_threshold_percent)
    percent_pixels_above_jnd = (pixels_above_jnd / diff_percentage.size) * 100

    # Perceptibility score (0-100, where 0 = identical, 100 = completely different)
    # Based on percentage of pixels above JND and magnitude of differences
    perceptibility_score = min(
        100, percent_pixels_above_jnd + (np.mean(diff_percentage) * 10)
    )

    results = {
        "diff_8bit_equiv_mean": np.mean(diff_8bit_equiv),
        "diff_8bit_equiv_max": np.max(diff_8bit_equiv),
        "diff_percentage_mean": np.mean(diff_percentage),
        "diff_percentage_max": np.max(diff_percentage),
        "percent_pixels_above_jnd": percent_pixels_above_jnd,
        "perceptibility_score": perceptibility_score,
    }

    if delta_e is not None:
        results["delta_e_mean"] = np.mean(delta_e)
        results["delta_e_max"] = np.max(delta_e)
        results["delta_e_above_1"] = (np.sum(delta_e > 1.0) / delta_e.size) * 100
        results["delta_e_above_2.3"] = (np.sum(delta_e > 2.3) / delta_e.size) * 100

    return results


def format_perceptual_analysis(metrics):
    """
    Format perceptual analysis results as a string.

    Returns:
        Formatted string with perceptual analysis
    """
    lines = []
    lines.append("\n=== PERCEPTUAL ANALYSIS ===")
    lines.append(
        f"8-bit Equivalent Difference - mean: {metrics['diff_8bit_equiv_mean']:.4f}, max: {metrics['diff_8bit_equiv_max']:.4f}"
    )
    lines.append(
        f"Percentage Difference - mean: {metrics['diff_percentage_mean']:.4f}%, max: {metrics['diff_percentage_max']:.4f}%"
    )

    if "delta_e_mean" in metrics:
        lines.append(
            f"\nDelta E (CIELAB) - mean: {metrics['delta_e_mean']:.2f}, max: {metrics['delta_e_max']:.2f}"
        )
        lines.append(
            f"Pixels with Delta E > 1.0 (perceptible): {metrics['delta_e_above_1']:.2f}%"
        )
        lines.append(
            f"Pixels with Delta E > 2.3 (noticeable): {metrics['delta_e_above_2.3']:.2f}%"
        )

    lines.append(
        f"\nPixels above JND threshold (1%): {metrics['percent_pixels_above_jnd']:.2f}%"
    )
    lines.append(f"Perceptibility Score: {metrics['perceptibility_score']:.1f}/100")

    # Interpretation
    score = metrics["perceptibility_score"]
    if score < 1:
        interpretation = "✓ IMPERCEPTIBLE - Differences are not visible to human eye"
    elif score < 5:
        interpretation = (
            "✓ BARELY PERCEPTIBLE - Differences only visible with careful examination"
        )
    elif score < 15:
        interpretation = "⚠ SLIGHTLY PERCEPTIBLE - Some differences may be noticed"
    elif score < 30:
        interpretation = "⚠ PERCEPTIBLE - Differences are noticeable"
    else:
        interpretation = "✗ CLEARLY VISIBLE - Obvious differences"

    lines.append(f"\n{interpretation}")
    lines.append("=" * 50)

    return "\n".join(lines)


def compare_px_diff(
    image1_path,
    image2_path,
    visualize=False,
    output_path=None,
    amplification=1.0,
    font_family="monospace",
    return_output=False,
):
    """
    Compare two images and optionally generate heatmap visualization.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        visualize: If True, generate visual comparison
        output_path: Path to save visualization (if None, displays instead)
        amplification: Multiplier for difference visibility (default 1.0)
        font_family: Font family for text annotations (default: monospace)
        return_output: If True, return formatted output string instead of printing

    Returns:
        If return_output is True: tuple of (abs_diff, rel_diff, output_string)
        Otherwise: tuple of (abs_diff, rel_diff)
    """
    # Load images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Convert images to numpy arrays (use ascontiguousarray for better cache performance)
    arr1 = np.ascontiguousarray(np.array(img1, dtype=np.float32))
    arr2 = np.ascontiguousarray(np.array(img2, dtype=np.float32))

    # Check if dimensions match
    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions for comparison.")

    # Compute absolute difference
    abs_diff = np.abs(arr1 - arr2)

    # Calculate per-pixel difference magnitude (for RGB, use mean across channels)
    # This is the standard way to represent overall pixel difference
    if len(abs_diff.shape) == 3:
        diff_magnitude = np.mean(abs_diff, axis=2)
    else:
        diff_magnitude = abs_diff

    # Compute relative difference, avoiding division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(arr1 != 0, abs_diff / np.abs(arr1), 0)

    # Build statistics output
    output_lines = []
    output_lines.append(
        f"Absolute Difference (per-pixel magnitude) - min: {np.min(diff_magnitude):.2f}, max: {np.max(diff_magnitude):.2f}, mean: {np.mean(diff_magnitude):.2f}"
    )
    output_lines.append(
        f"Absolute Difference (per-channel) - min: {np.min(abs_diff):.0f}, max: {np.max(abs_diff):.0f}, mean: {np.mean(abs_diff):.2f}"
    )
    output_lines.append(
        f"Relative Difference - min: {np.min(rel_diff):.2f}, max: {np.max(rel_diff):.2f}, mean: {np.mean(rel_diff):.2f}"
    )

    # Perceptual analysis
    bit_depth = 16  # Assume 16-bit images based on typical TIFF format
    perceptual_metrics = analyze_perceptual_difference(arr1, arr2, abs_diff, bit_depth)
    output_lines.append(format_perceptual_analysis(perceptual_metrics))

    # Generate visualization if requested
    if visualize:
        viz_msg = visualize_difference(
            arr1,
            arr2,
            abs_diff,
            output_path,
            amplification,
            perceptual_metrics,
            font_family,
        )
        if viz_msg:
            output_lines.append(viz_msg)

    output_string = "\n".join(output_lines)

    if return_output:
        return abs_diff, rel_diff, output_string
    else:
        print(output_string)
        return abs_diff, rel_diff


def visualize_difference(
    arr1,
    arr2,
    abs_diff,
    output_path=None,
    amplification=1.0,
    perceptual_metrics=None,
    font_family="monospace",
):
    """
    Create a side-by-side visualization with heatmap of differences.

    Args:
        arr1: First image array
        arr2: Second image array
        abs_diff: Absolute difference array
        output_path: Path to save the visualization
        amplification: Multiplier to make subtle differences more visible
        perceptual_metrics: Dictionary of perceptual analysis metrics
        font_family: Font family for text annotations (default: monospace)
    """

    # Normalize images for display (handle both RGB and grayscale)
    # Use in-place operations where possible
    max1 = arr1.max()
    max2 = arr2.max()

    if max1 > 0:
        img1_display = np.clip(arr1 / max1, 0, 1)
    else:
        img1_display = arr1

    if max2 > 0:
        img2_display = np.clip(arr2 / max2, 0, 1)
    else:
        img2_display = arr2

    # Calculate per-pixel difference magnitude (for RGB, use mean across channels)
    if len(abs_diff.shape) == 3:
        diff_magnitude = np.mean(abs_diff, axis=2)
    else:
        diff_magnitude = abs_diff

    # Apply amplification to make subtle differences visible
    diff_amplified = diff_magnitude * amplification

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Build comprehensive title with metrics
    title = "Image Comparison Analysis"
    if perceptual_metrics:
        title += (
            f" | Perceptibility: {perceptual_metrics['perceptibility_score']:.1f}/100"
        )
        if "delta_e_mean" in perceptual_metrics:
            title += f" | ΔE: {perceptual_metrics['delta_e_mean']:.2f}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Image 1
    axes[0, 0].imshow(img1_display, cmap="gray" if len(arr1.shape) == 2 else None)
    axes[0, 0].set_title("Image 1 (Original)", fontsize=12)
    axes[0, 0].axis("off")

    # Image 2
    axes[0, 1].imshow(img2_display, cmap="gray" if len(arr2.shape) == 2 else None)
    axes[0, 1].set_title("Image 2 (Modified)", fontsize=12)
    axes[0, 1].axis("off")

    # Heatmap with perceptual colors (blue=similar, red=different)
    im = axes[1, 0].imshow(diff_amplified, cmap="turbo", interpolation="nearest")
    axes[1, 0].set_title(
        f"Difference Heatmap (amplification: {amplification}x)", fontsize=12
    )
    axes[1, 0].axis("off")
    plt.colorbar(
        im, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Difference Magnitude"
    )

    # Histogram of differences
    axes[1, 1].hist(
        diff_magnitude.flatten(),
        bins=100,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    axes[1, 1].set_title("Difference Distribution", fontsize=12)
    axes[1, 1].set_xlabel("Absolute Difference")
    axes[1, 1].set_ylabel("Frequency (log scale)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    # Add comprehensive statistics text with all perceptual metrics
    stats_text = f"Difference Statistics:\n"
    stats_text += f"Mean: {np.mean(diff_magnitude):.2f}\n"
    stats_text += f"Max: {np.max(diff_magnitude):.2f}\n"
    stats_text += f"Std: {np.std(diff_magnitude):.2f}"

    if perceptual_metrics:
        stats_text += f"\n\nPerceptual Metrics:"
        stats_text += (
            f"\n8-bit equiv mean: {perceptual_metrics['diff_8bit_equiv_mean']:.4f}"
        )
        stats_text += (
            f"\n8-bit equiv max: {perceptual_metrics['diff_8bit_equiv_max']:.4f}"
        )
        stats_text += (
            f"\n% diff mean: {perceptual_metrics['diff_percentage_mean']:.4f}%"
        )
        stats_text += f"\n% diff max: {perceptual_metrics['diff_percentage_max']:.4f}%"
        stats_text += (
            f"\nPixels > JND: {perceptual_metrics['percent_pixels_above_jnd']:.2f}%"
        )
        stats_text += (
            f"\nPerceptibility: {perceptual_metrics['perceptibility_score']:.1f}/100"
        )

        if "delta_e_mean" in perceptual_metrics:
            stats_text += f"\n\nΔE (CIELAB):"
            stats_text += f"\nMean: {perceptual_metrics['delta_e_mean']:.2f}"
            stats_text += f"\nMax: {perceptual_metrics['delta_e_max']:.2f}"
            stats_text += f"\n>1.0: {perceptual_metrics['delta_e_above_1']:.2f}%"
            stats_text += f"\n>2.3: {perceptual_metrics['delta_e_above_2.3']:.2f}%"

        # Add interpretation
        score = perceptual_metrics["perceptibility_score"]
        if score < 1:
            interpretation = "IMPERCEPTIBLE"
        elif score < 5:
            interpretation = "BARELY PERCEPTIBLE"
        elif score < 15:
            interpretation = "SLIGHTLY PERCEPTIBLE"
        elif score < 30:
            interpretation = "PERCEPTIBLE"
        else:
            interpretation = "CLEARLY VISIBLE"
        stats_text += f"\n\nInterpretation:\n{interpretation}"

    axes[1, 1].text(
        0.98,
        0.97,
        stats_text,
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
        family=font_family,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return f"Visualization saved to: {os.path.abspath(output_path)}"
    else:
        plt.show()
        return None


def compare_single_file(
    filename, dir1, dir2, visualize, output_dir, amplification, font_family
):
    """
    Compare a single file from two directories.
    Worker function for parallel processing.

    Args:
        filename: Name of the file to compare
        dir1: First directory path
        dir2: Second directory path
        visualize: If True, generate visualizations
        output_dir: Directory to save visualizations
        amplification: Multiplier for difference visibility
        font_family: Font family for text annotations

    Returns:
        Tuple of (filename, success_status)
    """
    try:
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)

        # Determine output path for this comparison
        output_path = None
        if visualize and output_dir:
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_comparison.png")

        # Get output from comparison
        _, _, output_string = compare_px_diff(
            path1,
            path2,
            visualize=visualize,
            output_path=output_path,
            amplification=amplification,
            font_family=font_family,
            return_output=True,
        )

        # Print everything atomically to avoid interleaving
        complete_output = (
            f"\n{'='*60}\nComparing {filename}...\n{output_string}\n{'='*60}"
        )
        print(complete_output)

        return (filename, True)
    except Exception as e:
        error_msg = f"\n{'='*60}\nError comparing {filename}: {e}\n{'='*60}"
        print(error_msg)
        return (filename, False)


def compare_image_dirs(
    dir1,
    dir2,
    allowed_extensions=(".tiff", ".tif"),
    visualize=False,
    output_dir=None,
    amplification=1.0,
    font_family="monospace",
    workers=None,
):
    """
    Compare all matching images in two directories.

    Args:
        dir1: First directory path
        dir2: Second directory path
        allowed_extensions: Tuple of allowed file extensions
        visualize: If True, generate visualizations for each pair
        output_dir: Directory to save visualizations (if None, displays instead)
        amplification: Multiplier for difference visibility
        font_family: Font family for text annotations (default: monospace)
        workers: Number of parallel workers (default: None = auto-detect)
    """
    files1 = sorted(
        [f for f in os.listdir(dir1) if f.lower().endswith(allowed_extensions)]
    )
    files2 = sorted(
        [f for f in os.listdir(dir2) if f.lower().endswith(allowed_extensions)]
    )

    common_files = sorted(set(files1).intersection(set(files2)))

    if not common_files:
        print("No common files found to compare.")
        return

    # Create output directory if needed
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine number of workers
    if workers is None:
        workers = cpu_count() // 2 or 1  # Use half of available CPUs by default

    print(f"\nComparing {len(common_files)} files using {workers} workers...")

    # Create partial function with fixed parameters
    compare_func = partial(
        compare_single_file,
        dir1=dir1,
        dir2=dir2,
        visualize=visualize,
        output_dir=output_dir,
        amplification=amplification,
        font_family=font_family,
    )

    # Process files in parallel
    if workers == 1:
        # Serial processing (no multiprocessing overhead)
        results = [compare_func(filename) for filename in common_files]
    else:
        # Parallel processing
        with Pool(processes=workers) as pool:
            results = pool.map(compare_func, common_files)

    # Summary
    successful = sum(1 for _, success in results if success)
    print(f"\n{'='*60}")
    print(
        f"Comparison complete: {successful}/{len(common_files)} files processed successfully"
    )
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare images and analyze differences with perceptual metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two images
  python compare_images.py --image1 img1.tiff --image2 img2.tiff

  # Compare with visualization
  python compare_images.py --image1 img1.tiff --image2 img2.tiff --visualize --output result.png

  # Compare directories
  python compare_images.py --dir1 ./folder1 --dir2 ./folder2 --visualize --output-dir ./results

  # Amplify differences for better visibility
  python compare_images.py --dir1 ./folder1 --dir2 ./folder2 --visualize --amplification 10.0
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--image1", type=str, help="First image path (for single image comparison)"
    )
    mode_group.add_argument(
        "--dir1", type=str, help="First directory path (for directory comparison)"
    )

    # Second input (required when first is specified)
    parser.add_argument(
        "--image2", type=str, help="Second image path (required with --image1)"
    )
    parser.add_argument(
        "--dir2", type=str, help="Second directory path (required with --dir1)"
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Generate visualization with heatmap and metrics",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for single image comparison visualization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch comparison visualizations",
    )
    parser.add_argument(
        "--amplification",
        "-a",
        type=float,
        default=1.0,
        help="Difference amplification factor for visualization (default: 1.0)",
    )
    parser.add_argument(
        "--font-family",
        "-f",
        type=str,
        default="monospace",
        help="Font family for text annotations (default: monospace, options: sans-serif, serif, monospace, cursive, fantasy)",
    )

    # File filtering for directory mode
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".tiff", ".tif"],
        help="File extensions to compare in directory mode (default: .tiff .tif)",
    )

    # Parallel processing options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help=f"Number of parallel workers for directory comparison (default: auto-detect = {cpu_count()})",
    )

    args = parser.parse_args()

    # Show performance optimization status
    if NUMBA_AVAILABLE:
        print("⚡ Numba acceleration: ENABLED (fast RGB→LAB conversion)")
    else:
        print("⚠️  Numba acceleration: DISABLED (install with: pip install numba)")
        print("   Processing will be slower for large images.\n")

    # Validate input combinations
    if args.image1:
        if not args.image2:
            parser.error("--image2 is required when using --image1")

        # Single image comparison
        print(f"Comparing images:")
        print(f"  Image 1: {args.image1}")
        print(f"  Image 2: {args.image2}")

        compare_px_diff(
            args.image1,
            args.image2,
            visualize=args.visualize,
            output_path=args.output,
            amplification=args.amplification,
            font_family=args.font_family,
        )

    elif args.dir1:
        if not args.dir2:
            parser.error("--dir2 is required when using --dir1")

        # Directory comparison
        print(f"Comparing directories:")
        print(f"  Directory 1: {args.dir1}")
        print(f"  Directory 2: {args.dir2}")

        # Convert extensions to tuple
        extensions = tuple(args.extensions)

        compare_image_dirs(
            args.dir1,
            args.dir2,
            allowed_extensions=extensions,
            visualize=args.visualize,
            output_dir=args.output_dir,
            amplification=args.amplification,
            font_family=args.font_family,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
