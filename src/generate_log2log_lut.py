"""
General Log-to-Log LUT Generator
Supports conversion between mainstream log curves with their native color spaces.
"""

import argparse
import os
from typing import Optional

import colour
import numpy as np

# Log curve and color space configuration
LOG_CONFIGS = {
    "S-Log3": {
        "encoding": colour.models.log_encoding_SLog3,
        "decoding": colour.models.log_decoding_SLog3,
        "colorspace": colour.models.RGB_COLOURSPACE_S_GAMUT3,
        "full_name": "Sony S-Log3 / S-Gamut3",
    },
    "S-Log3.Cine": {
        "encoding": colour.models.log_encoding_SLog3,
        "decoding": colour.models.log_decoding_SLog3,
        "colorspace": colour.models.RGB_COLOURSPACE_S_GAMUT3_CINE,
        "full_name": "Sony S-Log3 / S-Gamut3.Cine",
    },
    "F-Log": {
        "encoding": colour.models.log_encoding_FLog,
        "decoding": colour.models.log_decoding_FLog,
        "colorspace": colour.models.RGB_COLOURSPACE_F_GAMUT,
        "full_name": "Fujifilm F-Log / F-Gamut",
    },
    "F-Log2": {
        "encoding": colour.models.log_encoding_FLog2,
        "decoding": colour.models.log_decoding_FLog2,
        "colorspace": colour.models.RGB_COLOURSPACE_F_GAMUT,
        "full_name": "Fujifilm F-Log2 / F-Gamut",
    },
    "F-Log2C": {
        "encoding": colour.models.log_encoding_FLog2,
        "decoding": colour.models.log_decoding_FLog2,
        "colorspace": colour.models.RGB_COLOURSPACE_F_GAMUT_C,
        "full_name": "Fujifilm F-Log2 C / F-Gamut C",
    },
    "C-Log2": {
        "encoding": colour.models.log_encoding_CanonLog2,
        "decoding": colour.models.log_decoding_CanonLog2,
        "colorspace": colour.models.RGB_COLOURSPACE_CINEMA_GAMUT,
        "full_name": "Canon Log 2 / Cinema Gamut",
    },
    "C-Log3": {
        "encoding": colour.models.log_encoding_CanonLog3,
        "decoding": colour.models.log_decoding_CanonLog3,
        "colorspace": colour.models.RGB_COLOURSPACE_CINEMA_GAMUT,
        "full_name": "Canon Log 3 / Cinema Gamut",
    },
    "LogC3": {
        "encoding": colour.models.log_encoding_ARRILogC3,
        "decoding": colour.models.log_decoding_ARRILogC3,
        "colorspace": colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3,
        "full_name": "ARRI LogC3 / ARRI Wide Gamut 3",
    },
    "LogC4": {
        "encoding": colour.models.log_encoding_ARRILogC4,
        "decoding": colour.models.log_decoding_ARRILogC4,
        "colorspace": colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4,
        "full_name": "ARRI LogC4 / ARRI Wide Gamut 4",
    },
    "V-Log": {
        "encoding": colour.models.log_encoding_VLog,
        "decoding": colour.models.log_decoding_VLog,
        "colorspace": colour.models.RGB_COLOURSPACE_V_GAMUT,
        "full_name": "Panasonic V-Log / V-Gamut",
    },
    "N-Log": {
        "encoding": colour.models.log_encoding_NLog,
        "decoding": colour.models.log_decoding_NLog,
        "colorspace": colour.models.RGB_COLOURSPACE_N_GAMUT,
        "full_name": "Nikon N-Log / N-Gamut",
    },
    "L-Log": {
        "encoding": colour.models.log_encoding_LLog,
        "decoding": colour.models.log_decoding_LLog,
        # L-log uses BT.2020 gamut
        "colorspace": colour.models.RGB_COLOURSPACE_BT2020,
        "full_name": "Leica L-Log / L-Gamut",
    },
    "DaVinci Intermediate": {
        "encoding": colour.models.oetf_DaVinciIntermediate,
        "decoding": colour.models.oetf_inverse_DaVinciIntermediate,
        "colorspace": colour.models.RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT,
        "full_name": "DaVinci Intermediate / DaVinci Wide Gamut",
    },
    "Log3G10": {
        "encoding": colour.models.log_encoding_Log3G10,
        "decoding": colour.models.log_decoding_Log3G10,
        "colorspace": colour.models.RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB,
        "full_name": "RED Log3G10 / RED Wide Gamut RGB",
    },
    "Rec.709 Gamma 2.4": {
        "encoding": colour.models.oetf_BT709,
        "decoding": colour.models.oetf_inverse_BT709,
        "colorspace": colour.models.RGB_COLOURSPACE_BT709,
        "full_name": "Rec.709 Gamma 2.4 / Rec.709",
    },
    "Rec.2020 Gamma 2.4": {
        "encoding": colour.models.oetf_BT2020,
        "decoding": colour.models.oetf_inverse_BT2020,
        "colorspace": colour.models.RGB_COLOURSPACE_BT2020,
        "full_name": "Rec.2020 Gamma 2.4 / Rec.2020",
    },
    "sRGB": {
        "encoding": colour.models.eotf_inverse_sRGB,
        "decoding": colour.models.eotf_sRGB,
        "colorspace": colour.models.RGB_COLOURSPACE_sRGB,
        "full_name": "sRGB / sRGB",
    },
}


def normalize_log_name(log_name: str) -> str:
    """
    Normalize log name to match LOG_CONFIGS keys case-insensitively.

    Args:
        log_name: User-provided log name (any case)

    Returns:
        Properly cased log name from LOG_CONFIGS, or None if not found
    """
    # Create case-insensitive lookup
    for key in LOG_CONFIGS.keys():
        if key.lower() == log_name.lower():
            return key
    return None


def generate_log_to_log_lut(
    source_log: str,
    target_log: str,
    lut_size: int = 65,
    out_path: Optional[str] = None,
    chromatic_adaptation: str = "CAT02",
) -> str:
    """
    Generate a 3D LUT for converting between two log curves with gamut conversion.

    Args:
        source_log: Source log curve name (e.g., "Arri LogC4", "SLog3.Cine")
        target_log: Target log curve name
        lut_size: LUT resolution (17, 33, 65, or 129 recommended)
        out_path: Optional output file path. If None, auto-generates filename.
        chromatic_adaptation: CAT method for gamut conversion ("CAT02", "Bradford", etc.)

    Returns:
        Path to the generated LUT file
    """

    # Validate inputs (case-insensitive)
    normalized_source = normalize_log_name(source_log)
    if normalized_source is None:
        raise ValueError(
            f"Unknown source log: {source_log}. Available: {list(LOG_CONFIGS.keys())}"
        )

    normalized_target = normalize_log_name(target_log)
    if normalized_target is None:
        raise ValueError(
            f"Unknown target log: {target_log}. Available: {list(LOG_CONFIGS.keys())}"
        )

    source_config = LOG_CONFIGS[normalized_source]
    target_config = LOG_CONFIGS[normalized_target]

    # Generate output path if not provided
    if out_path is None:
        source_name = normalized_source.replace(" ", "_").replace(".", "")
        target_name = normalized_target.replace(" ", "_").replace(".", "")
        out_path = f"{source_name}_to_{target_name}_{lut_size}.cube"

    # Convert to absolute path for display
    out_path = os.path.abspath(out_path)

    print(
        f"Generating LUT: {source_config['full_name']} -> {target_config['full_name']}"
    )
    print(f"Output: {out_path}")
    print(f"LUT Size: {lut_size}^3 = {lut_size**3:,} samples")

    # Create LUT grid
    lut_name = f"{normalized_source}_to_{normalized_target}"
    LUT = colour.LUT3D(size=lut_size, name=lut_name)
    rgb_source_log = LUT.table.reshape(-1, 3)

    # Processing pipeline
    print("  [1/4] Decoding source log curve to linear...")
    # Suppress numpy warnings for out-of-gamut values during decoding
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb_linear_source = source_config["decoding"](rgb_source_log)

    print("  [2/4] Converting color gamut...")
    rgb_linear_target = colour.RGB_to_RGB(
        rgb_linear_source,
        input_colourspace=source_config["colorspace"],
        output_colourspace=target_config["colorspace"],
        chromatic_adaptation_transform=chromatic_adaptation,
    )

    print("  [3/4] Encoding to target log curve...")
    # Suppress numpy warnings for edge cases (negative values, etc.)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb_target_log = target_config["encoding"](rgb_linear_target)

    print("  [4/4] Finalizing LUT...")
    # Reshape back to 3D LUT structure
    LUT.table = rgb_target_log.reshape(lut_size, lut_size, lut_size, 3)

    # Sanitize: Clip to standard 0-1 range and handle NaN/Inf
    # This is simpler than custom encoding functions!
    LUT.table = np.nan_to_num(LUT.table, nan=0.0, posinf=1.0, neginf=0.0)
    LUT.table = np.clip(LUT.table, 0.0, 1.0)

    # Save LUT
    colour.write_LUT(LUT, out_path)
    print(f"✓ Success! Saved: {os.path.abspath(out_path)}\n")

    return out_path


def generate_multiple_luts(
    source_log: str, target_logs: list = None, lut_size: int = 65, output_dir: str = "."
):
    """
    Generate LUTs from one source log to multiple target logs.

    Args:
        source_log: Source log curve name
        target_logs: List of target log names. If None, generates for all available.
        lut_size: LUT resolution
        output_dir: Directory to save LUT files
    """
    # Normalize source log name
    normalized_source = normalize_log_name(source_log)
    if normalized_source is None:
        raise ValueError(
            f"Unknown source log: {source_log}. Available: {list(LOG_CONFIGS.keys())}"
        )

    if target_logs is None:
        target_logs = [log for log in LOG_CONFIGS.keys() if log != normalized_source]
    else:
        # Normalize target log names
        normalized_targets = []
        for target in target_logs:
            normalized = normalize_log_name(target)
            if normalized is None:
                raise ValueError(
                    f"Unknown target log: {target}. Available: {list(LOG_CONFIGS.keys())}"
                )
            normalized_targets.append(normalized)
        target_logs = normalized_targets

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    print(f"Batch Generation: {normalized_source} -> [{', '.join(target_logs)}]")
    print("=" * 70)

    generated_files = []
    for target_log in target_logs:
        if target_log == normalized_source:
            print(
                f"Skipping {normalized_source} -> {normalized_source} (same format)\n"
            )
            continue

        source_name = normalized_source.replace(" ", "_").replace(".", "")
        target_name = target_log.replace(" ", "_").replace(".", "")
        out_path = os.path.join(
            output_dir, f"{source_name}_to_{target_name}_{lut_size}.cube"
        )

        try:
            result_path = generate_log_to_log_lut(
                source_log=normalized_source,
                target_log=target_log,
                lut_size=lut_size,
                out_path=out_path,
            )
            generated_files.append(result_path)
        except Exception as e:
            print(f"✗ Error generating {source_log} -> {target_log}: {e}\n")

    print("=" * 70)
    print(f"Batch complete! Generated {len(generated_files)} LUTs")
    return generated_files


def list_formats():
    """List all available log formats."""
    print("Available Log Formats:")
    print("-" * 70)
    for key, config in LOG_CONFIGS.items():
        print(f"  • {key:20s} - {config['full_name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D LUTs for converting between different log curves and color spaces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single LUT conversion
  python generate_log2log_lut.py --source "LogC4" --target "F-Log2C" --size 65 --output output.cube

  # Batch conversion to all formats
  python generate_log2log_lut.py --source "LogC4" --batch --output-dir ./luts

  # Batch conversion to specific formats
  python generate_log2log_lut.py --source "S-Log3.Cine" --batch --targets "F-Log2C" "C-Log3" --output-dir ./luts

  # List available formats
  python generate_log2log_lut.py --list
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available log formats and exit"
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="Source log curve (e.g., 'LogC4', 'S-Log3.Cine')",
    )

    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Target log curve (required for single conversion)",
    )

    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Batch mode: convert source to multiple target formats",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        help="Specific target log curves for batch mode (default: all except source)",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=65,
        choices=[17, 33, 65, 129],
        help="LUT resolution (default: 65)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (for single conversion, auto-generated if not specified)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (for batch mode, default: current directory)",
    )

    parser.add_argument(
        "--cat",
        type=str,
        default="CAT02",
        help="Chromatic adaptation transform method (default: CAT02)",
    )

    args = parser.parse_args()

    # Handle --list flag
    if args.list:
        list_formats()
        return

    # Validate required arguments
    if not args.source:
        parser.error("--source is required (unless using --list)")

    # Batch mode
    if args.batch:
        print(f"Batch Mode: Converting {args.source} to multiple formats")
        print("=" * 70)
        generate_multiple_luts(
            source_log=args.source,
            target_logs=args.targets,
            lut_size=args.size,
            output_dir=args.output_dir,
        )
    # Single conversion mode
    elif args.target:
        generate_log_to_log_lut(
            source_log=args.source,
            target_log=args.target,
            lut_size=args.size,
            out_path=args.output,
            chromatic_adaptation=args.cat,
        )
    else:
        parser.error(
            "Either --target (for single conversion) or --batch (for batch conversion) is required"
        )


if __name__ == "__main__":
    main()
