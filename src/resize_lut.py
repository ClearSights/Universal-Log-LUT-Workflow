import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
import os
import sys


def resize_lut(input_path, output_path, target_size):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"Reading {input_path}...")

    data = []
    source_size = 0

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                source_size = int(line.split()[-1])
                continue

            # Extract RGB values
            parts = line.split()
            if len(parts) == 3:
                try:
                    data.append([float(x) for x in parts])
                except ValueError:
                    continue

    if not source_size:
        print("Error: Could not find LUT_3D_SIZE in the header.")
        sys.exit(1)

    # 3D LUTs in .cube format: Blue varies fastest, then Green, then Red
    # Reshape to (Red, Green, Blue, Channels)
    try:
        lut_array = np.array(data).reshape((source_size, source_size, source_size, 3))
    except ValueError as e:
        print(f"Error: Data dimensions do not match LUT_3D_SIZE {source_size}. {e}")
        sys.exit(1)

    # Setup Interpolation on a normalized 0-1 scale
    coords = np.linspace(0, 1, source_size)
    interp = RegularGridInterpolator(
        (coords, coords, coords), lut_array, method="linear"
    )

    # Generate Target Grid
    t_coords = np.linspace(0, 1, target_size)
    # indexing='ij' ensures the Blue-fastest ordering is maintained during flattening
    grid_r, grid_g, grid_b = np.meshgrid(t_coords, t_coords, t_coords, indexing="ij")
    pts = np.stack([grid_r, grid_g, grid_b], axis=-1).reshape(-1, 3)

    print(f"Interpolating {source_size}^3 -> {target_size}^3...")
    new_data = interp(pts)

    # Write Output
    with open(output_path, "w") as f:
        f.write(f"# Resized via Python CLI\n")
        f.write(f"LUT_3D_SIZE {target_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for val in new_data:
            f.write(f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n")

    print(f"Success! Saved {target_size} grid LUT to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize a 3D .cube LUT using trilinear interpolation.",
        epilog="Examples:\n"
        "  %(prog)s input.cube --target-size 33\n"
        "  %(prog)s input.cube -s 65 -o output.cube\n"
        "  %(prog)s my_lut.cube --target-size 17",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to the source .cube file")
    parser.add_argument(
        "-s",
        "--target-size",
        type=int,
        default=33,
        help="Target grid size (e.g., 33, 65, 17). Default is 33.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. If omitted, will append '_resized' to the filename.",
    )

    args = parser.parse_args()

    # Determine output filename if not provided
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_{args.target_size}{ext}"

    resize_lut(args.input, args.output, args.target_size)


if __name__ == "__main__":
    main()
