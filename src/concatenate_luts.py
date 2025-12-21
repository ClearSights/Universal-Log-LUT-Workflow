import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import colour


def _combine_single_pair(lut1_path, lut2_path, output_path):
    """
    Helper function to load two LUTs, concatenate them, and save the result.
    Mathematically: Output = LUT2(LUT1(x))

    Args:
        lut1_path: Path to first LUT file
        lut2_path: Path to second LUT file
        output_path: Full file path (for file+file) or directory path (for batch)
    """
    try:
        # 1. Read the LUTs
        # colour.read_LUT returns a LUT3D (or LUT1D/3x1D) object
        lut1 = colour.read_LUT(lut1_path)
        lut2 = colour.read_LUT(lut2_path)

        # Ensure they are handled as 3D LUTs for consistency
        if not isinstance(lut1, colour.LUT3D) or not isinstance(lut2, colour.LUT3D):
            print(f"Warning: {lut1_path} or {lut2_path} is not a 3D LUT. Converting...")
            lut1 = colour.LUT3D(table=lut1.table, name=lut1.name)
            lut2 = colour.LUT3D(table=lut2.table, name=lut2.name)

        # 2. Mathematical Concatenation (Composition)
        # We take the grid points (table) of LUT1 and pass them through LUT2.
        # LUT2.apply() uses Trilinear Interpolation to find values for inputs
        # that don't align perfectly with LUT2's own grid nodes.
        combined_table = lut2.apply(lut1.table)

        # 3. Create the new LUT object
        # The resolution (size) will match LUT1, which is standard for concatenation.
        # new_name = f"{os.path.splitext(os.path.basename(lut1_path))[0]}_PLUS_{os.path.splitext(os.path.basename(lut2_path))[0]}"
        # use 1st LUT name only
        new_name = f"{os.path.splitext(os.path.basename(lut2_path))[0]}"

        combined_lut = colour.LUT3D(table=combined_table, name=new_name)

        # 4. Save
        # If output_path is a file path (ends with .cube), use it directly
        # Otherwise, treat it as a directory
        if output_path.lower().endswith(".cube"):
            out_path = output_path
            out_dir = os.path.dirname(output_path)
        else:
            out_dir = output_path
            out_path = os.path.join(out_dir, f"{new_name}.cube")

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        colour.write_LUT(combined_lut, out_path)
        print(f"[OK] Saved: {os.path.abspath(out_path)}")

    except Exception as e:
        print(f"[ERROR] Failed combining {lut1_path} and {lut2_path}: {e}")


def process_luts(input1, input2, output_path, max_workers=None):
    """
    Concatenates two 3D LUTs.
    Order: input1 is applied first, then input2.

    Args:
        input1 (str): File path or Directory path.
        input2 (str): File path or Directory path.
        output_path (str): Output file path (for file+file) or directory path (for batch).
        max_workers (int, optional): Number of parallel processes for batch operations.
                                     Defaults to CPU count. Set to 1 for sequential processing.
    """

    if max_workers is None:
        max_workers = cpu_count()

    is_dir1 = os.path.isdir(input1)
    is_dir2 = os.path.isdir(input2)

    # Validation: At most one is a directory
    if is_dir1 and is_dir2:
        raise ValueError(
            "Both inputs cannot be directories. At most one can be a directory."
        )

    # Case 1: File + File
    if not is_dir1 and not is_dir2:
        # For file+file, output_path should be a file path
        # If user provides a directory, generate a filename
        if os.path.isdir(output_path) or not output_path.lower().endswith(".cube"):
            base1 = os.path.splitext(os.path.basename(input1))[0]
            base2 = os.path.splitext(os.path.basename(input2))[0]
            out_filename = f"{base1}_PLUS_{base2}.cube"
            if os.path.isdir(output_path):
                output_file = os.path.join(output_path, out_filename)
            else:
                # If output_path is not a dir and not ending with .cube, treat as dir
                output_file = os.path.join(output_path, out_filename)
        else:
            output_file = output_path
        _combine_single_pair(input1, input2, output_file)

    # Case 2: Dir + File (Iterate input1, apply input2 to all)
    elif is_dir1:
        # For batch operations, output_path should be a directory
        if output_path.lower().endswith(".cube"):
            raise ValueError(
                "When processing a directory, output_path must be a directory, not a file path."
            )

        cube_files = glob.glob(os.path.join(input1, "*.cube"))
        print(
            f"Found {len(cube_files)} LUTs in {input1} to process against {input2}..."
        )
        print(f"Using {max_workers} parallel workers...")

        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _combine_single_pair, file_path, input2, output_path
                ): file_path
                for file_path in cube_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Failed processing {file_path}: {e}")

    # Case 3: File + Dir (Apply file first, then iterate input2)
    elif is_dir2:
        # For batch operations, output_path should be a directory
        if output_path.lower().endswith(".cube"):
            raise ValueError(
                "When processing a directory, output_path must be a directory, not a file path."
            )

        cube_files = glob.glob(os.path.join(input2, "*.cube"))
        print(f"Found {len(cube_files)} LUTs in {input2} to apply after {input1}...")
        print(f"Using {max_workers} parallel workers...")

        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _combine_single_pair, input1, file_path, output_path
                ): file_path
                for file_path in cube_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Failed processing {file_path}: {e}")


# --- Usage Example ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate two 3D LUTs. Order: input1 is applied first, then input2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two files -> Output file
  python concatenate_luts.py -i1 logc4_to_logc4.cube -i2 logc4_to_rec709.cube -o output.cube

  # Two files -> Output directory (filename auto-generated)
  python concatenate_luts.py -i1 logc4_to_slog3.cube -i2 slog3_to_rec709.cube -o ./output_folder

  # Directory + File -> Output directory with parallel processing
  python concatenate_luts.py -i1 ./camera_log_luts -i2 Teal_Orange.cube -o ./output_folder -w 4

  # File + Directory -> Output directory with max parallelism
  python concatenate_luts.py -i1 logc4_to_slog3.cube -i2 ./slog3_creative_luts -o ./output_folder
        """,
    )

    parser.add_argument(
        "-i1",
        "--input1",
        required=True,
        help="First LUT file or directory path (applied first)",
    )

    parser.add_argument(
        "-i2",
        "--input2",
        required=True,
        help="Second LUT file or directory path (applied second)",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path (for file+file) or directory path (for batch operations)",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for batch operations (default: CPU count)",
    )

    args = parser.parse_args()

    # Convert relative paths to absolute paths
    input1 = os.path.abspath(args.input1)
    input2 = os.path.abspath(args.input2)
    output = os.path.abspath(args.output)

    print(f"Input 1: {input1}")
    print(f"Input 2: {input2}")
    print(f"Output: {output}")
    print(f"Workers: {args.workers if args.workers else cpu_count()}")
    print()

    process_luts(input1, input2, output, max_workers=args.workers)
