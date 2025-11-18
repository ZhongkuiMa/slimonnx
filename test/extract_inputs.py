"""Extract input constraints from VNNLib files using torchvnnlib.

This module uses torchvnnlib to convert VNNLib files to .pth format containing
input bounds and output constraints as PyTorch tensors. The converted files are
saved in a directory structure organized by benchmark and VNNLib filename.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "extract_inputs_from_vnnlib",
    "extract_all_inputs",
]

import os
from pathlib import Path

from torchvnnlib import TorchVNNLIB


def extract_inputs_from_vnnlib(vnnlib_path: str, output_dir: str) -> None:
    """Convert VNNLib file to .pth files using torchvnnlib.

    Creates directory structure with OR groups and sub-properties:
    output_dir/
      or_group_0/
        sub_prop_0.pth  # {"input": Tensor(n_inputs, 2), "output": [Tensor]}
        sub_prop_1.pth
      or_group_1/
        ...

    :param vnnlib_path: Path to VNNLib file
    :param output_dir: Directory to store converted .pth files
    """
    converter = TorchVNNLIB(verbose=False, detect_fast_type=True)
    converter.convert(vnnlib_path, output_dir)


def extract_all_inputs(benchmarks_dir: str = "benchmarks") -> None:
    """Extract inputs from all VNNLib files using torchvnnlib.

    Uses instances.csv to discover VNNLib files and converts each unique file.
    Creates directory structure:
    benchmarks/
      benchmark_name/
        torchvnnlib/        # Converted .pth files from torchvnnlib
          vnnlib_name/      # VNNLib filename without .vnnlib suffix
            or_group_0/
              sub_prop_0.pth
              sub_prop_1.pth
            or_group_1/
              ...

    :param benchmarks_dir: Root directory containing benchmark subdirectories
    """
    benchmarks_path = Path(benchmarks_dir)

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    # Find all benchmark directories
    benchmark_dirs = sorted([d for d in benchmarks_path.iterdir() if d.is_dir()])

    if not benchmark_dirs:
        print(f"No benchmark directories found in {benchmarks_dir}")
        return

    print(f"Processing {len(benchmark_dirs)} benchmarks")
    print("=" * 60)

    total_success = 0
    total_failed = 0

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        # Skip hidden directories
        if benchmark_name.startswith("."):
            continue

        # Check for instances.csv
        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            print(f"[{benchmark_name}] No instances.csv, skipping")
            continue

        # Parse instances.csv to get unique VNNLib files
        unique_vnnlibs = set()
        try:
            with open(instances_csv) as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        vnnlib_path = parts[1].strip()
                        unique_vnnlibs.add(vnnlib_path)
        except Exception as e:
            print(f"[{benchmark_name}] Error reading instances.csv: {e}")
            continue

        if not unique_vnnlibs:
            print(f"[{benchmark_name}] No VNNLib files in instances.csv")
            continue

        # Create torchvnnlib directory inside benchmark
        torchvnnlib_dir = benchmark_dir / "torchvnnlib"
        torchvnnlib_dir.mkdir(parents=True, exist_ok=True)

        # Convert each unique VNNLib file
        success = 0
        failed = []

        for vnnlib_rel_path in sorted(unique_vnnlibs):
            vnnlib_file = benchmark_dir / vnnlib_rel_path
            vnnlib_name = Path(vnnlib_rel_path).stem  # e.g., "prop_1"

            if not vnnlib_file.exists():
                failed.append((vnnlib_name, "File not found"))
                continue

            # Output directory named after VNNLib file
            output_dir = torchvnnlib_dir / vnnlib_name

            try:
                extract_inputs_from_vnnlib(str(vnnlib_file), str(output_dir))
                success += 1
            except Exception as e:
                failed.append((vnnlib_name, str(e)))

        total_success += success
        total_failed += len(failed)

        print(f"[{benchmark_name}] Converted {success} files, {len(failed)} failed")
        if failed:
            for name, error in failed[:3]:  # Show first 3 errors
                print(f"  {name}: {error}")
            if len(failed) > 3:
                print(f"  ... and {len(failed)-3} more")

    print("=" * 60)
    print(f"Total: {total_success} converted, {total_failed} failed")


if __name__ == "__main__":
    extract_all_inputs()
