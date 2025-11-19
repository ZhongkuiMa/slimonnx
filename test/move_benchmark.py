"""Copy ONNX models, VNNLib files, and instances.csv from VNN-COMP 2024 benchmarks.

This script copies files from vnncomp2024_benchmarks, using instances.csv to
maintain correct ONNX-VNNLib pairings. Creates filtered instances.csv files
containing only the copied file pairs.
"""

__docformat__ = "restructuredtext"
__all__ = ["parse_instances_csv", "copy_benchmark_files", "main"]

import shutil
from pathlib import Path


def parse_instances_csv(csv_path: str) -> dict[str, list[tuple[str, str]]]:
    """Parse instances.csv to get ONNX to VNNLib mappings.

    CSV format: model_path,property_path,timeout
    First line is header and should be skipped.

    :param csv_path: Path to instances.csv file
    :return: Dictionary mapping ONNX paths to list of (vnnlib_path, timeout) tuples
    """
    onnx_to_vnnlib = {}

    with open(csv_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) >= 2:
                model_path = parts[0].strip()
                property_path = parts[1].strip()
                timeout = parts[2].strip() if len(parts) >= 3 else "300"

                if model_path not in onnx_to_vnnlib:
                    onnx_to_vnnlib[model_path] = []
                onnx_to_vnnlib[model_path].append((property_path, timeout))

    return onnx_to_vnnlib


def copy_benchmark_files(
    source_dir: str = "../../../vnncomp2024_benchmarks/benchmarks",
    target_dir: str = "benchmarks",
    max_onnx_per_benchmark: int = 20,
    max_vnnlib_per_onnx: int = 2,
) -> tuple[int, int, int]:
    """Copy ONNX models and VNNLib files from VNN-COMP benchmarks.

    Uses instances.csv to maintain correct ONNX-VNNLib pairings.
    Creates filtered instances.csv containing only copied file pairs.

    :param source_dir: Path to vnncomp2024_benchmarks/benchmarks directory
    :param target_dir: Target directory for copied files
    :param max_onnx_per_benchmark: Maximum ONNX files per benchmark
    :param max_vnnlib_per_onnx: Maximum VNNLib files per ONNX model
    :return: Tuple of (num_onnx, num_vnnlib, num_benchmarks)
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir)

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    print(f"Source: {source}")
    print(f"Target: {target.resolve()}")
    print(f"Max ONNX per benchmark: {max_onnx_per_benchmark}")
    print(f"Max VNNLib per ONNX: {max_vnnlib_per_onnx}")
    print("=" * 70)

    # Create target directory
    target.mkdir(parents=True, exist_ok=True)

    # Find all benchmark directories
    benchmark_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    print(f"Found {len(benchmark_dirs)} benchmark directories\n")

    total_onnx = 0
    total_vnnlib = 0
    total_benchmarks = 0

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

        # Parse instances.csv
        try:
            onnx_to_vnnlib = parse_instances_csv(str(instances_csv))
        except Exception as e:
            print(f"[{benchmark_name}] Error parsing instances.csv: {e}, skipping")
            continue

        if not onnx_to_vnnlib:
            print(f"[{benchmark_name}] Empty instances.csv, skipping")
            continue

        # Create target benchmark directory
        target_benchmark_dir = target / benchmark_name
        target_benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Select ONNX models (first N, sorted alphabetically)
        all_onnx_models = sorted(onnx_to_vnnlib.keys())
        selected_onnx_models = all_onnx_models[:max_onnx_per_benchmark]

        # Track copied files for filtered instances.csv
        filtered_instances = []
        onnx_copied = 0
        vnnlib_copied = 0

        for onnx_path in selected_onnx_models:
            # Copy ONNX file
            source_onnx = benchmark_dir / onnx_path
            target_onnx = target_benchmark_dir / onnx_path

            if source_onnx.exists():
                target_onnx.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_onnx, target_onnx)
                onnx_copied += 1

                # Copy associated VNNLib files (max N per ONNX)
                vnnlib_list = onnx_to_vnnlib[onnx_path][:max_vnnlib_per_onnx]

                for vnnlib_path, timeout in vnnlib_list:
                    source_vnnlib = benchmark_dir / vnnlib_path
                    target_vnnlib = target_benchmark_dir / vnnlib_path

                    if source_vnnlib.exists():
                        target_vnnlib.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_vnnlib, target_vnnlib)
                        vnnlib_copied += 1

                        # Add to filtered instances
                        filtered_instances.append(
                            f"{onnx_path},{vnnlib_path},{timeout}"
                        )
                    else:
                        print(
                            f"[{benchmark_name}] Warning: VNNLib not found: {vnnlib_path}"
                        )
            else:
                print(f"[{benchmark_name}] Warning: ONNX not found: {onnx_path}")

        # Write filtered instances.csv
        if filtered_instances:
            target_csv = target_benchmark_dir / "instances.csv"
            with open(target_csv, "w") as f:
                f.write("model_path,property_path,timeout\n")
                for instance in filtered_instances:
                    f.write(instance + "\n")

        total_onnx += onnx_copied
        total_vnnlib += vnnlib_copied
        total_benchmarks += 1

        print(
            f"[{benchmark_name}] Copied {onnx_copied} ONNX, {vnnlib_copied} VNNLib, {len(filtered_instances)} instances"
        )

    print("=" * 70)
    print(
        f"Total: {total_onnx} ONNX, {total_vnnlib} VNNLib from {total_benchmarks} benchmarks"
    )

    return total_onnx, total_vnnlib, total_benchmarks


def main() -> None:
    """Main entry point for script execution.

    :return: None
    """
    try:
        copy_benchmark_files()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
