"""Extract input constraints from VNNLib files using torchvnnlib."""

__docformat__ = "restructuredtext"
__all__ = ["extract_inputs_from_vnnlib", "extract_all_inputs"]

from pathlib import Path

from torchvnnlib import TorchVNNLIB


def _parse_vnnlib_paths(csv_path: Path) -> set[str]:
    """Parse instances.csv and return unique VNNLib paths.

    :param csv_path: Path to instances.csv
    :return: Set of unique VNNLib paths
    """
    unique_vnnlibs = set()
    try:
        with open(csv_path) as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        unique_vnnlibs.add(parts[1].strip())
    except Exception as e:
        raise RuntimeError(f"Error parsing {csv_path}: {e}")
    return unique_vnnlibs


def extract_inputs_from_vnnlib(vnnlib_path: str, output_dir: str) -> None:
    """Convert VNNLib file to .pth files using torchvnnlib.

    Creates directory structure with OR groups and sub-properties:
    output_dir/
      or_group_0/
        sub_prop_0.pth
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

    :param benchmarks_dir: Root directory containing benchmark subdirectories
    """
    benchmarks_path = Path(benchmarks_dir)

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

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

        if benchmark_name.startswith("."):
            continue

        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            print(f"[{benchmark_name}] No instances.csv, skipping")
            continue

        try:
            unique_vnnlibs = _parse_vnnlib_paths(instances_csv)
        except RuntimeError as e:
            print(f"[{benchmark_name}] {e}")
            continue

        if not unique_vnnlibs:
            print(f"[{benchmark_name}] No VNNLib files in instances.csv")
            continue

        torchvnnlib_dir = benchmark_dir / "torchvnnlib"
        torchvnnlib_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = []

        for vnnlib_rel_path in sorted(unique_vnnlibs):
            vnnlib_file = benchmark_dir / vnnlib_rel_path
            vnnlib_name = Path(vnnlib_rel_path).stem

            if not vnnlib_file.exists():
                failed.append((vnnlib_name, "File not found"))
                continue

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
            for name, error in failed[:3]:
                print(f"  {name}: {error}")
            if len(failed) > 3:
                print(f"  ... and {len(failed) - 3} more")

    print("=" * 60)
    print(f"Total: {total_success} converted, {total_failed} failed")


if __name__ == "__main__":
    extract_all_inputs()
