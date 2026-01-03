"""Benchmark file discovery and path manipulation utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "find_benchmark_folders",
    "find_benchmarks",
    "find_models",
    "find_onnx_files_from_instances",
    "find_vnnlib_files",
    "find_vnnlib_files_from_instances",
    "get_benchmark_dir",
    "get_benchmark_name",
    "get_model_benchmark_name",
    "get_model_data_path",
    "get_model_relative_path",
    "read_instances_csv",
]

from pathlib import Path


def read_instances_csv(benchmark_path: Path) -> list[tuple[str, str, str]]:
    """Read instances.csv and return list of (model_path, vnnlib_path, timeout) tuples.

    :param benchmark_path: Path to benchmark directory
    :return: List of (model_path, vnnlib_path, timeout) tuples
    """
    instances_csv = benchmark_path / "instances.csv"
    if not instances_csv.exists():
        return []

    instances = []
    try:
        with instances_csv.open() as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    model_path = parts[0].strip()
                    vnnlib_path = parts[1].strip()
                    timeout = parts[2].strip() if len(parts) >= 3 else "300"
                    instances.append((model_path, vnnlib_path, timeout))
    except OSError:
        return []

    return instances


def find_benchmarks(base_dir: str) -> list[Path]:
    """Find all benchmark directories containing instances.csv.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    return [
        entry
        for entry in sorted(base_path.iterdir())
        if entry.is_dir() and not entry.name.startswith(".") and (entry / "instances.csv").exists()
    ]


def find_models(benchmarks: list[Path], max_per_benchmark: int = 20) -> list[Path]:
    """Find ONNX model files from benchmark directories.

    :param benchmarks: List of benchmark directory paths
    :param max_per_benchmark: Maximum models per benchmark
    :return: List of ONNX model file paths
    """
    models = []
    for benchmark in benchmarks:
        instances = read_instances_csv(benchmark)
        seen = set()

        for model_path, _, _ in instances:
            if model_path in seen:
                continue

            full_path = benchmark / model_path
            if full_path.exists():
                models.append(full_path)
                seen.add(model_path)

                if len(seen) >= max_per_benchmark:
                    break

    return sorted(models)


def find_vnnlib_files(benchmarks: list[Path]) -> list[Path]:
    """Find VNNLib property files from benchmark directories.

    :param benchmarks: List of benchmark directory paths
    :return: List of VNNLib file paths
    """
    vnnlib_files = []
    seen = set()

    for benchmark in benchmarks:
        instances = read_instances_csv(benchmark)

        for _, vnnlib_path, _ in instances:
            if vnnlib_path in seen:
                continue

            full_path = benchmark / vnnlib_path
            if full_path.exists():
                vnnlib_files.append(full_path)
                seen.add(vnnlib_path)

    return sorted(vnnlib_files)


def get_model_benchmark_name(
    model_path: Path, benchmarks_root: str = "vnncomp2024_benchmarks"
) -> str:
    """Extract benchmark name from model path.

    :param model_path: Path to model file
    :param benchmarks_root: Name of benchmarks root directory
    :return: Benchmark name
    """
    parts = model_path.parts
    try:
        bench_idx = parts.index(benchmarks_root)
        if bench_idx + 1 < len(parts):
            return parts[bench_idx + 1]
    except ValueError:
        pass

    return model_path.parent.name


def get_model_relative_path(model_path: Path, benchmarks_root: Path) -> Path:
    """Get model path relative to benchmarks root.

    :param model_path: Full path to model file
    :param benchmarks_root: Path to benchmarks root directory
    :return: Relative path from benchmarks root
    """
    try:
        return model_path.relative_to(benchmarks_root)
    except ValueError:
        return Path(model_path.name)


def get_model_data_path(model_path: Path, benchmarks_root: Path) -> Path:
    """Get corresponding data file path for a model.

    Preserves subdirectory structure from onnx/ to data/ directory.
    For example:
    - benchmarks/safenlp/onnx/medical/model.onnx
    - benchmarks/safenlp/data/medical/model.npz

    :param model_path: Path to model file
    :param benchmarks_root: Path to benchmarks root directory
    :return: Path to data file (npz format)
    """
    rel_path = get_model_relative_path(model_path, benchmarks_root)
    benchmark_name = get_model_benchmark_name(model_path)
    benchmark_dir = benchmarks_root / benchmark_name

    rel_to_benchmark = rel_path.relative_to(benchmark_name)
    if rel_to_benchmark.parts[0] == "onnx":
        subpath = Path(*rel_to_benchmark.parts[1:])
    else:
        subpath = rel_to_benchmark

    data_path = benchmark_dir / "data" / subpath.parent / f"{model_path.stem}.npz"
    return data_path


def get_benchmark_dir(
    onnx_path: str | Path, benchmarks_dir: str = "vnncomp2024_benchmarks"
) -> Path:
    """Find the benchmark root directory for a given ONNX file.

    Searches upward from the ONNX file path to find the benchmark directory
    that contains instances.csv.

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Path to benchmark directory
    :raises FileNotFoundError: If benchmark directory cannot be found
    """
    current = Path(onnx_path).parent

    max_depth = 5
    for _ in range(max_depth):
        if (current / "instances.csv").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    current = Path(onnx_path)
    for parent in current.parents:
        if parent.name == benchmarks_dir and parent.parent.name != benchmarks_dir:
            rel = Path(onnx_path).relative_to(parent)
            if rel.parts:
                benchmark_subdir = parent / rel.parts[0]
                if (benchmark_subdir / "instances.csv").exists():
                    return benchmark_subdir

    raise FileNotFoundError(
        f"Could not find benchmark directory with instances.csv for {onnx_path}"
    )


def find_benchmark_folders(base_dir: str) -> list[str]:
    """Find all benchmark directories (backward compatibility alias).

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths as strings
    """
    return [str(b) for b in find_benchmarks(base_dir)]


def find_onnx_files_from_instances(benchmark_dirs: list[str], num_limit: int = 20) -> list[str]:
    """Find ONNX files from instances.csv (backward compatibility alias).

    :param benchmark_dirs: List of benchmark directory paths as strings
    :param num_limit: Maximum ONNX files per benchmark directory
    :return: List of ONNX file paths as strings
    """
    benchmarks = [Path(d) for d in benchmark_dirs]
    models = find_models(benchmarks, num_limit)
    return [str(m) for m in models]


def find_vnnlib_files_from_instances(benchmark_dirs: list[str]) -> list[str]:
    """Find VNNLib files from instances.csv (backward compatibility alias).

    :param benchmark_dirs: List of benchmark directory paths as strings
    :return: List of VNNLib file paths as strings
    """
    benchmarks = [Path(d) for d in benchmark_dirs]
    vnnlib_files = find_vnnlib_files(benchmarks)
    return [str(v) for v in vnnlib_files]


def get_benchmark_name(onnx_path: str, benchmarks_dir: str = "vnncomp2024_benchmarks") -> str:
    """Extract benchmark name from ONNX file path (backward compatibility alias).

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Benchmark name (subdirectory name)
    """
    return get_model_benchmark_name(Path(onnx_path), benchmarks_dir)
