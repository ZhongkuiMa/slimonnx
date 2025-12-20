"""Build test data efficiently from vnncomp2024_benchmarks symlink.

Optimizations:
- Only uses FIRST vnnlib per model (faster generation)
- Limits to max_per_benchmark models (configurable)
- Works with symlink structure (vnncomp2024_benchmarks)
- Direct generation to data/ (no intermediate vnnlib_data)
"""

import csv
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort


def get_models_and_vnnlibs(benchmark_dir: Path, max_models: int = 20):
    """Get list of models and their first vnnlib file from instances.csv.

    :param benchmark_dir: Benchmark directory path
    :param max_models: Maximum number of unique models to process
    :return: List of (onnx_path, vnnlib_path) tuples
    """
    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.exists():
        return []

    model_to_vnnlib = {}

    with instances_csv.open() as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) < 2:
                continue

            onnx_path = row[0].strip()
            vnnlib_path = row[1].strip()

            # Only keep FIRST vnnlib per model
            if onnx_path not in model_to_vnnlib:
                model_to_vnnlib[onnx_path] = vnnlib_path

                if len(model_to_vnnlib) >= max_models:
                    break

    # Return as list of tuples
    return [
        (benchmark_dir / onnx, benchmark_dir / vnnlib) for onnx, vnnlib in model_to_vnnlib.items()
    ]


def _get_model_input_info(onnx_path: Path) -> tuple[str, list[int]] | None:
    """Get model input name and expected shape.

    :param onnx_path: Path to ONNX model
    :return: Tuple of (input_name, expected_shape) or None if invalid
    """
    model = onnx.load(str(onnx_path))
    input_names = [
        inp.name
        for inp in model.graph.input
        if not any(init.name == inp.name for init in model.graph.initializer)
    ]

    if not input_names:
        return None

    input_name = input_names[0]
    input_tensor = next((inp for inp in model.graph.input if inp.name == input_name), None)

    if not input_tensor:
        return None

    expected_shape = [
        dim.dim_value if dim.dim_value > 0 else 1 for dim in input_tensor.type.tensor_type.shape.dim
    ]

    return input_name, expected_shape


def _process_bounds(
    npz_files: list[Path],
    session: ort.InferenceSession,
    input_name: str,
    expected_shape: list[int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Process VNNLib bounds and generate inputs/outputs.

    :param npz_files: List of npz files to process
    :param session: ONNX Runtime session
    :param input_name: Model input name
    :param expected_shape: Expected input shape
    :return: Tuple of (lower_inputs, lower_outputs, upper_inputs, upper_outputs)
    """
    lower_inputs = []
    lower_outputs = []
    upper_inputs = []
    upper_outputs = []

    for npz_file in npz_files[:2]:  # Limit to 2 sub-properties
        data = np.load(npz_file)
        if "input" not in data:
            continue

        input_bounds = data["input"]
        if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
            continue

        # Lower bounds
        try:
            input_array = input_bounds[:, 0].astype(np.float32).reshape(expected_shape)
            output = session.run(None, {input_name: input_array})
            lower_inputs.append(input_array)
            lower_outputs.append(output[0])
        except (ValueError, RuntimeError, KeyError):
            pass

        # Upper bounds
        try:
            input_array = input_bounds[:, 1].astype(np.float32).reshape(expected_shape)
            output = session.run(None, {input_name: input_array})
            upper_inputs.append(input_array)
            upper_outputs.append(output[0])
        except (ValueError, RuntimeError, KeyError):
            pass

    return lower_inputs, lower_outputs, upper_inputs, upper_outputs


def generate_data_from_vnnlib(
    onnx_path: Path,
    vnnlib_path: Path,
    output_dir: Path,
) -> bool:
    """Generate test data from ONNX model and VNNLib file.

    :param onnx_path: Path to ONNX model
    :param vnnlib_path: Path to VNNLib file
    :param output_dir: Output directory for .npz file
    :return: True if successful, False otherwise
    """
    try:
        # Load model and get input info
        input_info = _get_model_input_info(onnx_path)
        if not input_info:
            return False

        input_name, expected_shape = input_info

        # Convert VNNLib to numpy using torchvnnlib
        import tempfile

        from torchvnnlib import TorchVNNLIB

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = TorchVNNLIB(verbose=False, detect_fast_type=True, output_format="numpy")
            converter.convert(str(vnnlib_path), tmpdir)

            npz_files = list(Path(tmpdir).rglob("*.npz"))
            if not npz_files:
                return False

            # Create ONNX Runtime session
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

            # Process VNNLib bounds
            lower_inputs, lower_outputs, upper_inputs, upper_outputs = _process_bounds(
                npz_files, session, input_name, expected_shape
            )

            if not lower_inputs and not upper_inputs:
                return False

            # Save results
            vnnlib_name = vnnlib_path.stem
            results: dict[str, dict[str, dict[str, list]]] = {vnnlib_name: {}}

            if lower_inputs:
                results[vnnlib_name]["lower"] = {
                    "inputs": lower_inputs,
                    "outputs": lower_outputs,
                }

            if upper_inputs:
                results[vnnlib_name]["upper"] = {
                    "inputs": upper_inputs,
                    "outputs": upper_outputs,
                }

            output_file = output_dir / f"{onnx_path.stem}.npz"
            np.savez_compressed(output_file, **cast(dict[str, Any], results))

            return True

    except (OSError, ValueError, RuntimeError, AttributeError, KeyError) as e:
        print(f"  Error processing {onnx_path.name}: {e}")
        return False


def build_test_data(
    benchmarks_root: str = "vnncomp2024_benchmarks",
    data_root: str = "data",
    max_per_benchmark: int = 20,
):
    """Build test data from benchmarks.

    :param benchmarks_root: Root directory with benchmarks (symlink)
    :param data_root: Output directory for test data
    :param max_per_benchmark: Maximum models per benchmark
    """
    benchmarks_path = Path(benchmarks_root)
    data_path = Path(data_root)

    if not benchmarks_path.exists():
        print(f"Error: {benchmarks_root} not found")
        return

    # Create data directory
    data_path.mkdir(exist_ok=True)

    # Get all benchmark directories
    benchmark_dirs = sorted(
        [d for d in benchmarks_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    print(f"Building test data from {len(benchmark_dirs)} benchmarks")
    print(f"Max {max_per_benchmark} models per benchmark")
    print("Using FIRST vnnlib per model only")
    print("=" * 70)

    total_models = 0
    total_success = 0
    start_time = time.perf_counter()

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        # Get models and vnnlibs
        model_vnnlib_pairs = get_models_and_vnnlibs(benchmark_dir, max_per_benchmark)

        if not model_vnnlib_pairs:
            continue

        # Create output directory
        output_dir = data_path / benchmark_name
        output_dir.mkdir(exist_ok=True)

        success = 0
        for onnx_path, vnnlib_path in model_vnnlib_pairs:
            total_models += 1

            if generate_data_from_vnnlib(onnx_path, vnnlib_path, output_dir):
                success += 1
                total_success += 1

        print(f"[{benchmark_name}] Processed {success}/{len(model_vnnlib_pairs)} models")

    elapsed = time.perf_counter() - start_time

    print("=" * 70)
    print(f"Total: {total_success}/{total_models} models processed")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nData directory: {data_path.absolute()}")
    print(f"Size: {sum(f.stat().st_size for f in data_path.rglob('*.npz')) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import sys

    max_models = 20
    if len(sys.argv) > 1:
        max_models = int(sys.argv[1])

    print(f"\nBuilding test data with max_per_benchmark={max_models}")
    build_test_data(max_per_benchmark=max_models)
