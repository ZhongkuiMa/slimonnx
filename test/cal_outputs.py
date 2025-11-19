"""Calculate and store input-output pairs for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = ["calculate_model_outputs", "calculate_all_outputs"]

import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from utils import get_benchmark_name


def _parse_model_vnnlib_mapping(csv_path: Path, onnx_rel_path: str) -> list[str] | None:
    """Parse instances.csv and return VNNLib names for a model.

    :param csv_path: Path to instances.csv
    :param onnx_rel_path: Relative path to ONNX model
    :return: List of VNNLib names or None if error
    """
    vnnlib_names = []
    try:
        with open(csv_path) as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        model_path_norm = parts[0].strip().lstrip("./")
                        onnx_rel_path_norm = onnx_rel_path.lstrip("./")
                        if model_path_norm == onnx_rel_path_norm:
                            vnnlib_name = Path(parts[1].strip()).stem
                            if vnnlib_name not in vnnlib_names:
                                vnnlib_names.append(vnnlib_name)
    except Exception as e:
        print(f"  Error reading instances.csv: {e}")
        return None
    return vnnlib_names if vnnlib_names else None


def _get_model_input_info(onnx_path: str) -> tuple[str, list[int]] | None:
    """Get input name and expected shape from ONNX model.

    :param onnx_path: Path to ONNX model file
    :return: Tuple of (input_name, expected_shape) or None if error
    """
    try:
        model = onnx.load(onnx_path)
        input_names = [
            inp.name
            for inp in model.graph.input
            if not any(init.name == inp.name for init in model.graph.initializer)
        ]
        if not input_names:
            print("  No inputs found in model")
            return None

        input_name = input_names[0]
        input_tensor = next(
            (inp for inp in model.graph.input if inp.name == input_name), None
        )

        if input_tensor is None:
            print("  Cannot find input tensor info")
            return None

        expected_shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in input_tensor.type.tensor_type.shape.dim
        ]

        return input_name, expected_shape

    except Exception as e:
        print(f"  Error loading model: {e}")
        return None


def _process_bound(
    bound: torch.Tensor,
    expected_shape: list[int],
    session: ort.InferenceSession,
    input_name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Process input bound and compute output.

    :param bound: Input bound tensor
    :param expected_shape: Expected input shape
    :param session: ONNX Runtime session
    :param input_name: Input tensor name
    :return: Tuple of (input_tensor, output_tensor) or None if error
    """
    try:
        input_array = bound.numpy().astype(np.float32).reshape(expected_shape)
        output = session.run(None, {input_name: input_array})
        return torch.from_numpy(input_array), torch.from_numpy(output[0])
    except Exception:
        return None


def calculate_model_outputs(
    onnx_path: str,
    benchmark_dir: Path,
    torchvnnlib_dir: Path,
) -> dict[str, dict[str, dict[str, list]]] | None:
    """Calculate outputs for one ONNX model across all its VNNLib properties.

    Uses both lower and upper bounds from torchvnnlib as separate test inputs.

    :param onnx_path: Path to ONNX model file
    :param benchmark_dir: Benchmark directory containing instances.csv
    :param torchvnnlib_dir: Directory containing torchvnnlib inputs
    :return: Nested dict structure or None if error
    """
    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.exists():
        return None

    onnx_rel_path = str(Path(onnx_path).relative_to(benchmark_dir))
    vnnlib_names = _parse_model_vnnlib_mapping(instances_csv, onnx_rel_path)

    if not vnnlib_names:
        return None

    input_info = _get_model_input_info(onnx_path)
    if not input_info:
        return None

    input_name, expected_shape = input_info

    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  Error creating ONNX Runtime session: {e}")
        return None

    results = {}

    for vnnlib_name in vnnlib_names:
        vnnlib_dir = torchvnnlib_dir / vnnlib_name / "or_group_0"

        if not vnnlib_dir.exists():
            continue

        pth_files = sorted(vnnlib_dir.glob("sub_prop_*.pth"))
        if not pth_files:
            continue

        lower_inputs = []
        lower_outputs = []
        upper_inputs = []
        upper_outputs = []

        for pth_file in pth_files:
            try:
                data = torch.load(pth_file, weights_only=True)
                input_bounds = data["input"]

                lower_result = _process_bound(
                    input_bounds[:, 0], expected_shape, session, input_name
                )
                if lower_result:
                    lower_inputs.append(lower_result[0])
                    lower_outputs.append(lower_result[1])

                upper_result = _process_bound(
                    input_bounds[:, 1], expected_shape, session, input_name
                )
                if upper_result:
                    upper_inputs.append(upper_result[0])
                    upper_outputs.append(upper_result[1])

            except Exception as e:
                print(f"  Error loading {pth_file.name}: {e}")

        if lower_inputs or upper_inputs:
            results[vnnlib_name] = {}
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

    return results if results else None


def calculate_all_outputs(
    benchmarks_dir: str = "benchmarks", max_per_benchmark: int = 20
) -> None:
    """Calculate outputs for all ONNX models in benchmarks.

    :param benchmarks_dir: Root directory containing benchmark subdirectories
    :param max_per_benchmark: Maximum number of unique models per benchmark
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
    start_time = time.perf_counter()

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        if benchmark_name.startswith("."):
            continue

        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            print(f"[{benchmark_name}] No instances.csv, skipping")
            continue

        torchvnnlib_dir = benchmark_dir / "torchvnnlib"
        if not torchvnnlib_dir.exists():
            print(f"[{benchmark_name}] No torchvnnlib directory, skipping")
            continue

        unique_models = set()
        try:
            with open(instances_csv) as f:
                for line in f.readlines()[1:]:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            unique_models.add(parts[0].strip())
        except Exception as e:
            print(f"[{benchmark_name}] Error reading instances.csv: {e}")
            continue

        if not unique_models:
            print(f"[{benchmark_name}] No models in instances.csv")
            continue

        unique_models_list = sorted(unique_models)[:max_per_benchmark]

        data_dir = benchmark_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = []

        for model_rel_path in unique_models_list:
            model_path = benchmark_dir / model_rel_path
            model_name = Path(model_rel_path).stem

            if not model_path.exists():
                failed.append((model_name, "File not found"))
                continue

            try:
                results = calculate_model_outputs(
                    str(model_path), benchmark_dir, torchvnnlib_dir
                )

                if results:
                    output_file = data_dir / f"{model_name}.pth"
                    torch.save(results, output_file)
                    success += 1
                else:
                    failed.append((model_name, "No results"))

            except Exception as e:
                failed.append((model_name, str(e)))

        total_success += success
        total_failed += len(failed)

        print(f"[{benchmark_name}] Processed {success} models, {len(failed)} failed")
        if failed:
            for name, error in failed[:3]:
                print(f"  {name}: {error}")
            if len(failed) > 3:
                print(f"  ... and {len(failed) - 3} more")

    total_time = time.perf_counter() - start_time
    print("=" * 60)
    print(f"Total: {total_success} processed, {total_failed} failed")
    print(f"Time: {total_time:.2f}s")


if __name__ == "__main__":
    calculate_all_outputs()
