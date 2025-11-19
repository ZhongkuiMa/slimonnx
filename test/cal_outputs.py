"""Calculate and store input-output pairs for ONNX models.

This module runs ONNX models with torchvnnlib inputs via ONNX Runtime
and saves the input-output pairs for testing purposes.
"""

__docformat__ = "restructuredtext"
__all__ = ["calculate_model_outputs", "calculate_all_outputs"]

import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from utils import get_benchmark_name


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
    :return: Nested dict {vnnlib_name: {"lower": {"inputs": [...], "outputs": [...]}, "upper": {...}}} or None
    """
    # Parse instances.csv to find all VNNLib properties for this model
    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.exists():
        return None

    onnx_rel_path = os.path.relpath(onnx_path, benchmark_dir)
    vnnlib_names = []

    try:
        with open(instances_csv) as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    model_path = parts[0].strip()
                    # Normalize paths for comparison (remove ./ prefix)
                    model_path_norm = model_path.lstrip("./")
                    onnx_rel_path_norm = onnx_rel_path.lstrip("./")
                    if model_path_norm == onnx_rel_path_norm:
                        vnnlib_path = parts[1].strip()
                        vnnlib_name = Path(vnnlib_path).stem
                        if vnnlib_name not in vnnlib_names:
                            vnnlib_names.append(vnnlib_name)
    except Exception as e:
        print(f"  Error reading instances.csv: {e}")
        return None

    if not vnnlib_names:
        return None

    # Load ONNX model and get input info
    try:
        model = onnx.load(onnx_path)
        input_names = [
            inp.name
            for inp in model.graph.input
            if not any(init.name == inp.name for init in model.graph.initializer)
        ]
        if not input_names:
            print(f"  No inputs found in model")
            return None
        input_name = input_names[0]

        # Get expected input shape
        input_tensor = None
        for inp in model.graph.input:
            if inp.name == input_name:
                input_tensor = inp
                break

        if input_tensor is None:
            print(f"  Cannot find input tensor info")
            return None

        # Extract shape from tensor type
        expected_shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                expected_shape.append(dim.dim_value)
            else:
                expected_shape.append(1)  # Use 1 for dynamic dimensions

    except Exception as e:
        print(f"  Error loading model: {e}")
        return None

    # Create ONNX Runtime session
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  Error creating ONNX Runtime session: {e}")
        return None

    # Process each VNNLib property
    results = {}

    for vnnlib_name in vnnlib_names:
        vnnlib_dir = torchvnnlib_dir / vnnlib_name / "or_group_0"

        if not vnnlib_dir.exists():
            continue

        # Load all sub_prop_*.pth files
        pth_files = sorted(vnnlib_dir.glob("sub_prop_*.pth"))
        if not pth_files:
            continue

        lower_inputs = []
        lower_outputs = []
        upper_inputs = []
        upper_outputs = []

        for pth_file in pth_files:
            try:
                # Load input bounds
                data = torch.load(pth_file, weights_only=True)
                input_bounds = data["input"]  # Shape: (n_inputs, 2)

                # Extract lower and upper bounds
                lower = input_bounds[:, 0]
                upper = input_bounds[:, 1]

                # Process lower bound
                try:
                    lower_array = (
                        lower.numpy().astype(np.float32).reshape(expected_shape)
                    )
                    lower_output = session.run(None, {input_name: lower_array})
                    lower_inputs.append(torch.from_numpy(lower_array))
                    lower_outputs.append(torch.from_numpy(lower_output[0]))
                except Exception as e:
                    print(f"  Error processing lower bound in {pth_file.name}: {e}")

                # Process upper bound
                try:
                    upper_array = (
                        upper.numpy().astype(np.float32).reshape(expected_shape)
                    )
                    upper_output = session.run(None, {input_name: upper_array})
                    upper_inputs.append(torch.from_numpy(upper_array))
                    upper_outputs.append(torch.from_numpy(upper_output[0]))
                except Exception as e:
                    print(f"  Error processing upper bound in {pth_file.name}: {e}")

            except Exception as e:
                print(f"  Error loading {pth_file.name}: {e}")
                continue

        if (lower_inputs and lower_outputs) or (upper_inputs and upper_outputs):
            results[vnnlib_name] = {}
            if lower_inputs and lower_outputs:
                results[vnnlib_name]["lower"] = {
                    "inputs": lower_inputs,
                    "outputs": lower_outputs,
                }
            if upper_inputs and upper_outputs:
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

    # Find all benchmark directories
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

        # Skip hidden directories
        if benchmark_name.startswith("."):
            continue

        # Check for instances.csv
        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            print(f"[{benchmark_name}] No instances.csv, skipping")
            continue

        # Check for torchvnnlib directory
        torchvnnlib_dir = benchmark_dir / "torchvnnlib"
        if not torchvnnlib_dir.exists():
            print(f"[{benchmark_name}] No torchvnnlib directory, skipping")
            continue

        # Get unique ONNX models from instances.csv
        unique_models = set()
        try:
            with open(instances_csv) as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        model_path = parts[0].strip()
                        unique_models.add(model_path)
        except Exception as e:
            print(f"[{benchmark_name}] Error reading instances.csv: {e}")
            continue

        if not unique_models:
            print(f"[{benchmark_name}] No models in instances.csv")
            continue

        # Limit number of models
        unique_models = sorted(unique_models)[:max_per_benchmark]

        # Create data directory
        data_dir = benchmark_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Process each unique model
        success = 0
        failed = []

        for model_rel_path in unique_models:
            model_path = benchmark_dir / model_rel_path
            model_name = Path(model_rel_path).stem

            if not model_path.exists():
                failed.append((model_name, "File not found"))
                continue

            # Calculate outputs
            try:
                results = calculate_model_outputs(
                    str(model_path), benchmark_dir, torchvnnlib_dir
                )

                if results:
                    # Save to data directory
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
            for name, error in failed[:3]:  # Show first 3 errors
                print(f"  {name}: {error}")
            if len(failed) > 3:
                print(f"  ... and {len(failed)-3} more")

    total_time = time.perf_counter() - start_time
    print("=" * 60)
    print(f"Total: {total_success} processed, {total_failed} failed")
    print(f"Time: {total_time:.2f}s")


if __name__ == "__main__":
    calculate_all_outputs()
