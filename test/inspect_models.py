"""Inspect ONNX models to collect metadata and diagnose issues.

This module analyzes ONNX models to help diagnose why some models
fail during processing (extract_inputs.py, cal_outputs.py).
"""

__docformat__ = "restructuredtext"
__all__ = ["inspect_model", "inspect_all_models"]

import os
from pathlib import Path

import numpy as np
import onnx


def inspect_model(onnx_path: str) -> dict | None:
    """Inspect a single ONNX model and collect metadata.

    :param onnx_path: Path to ONNX model file
    :return: Dictionary with model metadata, or None if error
    """
    try:
        model = onnx.load(onnx_path)

        # Basic model info
        ir_version = model.ir_version
        opset_version = model.opset_import[0].version if model.opset_import else None
        producer_name = model.producer_name if model.producer_name else "Unknown"

        # Input information
        inputs_info = []
        for inp in model.graph.input:
            # Skip initializers
            if any(init.name == inp.name for init in model.graph.initializer):
                continue

            input_name = inp.name

            # Extract shape
            shape = []
            has_dynamic = False
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
                    has_dynamic = True

            # Extract dtype
            dtype_int = inp.type.tensor_type.elem_type
            dtype_map = {
                1: "float32",
                2: "uint8",
                3: "int8",
                6: "int32",
                7: "int64",
                11: "float64",
            }
            dtype = dtype_map.get(dtype_int, f"unknown({dtype_int})")

            # Calculate total elements (excluding dynamic dims)
            total_elements = 1
            for dim in shape:
                if dim > 0:
                    total_elements *= dim

            inputs_info.append(
                {
                    "name": input_name,
                    "shape": shape,
                    "dtype": dtype,
                    "has_dynamic": has_dynamic,
                    "total_elements": total_elements if not has_dynamic else -1,
                }
            )

        # Output information
        outputs_info = []
        for out in model.graph.output:
            output_name = out.name

            # Extract shape
            shape = []
            has_dynamic = False
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
                    has_dynamic = True

            # Extract dtype
            dtype_int = out.type.tensor_type.elem_type
            dtype_map = {
                1: "float32",
                2: "uint8",
                3: "int8",
                6: "int32",
                7: "int64",
                11: "float64",
            }
            dtype = dtype_map.get(dtype_int, f"unknown({dtype_int})")

            outputs_info.append(
                {
                    "name": output_name,
                    "shape": shape,
                    "dtype": dtype,
                    "has_dynamic": has_dynamic,
                }
            )

        # Graph information
        node_count = len(model.graph.node)
        initializer_count = len(model.graph.initializer)

        # Collect unique operator types
        op_types = set(node.op_type for node in model.graph.node)

        # Check initializer dtypes
        initializer_dtypes = set()
        for init in model.graph.initializer:
            dtype_int = init.data_type
            dtype_map = {
                1: "float32",
                2: "uint8",
                3: "int8",
                6: "int32",
                7: "int64",
                11: "float64",
            }
            initializer_dtypes.add(dtype_map.get(dtype_int, f"unknown({dtype_int})"))

        return {
            "path": onnx_path,
            "ir_version": ir_version,
            "opset_version": opset_version,
            "producer": producer_name,
            "inputs": inputs_info,
            "outputs": outputs_info,
            "node_count": node_count,
            "initializer_count": initializer_count,
            "op_types": sorted(op_types),
            "initializer_dtypes": sorted(initializer_dtypes),
        }

    except Exception as e:
        print(f"Error inspecting {onnx_path}: {e}")
        return None


def inspect_all_models(
    benchmarks_dir: str = "benchmarks", max_per_benchmark: int = 20
) -> None:
    """Inspect all ONNX models in benchmarks and print summary.

    :param benchmarks_dir: Root directory containing benchmark subdirectories
    :param max_per_benchmark: Maximum number of models to inspect per benchmark
    """
    benchmarks_path = Path(benchmarks_dir)

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    # Find all benchmark directories
    benchmark_dirs = sorted([d for d in benchmarks_path.iterdir() if d.is_dir()])

    if not benchmark_dirs:
        print(f"No benchmark directories found in {benchmarks_dir}")
        return

    print(f"Inspecting ONNX models in {len(benchmark_dirs)} benchmarks")
    print("=" * 80)

    total_inspected = 0
    total_failed = 0

    # Track issues
    high_ir_version = []
    high_opset_version = []
    dynamic_shapes = []
    float64_models = []

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        # Skip hidden directories
        if benchmark_name.startswith("."):
            continue

        # Check for instances.csv
        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            continue

        # Get unique ONNX models
        unique_models = set()
        try:
            with open(instances_csv) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        model_path = parts[0].strip()
                        unique_models.add(model_path)
        except Exception:
            continue

        if not unique_models:
            continue

        # Limit number of models
        unique_models = sorted(unique_models)[:max_per_benchmark]

        print(f"\n[{benchmark_name}] Inspecting {len(unique_models)} models...")

        for model_rel_path in unique_models:
            model_path = benchmark_dir / model_rel_path
            model_name = Path(model_rel_path).stem

            if not model_path.exists():
                total_failed += 1
                continue

            info = inspect_model(str(model_path))

            if info is None:
                total_failed += 1
                continue

            total_inspected += 1

            # Print model info
            print(f"\n  {model_name}:")
            print(f"    IR version: {info['ir_version']}")
            print(f"    Opset version: {info['opset_version']}")
            print(f"    Producer: {info['producer']}")
            print(
                f"    Nodes: {info['node_count']}, Initializers: {info['initializer_count']}"
            )

            # Print inputs
            for inp in info["inputs"]:
                shape_str = str(inp["shape"]).replace("-1", "dynamic")
                print(f"    Input '{inp['name']}': {shape_str} {inp['dtype']}")
                if inp["has_dynamic"]:
                    dynamic_shapes.append(f"{benchmark_name}/{model_name}")

            # Print outputs
            for out in info["outputs"]:
                shape_str = str(out["shape"]).replace("-1", "dynamic")
                print(f"    Output '{out['name']}': {shape_str} {out['dtype']}")

            # Track potential issues
            if info["ir_version"] > 10:
                high_ir_version.append(
                    f"{benchmark_name}/{model_name} (IR {info['ir_version']})"
                )

            if info["opset_version"] and info["opset_version"] > 22:
                high_opset_version.append(
                    f"{benchmark_name}/{model_name} (opset {info['opset_version']})"
                )

            if "float64" in info["initializer_dtypes"]:
                float64_models.append(f"{benchmark_name}/{model_name}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"Total models inspected: {total_inspected}")
    print(f"Total failed: {total_failed}")

    if high_ir_version:
        print(f"\nModels with IR version > 10 ({len(high_ir_version)}):")
        for model in high_ir_version[:10]:
            print(f"  - {model}")
        if len(high_ir_version) > 10:
            print(f"  ... and {len(high_ir_version) - 10} more")

    if high_opset_version:
        print(f"\nModels with opset > 22 ({len(high_opset_version)}):")
        for model in high_opset_version[:10]:
            print(f"  - {model}")
        if len(high_opset_version) > 10:
            print(f"  ... and {len(high_opset_version) - 10} more")

    if dynamic_shapes:
        print(f"\nModels with dynamic shapes ({len(dynamic_shapes)}):")
        for model in dynamic_shapes[:10]:
            print(f"  - {model}")
        if len(dynamic_shapes) > 10:
            print(f"  ... and {len(dynamic_shapes) - 10} more")

    if float64_models:
        print(f"\nModels with float64 initializers ({len(float64_models)}):")
        for model in float64_models[:10]:
            print(f"  - {model}")
        if len(float64_models) > 10:
            print(f"  ... and {len(float64_models) - 10} more")


if __name__ == "__main__":
    inspect_all_models()
