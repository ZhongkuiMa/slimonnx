"""Inspect ONNX models to collect metadata and diagnose issues."""

__docformat__ = "restructuredtext"
__all__ = ["inspect_model", "inspect_all_models"]

from pathlib import Path

import onnx

DTYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    6: "int32",
    7: "int64",
    11: "float64",
}


def _get_tensor_shape(tensor_type) -> tuple[list[int], bool]:
    """Extract shape and dynamic flag from tensor type.

    :param tensor_type: ONNX tensor type
    :return: Tuple of (shape list, has_dynamic flag)
    """
    shape = []
    has_dynamic = False
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(-1)
            has_dynamic = True
    return shape, has_dynamic


def _parse_instances_csv(csv_path: Path) -> set[str]:
    """Parse instances.csv and return unique model paths.

    :param csv_path: Path to instances.csv
    :return: Set of unique model paths
    """
    unique_models = set()
    try:
        with open(csv_path) as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        unique_models.add(parts[0].strip())
    except Exception:
        pass
    return unique_models


def inspect_model(onnx_path: str) -> dict | None:
    """Inspect a single ONNX model and collect metadata.

    :param onnx_path: Path to ONNX model file
    :return: Dictionary with model metadata, or None if error
    """
    try:
        model = onnx.load(onnx_path)

        ir_version = model.ir_version
        opset_version = model.opset_import[0].version if model.opset_import else None
        producer_name = model.producer_name if model.producer_name else "Unknown"

        inputs_info = []
        for inp in model.graph.input:
            if any(init.name == inp.name for init in model.graph.initializer):
                continue

            shape, has_dynamic = _get_tensor_shape(inp.type.tensor_type)
            dtype = DTYPE_MAP.get(
                inp.type.tensor_type.elem_type,
                f"unknown({inp.type.tensor_type.elem_type})",
            )

            total_elements = 1
            for dim in shape:
                if dim > 0:
                    total_elements *= dim

            inputs_info.append(
                {
                    "name": inp.name,
                    "shape": shape,
                    "dtype": dtype,
                    "has_dynamic": has_dynamic,
                    "total_elements": total_elements if not has_dynamic else -1,
                }
            )

        outputs_info = []
        for out in model.graph.output:
            shape, has_dynamic = _get_tensor_shape(out.type.tensor_type)
            dtype = DTYPE_MAP.get(
                out.type.tensor_type.elem_type,
                f"unknown({out.type.tensor_type.elem_type})",
            )

            outputs_info.append(
                {
                    "name": out.name,
                    "shape": shape,
                    "dtype": dtype,
                    "has_dynamic": has_dynamic,
                }
            )

        op_types = {node.op_type for node in model.graph.node}

        initializer_dtypes = {
            DTYPE_MAP.get(init.data_type, f"unknown({init.data_type})")
            for init in model.graph.initializer
        }

        return {
            "path": onnx_path,
            "ir_version": ir_version,
            "opset_version": opset_version,
            "producer": producer_name,
            "inputs": inputs_info,
            "outputs": outputs_info,
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
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

    benchmark_dirs = sorted([d for d in benchmarks_path.iterdir() if d.is_dir()])

    if not benchmark_dirs:
        print(f"No benchmark directories found in {benchmarks_dir}")
        return

    print(f"Inspecting ONNX models in {len(benchmark_dirs)} benchmarks")
    print("=" * 80)

    total_inspected = 0
    total_failed = 0
    high_ir_version = []
    high_opset_version = []
    dynamic_shapes = []
    float64_models = []

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        if benchmark_name.startswith("."):
            continue

        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            continue

        unique_models = _parse_instances_csv(instances_csv)
        if not unique_models:
            continue

        unique_models_list = sorted(unique_models)[:max_per_benchmark]

        print(f"\n[{benchmark_name}] Inspecting {len(unique_models_list)} models...")

        for model_rel_path in unique_models_list:
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

            print(f"\n  {model_name}:")
            print(f"    IR version: {info['ir_version']}")
            print(f"    Opset version: {info['opset_version']}")
            print(f"    Producer: {info['producer']}")
            print(
                f"    Nodes: {info['node_count']}, Initializers: {info['initializer_count']}"
            )

            for inp in info["inputs"]:
                shape_str = str(inp["shape"]).replace("-1", "dynamic")
                print(f"    Input '{inp['name']}': {shape_str} {inp['dtype']}")
                if inp["has_dynamic"]:
                    dynamic_shapes.append(f"{benchmark_name}/{model_name}")

            for out in info["outputs"]:
                shape_str = str(out["shape"]).replace("-1", "dynamic")
                print(f"    Output '{out['name']}': {shape_str} {out['dtype']}")

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
