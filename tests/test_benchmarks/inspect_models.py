"""Inspect ONNX models to collect metadata and diagnose issues."""

__docformat__ = "restructuredtext"
__all__ = ["inspect_all_models", "inspect_model"]

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

IR_VERSION_THRESHOLD = 10
OPSET_VERSION_THRESHOLD = 22
MAX_DISPLAY_ITEMS = 10


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
        with Path(csv_path).open(encoding="utf-8") as file_handle:
            for line in file_handle.readlines()[1:]:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        unique_models.add(parts[0].strip())
    except OSError as error:
        print(f"Error reading {csv_path}: {error}")
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

    except (OSError, ValueError, AttributeError) as error:
        print(f"Error inspecting {onnx_path}: {error}")
        return None


def _print_model_info(info: dict, benchmark_name: str, model_name: str) -> None:
    """Print model information.

    :param info: Model info dictionary
    :param benchmark_name: Benchmark name
    :param model_name: Model name
    """
    print(f"\n  {model_name}:")
    print(f"    IR version: {info['ir_version']}")
    print(f"    Opset version: {info['opset_version']}")
    print(f"    Producer: {info['producer']}")
    print(f"    Nodes: {info['node_count']}, Initializers: {info['initializer_count']}")

    for inp in info["inputs"]:
        shape_str = str(inp["shape"]).replace("-1", "dynamic")
        print(f"    Input '{inp['name']}': {shape_str} {inp['dtype']}")

    for out in info["outputs"]:
        shape_str = str(out["shape"]).replace("-1", "dynamic")
        print(f"    Output '{out['name']}': {shape_str} {out['dtype']}")


def _collect_model_diagnostics(
    info: dict, benchmark_name: str, model_name: str, lists: dict
) -> None:
    """Collect diagnostic information from model.

    :param info: Model info dictionary
    :param benchmark_name: Benchmark name
    :param model_name: Model name
    :param lists: Dictionary containing diagnostic lists
    """
    for inp in info["inputs"]:
        if inp["has_dynamic"]:
            lists["dynamic_shapes"].append(f"{benchmark_name}/{model_name}")
            break

    if info["ir_version"] > IR_VERSION_THRESHOLD:
        lists["high_ir_version"].append(f"{benchmark_name}/{model_name} (IR {info['ir_version']})")

    if info["opset_version"] and info["opset_version"] > OPSET_VERSION_THRESHOLD:
        lists["high_opset_version"].append(
            f"{benchmark_name}/{model_name} (opset {info['opset_version']})"
        )

    if "float64" in info["initializer_dtypes"]:
        lists["float64_models"].append(f"{benchmark_name}/{model_name}")


def _print_model_list(title: str, model_list: list) -> None:
    """Print a list of models with truncation.

    :param title: Section title
    :param model_list: List of model identifiers
    """
    if not model_list:
        return

    print(f"\n{title} ({len(model_list)}):")
    for model in model_list[:MAX_DISPLAY_ITEMS]:
        print(f"  - {model}")
    if len(model_list) > MAX_DISPLAY_ITEMS:
        print(f"  ... and {len(model_list) - MAX_DISPLAY_ITEMS} more")


def _print_diagnostic_summary(
    total_inspected: int,
    total_failed: int,
    high_ir_version: list,
    high_opset_version: list,
    dynamic_shapes: list,
    float64_models: list,
) -> None:
    """Print diagnostic summary.

    :param total_inspected: Number of models inspected
    :param total_failed: Number of failed models
    :param high_ir_version: Models with high IR version
    :param high_opset_version: Models with high opset version
    :param dynamic_shapes: Models with dynamic shapes
    :param float64_models: Models with float64 initializers
    """
    print("\n" + "=" * 80)
    print(f"Total models inspected: {total_inspected}")
    print(f"Total failed: {total_failed}")

    _print_model_list(f"Models with IR version > {IR_VERSION_THRESHOLD}", high_ir_version)
    _print_model_list(f"Models with opset > {OPSET_VERSION_THRESHOLD}", high_opset_version)
    _print_model_list("Models with dynamic shapes", dynamic_shapes)
    _print_model_list("Models with float64 initializers", float64_models)


def inspect_all_models(benchmarks_dir: str = "benchmarks", max_per_benchmark: int = 20) -> None:
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
    diagnostic_lists: dict[str, list[str]] = {
        "high_ir_version": [],
        "high_opset_version": [],
        "dynamic_shapes": [],
        "float64_models": [],
    }

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

            _print_model_info(info, benchmark_name, model_name)
            _collect_model_diagnostics(info, benchmark_name, model_name, diagnostic_lists)

    _print_diagnostic_summary(
        total_inspected,
        total_failed,
        diagnostic_lists["high_ir_version"],
        diagnostic_lists["high_opset_version"],
        diagnostic_lists["dynamic_shapes"],
        diagnostic_lists["float64_models"],
    )


if __name__ == "__main__":
    inspect_all_models()
