"""Build complete benchmark dataset from VNN-COMP source.

This script copies benchmarks with their original structure preserved.
Only files referenced in instances.csv are copied, and the structure
matches the original vnncomp2024_benchmarks layout.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "copy_benchmarks",
    "extract_inputs",
    "calculate_outputs",
    "build_all",
]

import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from torchvnnlib import TorchVNNLIB


def copy_benchmarks(
    source_dir: str = "../../../vnncomp2024_benchmarks/benchmarks",
    target_dir: str = "benchmarks",
    max_onnx_per_benchmark: int = 20,
    max_vnnlib_per_onnx: int = 2,
) -> tuple[int, int, int]:
    """Copy ONNX models and VNNLib files preserving original structure.

    Uses instances.csv to maintain correct ONNX-VNNLib pairings.
    Creates filtered instances.csv containing only copied file pairs.
    All files maintain their original directory structure from source.

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

    target.mkdir(parents=True, exist_ok=True)

    benchmark_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    print(f"Found {len(benchmark_dirs)} benchmark directories\n")

    total_onnx = 0
    total_vnnlib = 0
    total_benchmarks = 0

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        if benchmark_name.startswith("."):
            continue

        instances_csv = benchmark_dir / "instances.csv"
        if not instances_csv.exists():
            print(f"[{benchmark_name}] No instances.csv, skipping")
            continue

        try:
            onnx_to_vnnlib = {}
            with open(instances_csv) as f:
                for line in f.readlines()[1:]:
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
        except (IOError, OSError) as error:
            print(f"[{benchmark_name}] Error parsing instances.csv: {error}, skipping")
            continue

        if not onnx_to_vnnlib:
            print(f"[{benchmark_name}] Empty instances.csv, skipping")
            continue

        target_benchmark_dir = target / benchmark_name
        target_benchmark_dir.mkdir(parents=True, exist_ok=True)

        all_onnx_models = sorted(onnx_to_vnnlib.keys())
        selected_onnx_models = all_onnx_models[:max_onnx_per_benchmark]

        filtered_instances = []
        onnx_copied = 0
        vnnlib_copied = 0

        for onnx_path in selected_onnx_models:
            source_onnx = benchmark_dir / onnx_path
            target_onnx = target_benchmark_dir / onnx_path

            if source_onnx.exists():
                target_onnx.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_onnx, target_onnx)
                onnx_copied += 1

                vnnlib_list = onnx_to_vnnlib[onnx_path][:max_vnnlib_per_onnx]

                for vnnlib_path, timeout in vnnlib_list:
                    source_vnnlib = benchmark_dir / vnnlib_path
                    target_vnnlib = target_benchmark_dir / vnnlib_path

                    if source_vnnlib.exists():
                        target_vnnlib.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_vnnlib, target_vnnlib)
                        vnnlib_copied += 1

                        filtered_instances.append(
                            f"{onnx_path},{vnnlib_path},{timeout}"
                        )
                    else:
                        print(
                            f"[{benchmark_name}] Warning: VNNLib not found: {vnnlib_path}"
                        )
            else:
                print(f"[{benchmark_name}] Warning: ONNX not found: {onnx_path}")

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


def extract_inputs(benchmarks_dir: str = "benchmarks") -> tuple[int, int]:
    """Extract VNNLib input constraints to NumPy format.

    Uses instances.csv to discover VNNLib files and converts each unique file.

    :param benchmarks_dir: Root directory containing benchmark subdirectories
    :return: Tuple of (num_success, num_failed)
    """
    benchmarks_path = Path(benchmarks_dir)

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    benchmark_dirs = sorted([d for d in benchmarks_path.iterdir() if d.is_dir()])

    if not benchmark_dirs:
        print(f"No benchmark directories found in {benchmarks_dir}")
        return 0, 0

    print(f"Extracting inputs from {len(benchmark_dirs)} benchmarks")
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
            unique_vnnlibs = set()
            with open(instances_csv) as f:
                for line in f.readlines()[1:]:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            unique_vnnlibs.add(parts[1].strip())
        except (IOError, OSError) as error:
            print(f"[{benchmark_name}] Error parsing instances.csv: {error}")
            continue

        if not unique_vnnlibs:
            print(f"[{benchmark_name}] No VNNLib files in instances.csv")
            continue

        vnnlib_data_dir = benchmark_dir / "vnnlib_data"
        vnnlib_data_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = []

        for vnnlib_rel_path in sorted(unique_vnnlibs):
            vnnlib_file = benchmark_dir / vnnlib_rel_path
            vnnlib_name = Path(vnnlib_rel_path).stem

            if not vnnlib_file.exists():
                failed.append((vnnlib_name, "File not found"))
                continue

            vnnlib_rel_path_no_suffix = (
                str(vnnlib_rel_path)
                .replace(".vnnlib", "")
                .replace("vnnlib/", "")
                .replace("vnnlib\\", "")
            )
            output_dir = vnnlib_data_dir / vnnlib_rel_path_no_suffix
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                converter = TorchVNNLIB(
                    verbose=False, detect_fast_type=True, output_format="numpy"
                )
                converter.convert(str(vnnlib_file), str(output_dir))
                success += 1
            except (IOError, OSError, ValueError, RuntimeError) as error:
                failed.append((vnnlib_name, str(error)))

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

    return total_success, total_failed


def calculate_outputs(
    benchmarks_dir: str = "benchmarks", max_per_benchmark: int = 20
) -> tuple[int, int]:
    """Calculate input-output pairs for all ONNX models.

    Uses both lower and upper bounds from vnnlib data as separate test inputs.

    :param benchmarks_dir: Root directory containing benchmark subdirectories
    :param max_per_benchmark: Maximum number of unique models per benchmark
    :return: Tuple of (num_success, num_failed)
    """
    benchmarks_path = Path(benchmarks_dir)

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    benchmark_dirs = sorted([d for d in benchmarks_path.iterdir() if d.is_dir()])

    if not benchmark_dirs:
        print(f"No benchmark directories found in {benchmarks_dir}")
        return 0, 0

    print(f"Calculating outputs for {len(benchmark_dirs)} benchmarks")
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

        unique_models = set()
        try:
            with open(instances_csv) as f:
                for line in f.readlines()[1:]:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            unique_models.add(parts[0].strip())
        except (IOError, OSError) as error:
            print(f"[{benchmark_name}] Error reading instances.csv: {error}")
            continue

        if not unique_models:
            print(f"[{benchmark_name}] No models in instances.csv")
            continue

        unique_models_list = sorted(unique_models)[:max_per_benchmark]

        success = 0
        failed = []

        for model_rel_path in unique_models_list:
            model_path = benchmark_dir / model_rel_path
            model_name = Path(model_rel_path).stem

            model_rel_path_no_suffix = (
                str(model_rel_path)
                .replace(".onnx", "")
                .replace("onnx/", "")
                .replace("onnx\\", "")
            )
            model_rel_path_no_suffix = Path(model_rel_path_no_suffix).parent
            data_dir = benchmark_dir / "data" / model_rel_path_no_suffix
            data_dir.mkdir(parents=True, exist_ok=True)

            vnnlib_data_dir = benchmark_dir / "vnnlib_data" / model_rel_path_no_suffix

            if not model_path.exists():
                failed.append((model_name, "File not found"))
                continue

            try:
                results = _calculate_model_outputs(
                    str(model_path), benchmark_dir, vnnlib_data_dir
                )

                if results:
                    output_file = data_dir / f"{model_name}.npz"
                    np.savez_compressed(output_file, **results)
                    success += 1
                else:
                    failed.append((model_name, "No results"))

            except (IOError, OSError, ValueError, RuntimeError) as error:
                failed.append((model_name, str(error)))

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

    return total_success, total_failed


def _calculate_model_outputs(
    onnx_path: str,
    benchmark_dir: Path,
    data_dir: Path,
) -> dict[str, dict[str, dict[str, list]]] | None:
    """Calculate outputs for one ONNX model across all its VNNLib properties.

    :param onnx_path: Path to ONNX model file
    :param benchmark_dir: Benchmark directory containing instances.csv
    :param data_dir: Directory containing vnnlib input data (.npz files)
    :return: Nested dict structure or None if error
    """
    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.exists():
        return None

    onnx_rel_path = str(Path(onnx_path).relative_to(benchmark_dir)).replace("\\", "/")

    vnnlib_names = []
    try:
        with open(instances_csv) as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        model_path_norm = (
                            parts[0].strip().lstrip("./").replace("\\", "/")
                        )
                        onnx_rel_path_norm = onnx_rel_path.lstrip("./")
                        if model_path_norm == onnx_rel_path_norm:
                            vnnlib_name = Path(parts[1].strip()).stem
                            if vnnlib_name not in vnnlib_names:
                                vnnlib_names.append(vnnlib_name)
    except (IOError, OSError) as error:
        print(f"Error reading instances.csv: {error}")
        return None

    if not vnnlib_names:
        return None

    try:
        model = onnx.load(onnx_path)
        input_names = [
            inp.name
            for inp in model.graph.input
            if not any(init.name == inp.name for init in model.graph.initializer)
        ]
        if not input_names:
            return None

        input_name = input_names[0]
        input_tensor = next(
            (inp for inp in model.graph.input if inp.name == input_name), None
        )

        if input_tensor is None:
            return None

        expected_shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in input_tensor.type.tensor_type.shape.dim
        ]
    except (AttributeError, ValueError) as error:
        print(f"Error getting input shape: {error}")
        return None

    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except (RuntimeError, ValueError) as error:
        print(f"Error creating ONNX Runtime session: {error}")
        return None

    results = {}

    for vnnlib_name in vnnlib_names:
        vnnlib_subdir = data_dir / vnnlib_name / "or_group_0"

        if not vnnlib_subdir.exists():
            continue

        npz_files = sorted(vnnlib_subdir.glob("sub_prop_*.npz"))

        # Limit number of npz files to prevent long runtime
        MAX_NPZ_FILES_PER_BENCHMARK = 2
        npz_files = npz_files[:MAX_NPZ_FILES_PER_BENCHMARK]

        if not npz_files:
            continue

        lower_inputs = []
        lower_outputs = []
        upper_inputs = []
        upper_outputs = []

        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                input_bounds = data["input"]
                if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
                    continue

                try:
                    input_array = (
                        input_bounds[:, 0].astype(np.float32).reshape(expected_shape)
                    )
                    output = session.run(None, {input_name: input_array})
                    lower_inputs.append(input_array)
                    lower_outputs.append(output[0])
                except (RuntimeError, ValueError, TypeError, IndexError):
                    pass

                try:
                    input_array = (
                        input_bounds[:, 1].astype(np.float32).reshape(expected_shape)
                    )
                    output = session.run(None, {input_name: input_array})
                    upper_inputs.append(input_array)
                    upper_outputs.append(output[0])
                except (RuntimeError, ValueError, TypeError, IndexError):
                    pass

            except (IOError, OSError, ValueError, KeyError):
                continue

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


def build_all(
    source_dir: str = "../../../vnncomp2024_benchmarks/benchmarks",
    target_dir: str = "benchmarks",
    max_onnx_per_benchmark: int = 20,
    max_vnnlib_per_onnx: int = 2,
    max_calc_per_benchmark: int = 20,
) -> dict[str, tuple[int, int] | tuple[int, int, int]]:
    """Build complete benchmark dataset by running all steps.

    :param source_dir: Path to vnncomp2024_benchmarks/benchmarks directory
    :param target_dir: Target directory for copied files
    :param max_onnx_per_benchmark: Maximum ONNX files per benchmark
    :param max_vnnlib_per_onnx: Maximum VNNLib files per ONNX model
    :param max_calc_per_benchmark: Maximum models to calculate outputs for
    :return: Dictionary with results from each step
    """
    print("\n" + "=" * 70)
    print("STEP 1: Copy benchmarks from source")
    print("=" * 70)
    copy_result = copy_benchmarks(
        source_dir, target_dir, max_onnx_per_benchmark, max_vnnlib_per_onnx
    )

    print("\n" + "=" * 70)
    print("STEP 2: Extract VNNLib inputs to NumPy format")
    print("=" * 70)
    extract_result = extract_inputs(target_dir)

    print("\n" + "=" * 70)
    print("STEP 3: Calculate model input-output pairs")
    print("=" * 70)
    calc_result = calculate_outputs(target_dir, max_calc_per_benchmark)

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(
        f"Copied: {copy_result[0]} ONNX, {copy_result[1]} VNNLib from {copy_result[2]} benchmarks"
    )
    print(f"Extracted: {extract_result[0]} VNNLib files ({extract_result[1]} failed)")
    print(f"Calculated: {calc_result[0]} models ({calc_result[1]} failed)")

    return {
        "copy": copy_result,
        "extract": extract_result,
        "calculate": calc_result,
    }


def main() -> None:
    """Main entry point for script execution."""
    import sys

    if len(sys.argv) > 1:
        if "--copy-only" in sys.argv:
            copy_benchmarks()
        elif "--extract-only" in sys.argv:
            extract_inputs()
        elif "--calculate-only" in sys.argv:
            calculate_outputs()
        else:
            print(
                "Usage: python build_benchmarks.py [--copy-only|--extract-only|--calculate-only]"
            )
            print("No flags: Run all steps")
            sys.exit(1)
    else:
        build_all()


if __name__ == "__main__":
    main()
