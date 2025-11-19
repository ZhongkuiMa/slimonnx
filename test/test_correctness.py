"""Correctness validation for SlimONNX optimizations.

Verifies that optimized models produce the same outputs as original models.
Tests both structural equivalence and numerical correctness.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_model_correctness", "test_all_correctness"]

import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

from slimonnx import SlimONNX, get_preset
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    load_onnx_model,
    load_vnnlib_inputs,
)


def generate_random_inputs(model: onnx.ModelProto, num_samples: int = 3) -> list[dict]:
    """Generate random test inputs for a model.

    :param model: ONNX ModelProto
    :param num_samples: Number of random samples to generate
    :return: List of input dictionaries
    """
    inputs_list = []

    for _ in range(num_samples):
        inputs = {}
        for inp in model.graph.input:
            # Skip initializers
            if any(init.name == inp.name for init in model.graph.initializer):
                continue

            # Get shape
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(1)
                else:
                    shape.append(1)

            # Get dtype
            dtype_map = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.INT64: np.int64,
            }
            dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)

            # Generate random input
            if dtype in [np.float32, np.float64]:
                inputs[inp.name] = np.random.randn(*shape).astype(dtype)
            else:
                inputs[inp.name] = np.random.randint(0, 10, size=shape, dtype=dtype)

        inputs_list.append(inputs)

    return inputs_list


def run_model(model_path: str, inputs: dict) -> dict:
    """Run ONNX model and return outputs.

    :param model_path: Path to ONNX model file
    :param inputs: Dictionary of input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)
    output_names = [out.name for out in session.get_outputs()]
    return {name: output for name, output in zip(output_names, outputs)}


def compare_outputs(
    outputs1: dict, outputs2: dict, rtol: float = 1e-5, atol: float = 1e-6
) -> tuple[bool, list[str]]:
    """Compare outputs from two models.

    :param outputs1: First model outputs
    :param outputs2: Second model outputs
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: Tuple of (all_match, mismatch_messages)
    """
    if set(outputs1.keys()) != set(outputs2.keys()):
        return False, [
            f"Output keys mismatch: {set(outputs1.keys())} vs {set(outputs2.keys())}"
        ]

    mismatches = []
    for key in outputs1.keys():
        out1 = outputs1[key]
        out2 = outputs2[key]

        if out1.shape != out2.shape:
            mismatches.append(f"  {key}: shape {out1.shape} vs {out2.shape}")
            continue

        if not np.allclose(out1, out2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(out1 - out2))
            mean_diff = np.mean(np.abs(out1 - out2))
            mismatches.append(
                f"  {key}: max diff {max_diff:.2e}, mean diff {mean_diff:.2e}"
            )

    return len(mismatches) == 0, mismatches


def test_model_correctness(
    onnx_path: str,
    use_vnnlib_inputs: bool = True,
    num_random_samples: int = 3,
    verbose: bool = False,
) -> dict:
    """Test correctness of optimization for a single model.

    :param onnx_path: Path to ONNX model file
    :param use_vnnlib_inputs: Whether to use torchvnnlib-extracted inputs
    :param num_random_samples: Number of random samples if torchvnnlib inputs not found
    :param verbose: Whether to print verbose output
    :return: Dictionary with test results
    """
    benchmark_name = get_benchmark_name(onnx_path)
    config = get_preset(benchmark_name)

    # Load original model
    original_model = load_onnx_model(onnx_path)
    original_node_count = len(original_model.graph.node)

    # Create temp paths
    temp_v22_path = onnx_path.replace(".onnx", "_temp_v22.onnx")
    temp_optimized_path = onnx_path.replace(".onnx", "_temp_optimized.onnx")

    result = {
        "success": False,
        "benchmark": benchmark_name,
        "original_nodes": original_node_count,
        "optimized_nodes": 0,
        "input_source": "none",
        "samples_tested": 0,
        "samples_passed": 0,
        "structural_match": False,
        "numerical_match": False,
        "error": None,
        "mismatches": [],
    }

    try:
        # Save as v22
        onnx.save(original_model, temp_v22_path)

        # Run optimization
        slimonnx = SlimONNX(verbose=verbose)
        slimonnx.slim(temp_v22_path, temp_optimized_path, config=config)

        # Load optimized model
        optimized_model = onnx.load(temp_optimized_path)
        optimized_node_count = len(optimized_model.graph.node)
        result["optimized_nodes"] = optimized_node_count

        # Structural check: we expect node count to stay same or decrease
        result["structural_match"] = optimized_node_count <= original_node_count

        # Get test inputs
        test_inputs = None
        if use_vnnlib_inputs:
            test_inputs = load_vnnlib_inputs(onnx_path)
            if test_inputs:
                result["input_source"] = "torchvnnlib"

        if test_inputs is None:
            test_inputs = generate_random_inputs(original_model, num_random_samples)
            result["input_source"] = "random"

        result["samples_tested"] = len(test_inputs)

        # Test each input
        all_passed = True
        for i, inputs in enumerate(test_inputs):
            try:
                original_outputs = run_model(temp_v22_path, inputs)
                optimized_outputs = run_model(temp_optimized_path, inputs)

                match, mismatches = compare_outputs(original_outputs, optimized_outputs)

                if match:
                    result["samples_passed"] += 1
                else:
                    all_passed = False
                    result["mismatches"].extend([f"Sample {i+1}:"] + mismatches)

            except Exception as e:
                all_passed = False
                result["mismatches"].append(f"Sample {i+1}: Runtime error - {e}")

        result["numerical_match"] = all_passed
        result["success"] = result["structural_match"] and result["numerical_match"]

    except Exception as e:
        result["error"] = str(e)

    finally:
        # Clean up temp files
        if os.path.exists(temp_v22_path):
            os.remove(temp_v22_path)
        if os.path.exists(temp_optimized_path):
            os.remove(temp_optimized_path)

    return result


def test_all_correctness(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    use_vnnlib_inputs: bool = True,
    verbose: bool = False,
) -> bool:
    """Test correctness for all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark to test
    :param use_vnnlib_inputs: Whether to use torchvnnlib-extracted inputs
    :param verbose: Whether to print verbose output
    :return: True if all tests passed, False otherwise
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    if not onnx_files:
        print(f"No ONNX files found in {benchmark_dir}")
        return False

    print(f"Testing correctness for {len(onnx_files)} models")
    print("=" * 80)

    passed = 0
    failed = []
    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = test_model_correctness(onnx_path, use_vnnlib_inputs, verbose=verbose)

        if result["success"]:
            print(
                f"OK ({result['samples_passed']}/{result['samples_tested']} samples, "
                f"{result['original_nodes']}->{result['optimized_nodes']} nodes, "
                f"{result['input_source']} inputs)"
            )
            passed += 1
        else:
            print("FAILED")
            if result["error"]:
                print(f"  Error: {result['error']}")
            if not result["structural_match"]:
                print(
                    f"  Structural mismatch: {result['original_nodes']} -> {result['optimized_nodes']} nodes"
                )
            if not result["numerical_match"]:
                print(
                    f"  Numerical mismatch ({result['samples_passed']}/{result['samples_tested']} samples passed):"
                )
                for msg in result["mismatches"][:5]:  # Show first 5 mismatches
                    print(f"    {msg}")
                if len(result["mismatches"]) > 5:
                    print(f"    ... and {len(result['mismatches'])-5} more")

            failed.append(basename)

    total_time = time.perf_counter() - start_time

    # Summary
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST SUMMARY")
    print("=" * 80)
    print(f"Total models: {len(onnx_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {passed/len(onnx_files)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")

    if failed:
        print(f"\nFailed models:")
        for basename in failed:
            print(f"  {basename}")

    return len(failed) == 0


if __name__ == "__main__":
    success = test_all_correctness(
        use_vnnlib_inputs=True, max_per_benchmark=20, verbose=False
    )
    sys.exit(0 if success else 1)
