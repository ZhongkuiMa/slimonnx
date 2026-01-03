"""Numerical comparison utilities for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = [
    "compare_model_outputs",
    "generate_inputs_from_bounds",
    "run_onnx_inference",
]

import numpy as np
import onnx
import onnxruntime as ort


def generate_inputs_from_bounds(
    input_bounds: tuple[list[float], list[float]],
    input_shape: tuple[int, ...],
    num_samples: int = 5,
) -> list[dict[str, np.ndarray]]:
    """Generate test inputs from lower/upper bounds.

    :param input_bounds: Tuple of (lower_bounds, upper_bounds) lists
    :param input_shape: Shape of input tensor
    :param num_samples: Number of input samples to generate
    :return: List of input dictionaries
    """
    lower_bounds, upper_bounds = input_bounds

    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("Lower and upper bounds must have same length")

    inputs = []
    rng = np.random.default_rng()
    for _ in range(num_samples):
        sample = rng.uniform(low=lower_bounds, high=upper_bounds, size=input_shape).astype(
            np.float32
        )

        inputs.append({"input": sample})

    return inputs


def run_onnx_inference(
    model_path: str,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run ONNX Runtime inference on model.

    :param model_path: Path to ONNX model
    :param inputs: Dictionary of input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    if len(input_names) == 1 and "input" in inputs:
        feed_dict = {input_names[0]: inputs["input"]}
    else:
        feed_dict = inputs

    outputs = session.run(output_names, feed_dict)

    return dict(zip(output_names, outputs, strict=False))


def _load_test_inputs_from_file(
    test_data_path: str, num_samples: int
) -> list[dict[str, np.ndarray]] | None:
    """Load test inputs from file (.npy or .npz).

    :param test_data_path: Path to test data file
    :param num_samples: Maximum number of samples to load
    :return: List of input dictionaries or None if loading fails
    """
    try:
        # Load test inputs from file (npy or npz)
        if test_data_path.endswith(".npy"):
            data = np.load(test_data_path)
            if len(data.shape) == 1:
                test_inputs = [{"input": data}]
            else:
                test_inputs = [{"input": data[i]} for i in range(min(num_samples, data.shape[0]))]
        elif test_data_path.endswith(".npz"):
            data = np.load(test_data_path)
            if "inputs" in data:
                inputs_data = data["inputs"]
                test_inputs = [
                    {"input": inputs_data[i]} for i in range(min(num_samples, inputs_data.shape[0]))
                ]
            elif "X" in data:
                x_data = data["X"]
                test_inputs = [
                    {"input": x_data[i]} for i in range(min(num_samples, x_data.shape[0]))
                ]
            else:
                return None
        else:
            raise ValueError(f"Unsupported test data format: {test_data_path}")

        if test_inputs is not None and len(test_inputs) > num_samples:
            test_inputs = test_inputs[:num_samples]

        return test_inputs

    except (OSError, ValueError, IndexError, KeyError) as error:
        print(f"Failed to load test data: {error}")
        return None


def _generate_test_inputs(
    model1_path: str,
    input_bounds: tuple[list[float], list[float]] | None,
    num_samples: int,
) -> list[dict[str, np.ndarray]]:
    """Generate test inputs from bounds or randomly.

    :param model1_path: Path to model (for shape inference)
    :param input_bounds: Optional bounds for input generation
    :param num_samples: Number of samples to generate
    :return: List of input dictionaries
    """
    model = onnx.load(model1_path)

    if input_bounds is not None:
        input_shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
        return generate_inputs_from_bounds(input_bounds, input_shape, num_samples)

    from slimonnx import utils

    return utils.generate_random_inputs(model, num_samples)


def _map_inputs_for_model2(
    inputs: dict[str, np.ndarray],
    input_names1: list[str],
    input_names2: list[str],
) -> dict[str, np.ndarray]:
    """Map input names from model1 to model2.

    :param inputs: Input dictionary with model1 names
    :param input_names1: Input names for model1
    :param input_names2: Input names for model2
    :return: Input dictionary with model2 names
    """
    if input_names1 == input_names2:
        return inputs

    inputs2 = {}
    for name1, name2 in zip(input_names1, input_names2, strict=False):
        if name1 in inputs:
            inputs2[name2] = inputs[name1]
        elif "input" in inputs and len(input_names2) == 1:
            inputs2[name2] = inputs["input"]

    return inputs2


def _compare_outputs(
    i: int,
    outputs1: dict[str, np.ndarray],
    outputs2: dict[str, np.ndarray],
    output_names1: list[str],
    output_names2: list[str],
    rtol: float,
    atol: float,
) -> tuple[bool, float, list[str]]:
    """Compare outputs from two models for a single sample.

    :param i: Sample index
    :param outputs1: Outputs from model1
    :param outputs2: Outputs from model2
    :param output_names1: Output names for model1
    :param output_names2: Output names for model2
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: Tuple of (match, max_diff, mismatches)
    """
    match = True
    max_diff = 0.0
    mismatches = []

    for idx, (out1_name, out2_name) in enumerate(zip(output_names1, output_names2, strict=False)):
        if out1_name not in outputs1 or out2_name not in outputs2:
            match = False
            mismatches.append(f"Test {i}: Output name mismatch ({out1_name} vs {out2_name})")
            continue

        out1 = outputs1[out1_name]
        out2 = outputs2[out2_name]

        if not np.allclose(out1, out2, rtol=rtol, atol=atol):
            match = False
            diff = np.max(np.abs(out1 - out2))
            max_diff = max(max_diff, diff)
            mismatches.append(f"Test {i}: Output {idx} differs (max_diff={diff:.2e})")

    return match, max_diff, mismatches


def compare_model_outputs(
    model1_path: str,
    model2_path: str,
    input_bounds: tuple[list[float], list[float]] | None = None,
    test_inputs: list[dict[str, np.ndarray]] | None = None,
    test_data_path: str | None = None,
    num_samples: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict:
    """Compare outputs of two ONNX models.

    :param model1_path: Path to first model (e.g., original)
    :param model2_path: Path to second model (e.g., optimized)
    :param input_bounds: Tuple of (lower_bounds, upper_bounds) for input generation
    :param test_inputs: Pre-generated inputs (if bounds not provided)
    :param test_data_path: Path to test data file (.npy, .npz)
    :param num_samples: Number of random samples if generating inputs
    :param rtol: Relative tolerance for comparison
    :param atol: Absolute tolerance for comparison
    :return: Comparison report dictionary
    """
    if test_inputs is None:
        if test_data_path is not None:
            test_inputs = _load_test_inputs_from_file(test_data_path, num_samples)

        if test_inputs is None:
            test_inputs = _generate_test_inputs(model1_path, input_bounds, num_samples)

    session1 = ort.InferenceSession(model1_path, providers=["CPUExecutionProvider"])
    session2 = ort.InferenceSession(model2_path, providers=["CPUExecutionProvider"])

    input_names1 = [inp.name for inp in session1.get_inputs()]
    output_names1 = [out.name for out in session1.get_outputs()]
    input_names2 = [inp.name for inp in session2.get_inputs()]
    output_names2 = [out.name for out in session2.get_outputs()]

    passed = 0
    failed = 0
    max_diff = 0.0
    mismatches = []

    for i, inputs in enumerate(test_inputs):
        outputs1 = run_onnx_inference(model1_path, inputs)
        inputs2 = _map_inputs_for_model2(inputs, input_names1, input_names2)
        outputs2 = run_onnx_inference(model2_path, inputs2)

        match, sample_max_diff, sample_mismatches = _compare_outputs(
            i, outputs1, outputs2, output_names1, output_names2, rtol, atol
        )

        max_diff = max(max_diff, sample_max_diff)
        mismatches.extend(sample_mismatches)

        if match:
            passed += 1
        else:
            failed += 1

    return {
        "all_match": failed == 0,
        "num_tests": len(test_inputs),
        "passed": passed,
        "failed": failed,
        "max_diff": max_diff,
        "mismatches": mismatches,
    }
