"""Shared fixtures for SlimONNX tests."""

__docformat__ = "restructuredtext"

import tempfile
from pathlib import Path

import pytest
from onnx import TensorProto, helper

from slimonnx.configs import OptimizationConfig
from slimonnx.slimonnx import SlimONNX


def create_simple_model():
    """Create a simple ONNX model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "test_model", [X], [Y])
    model = helper.make_model(graph)
    return model


@pytest.fixture
def simple_model():
    """Fixture providing a simple ONNX model."""
    return create_simple_model()


@pytest.fixture
def slim_onnx():
    """Fixture providing a SlimONNX instance."""
    return SlimONNX()


@pytest.fixture
def temp_onnx_file(simple_model):
    """Fixture providing a temporary ONNX model file."""
    import onnx

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name

    onnx.save(simple_model, temp_path)

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_pair(simple_model):
    """Fixture providing a pair of temporary ONNX model files."""
    import onnx

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path1 = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path2 = f.name

    onnx.save(simple_model, path1)
    onnx.save(simple_model, path2)

    yield path1, path2

    Path(path1).unlink(missing_ok=True)
    Path(path2).unlink(missing_ok=True)


@pytest.fixture
def temp_file_paths():
    """Fixture providing temporary file paths for model I/O."""
    input_path = None
    output_path = None

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        input_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        output_path = f.name

    yield input_path, output_path

    Path(input_path).unlink(missing_ok=True)
    Path(output_path).unlink(missing_ok=True)


@pytest.fixture
def default_opt_config():
    """Fixture providing default optimization configuration."""
    return OptimizationConfig(remove_dropout=False)
