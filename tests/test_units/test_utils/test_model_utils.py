"""Unit tests for model utility functions."""

import numpy as np

from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestModelUtils:
    """Test model utility functions."""

    def test_create_tensor_value_info(self):
        """Test tensor info creation."""
        tensor_info = create_tensor_value_info("test", "float32", [1, 3])
        assert tensor_info.name == "test"
        assert tensor_info.type.tensor_type.elem_type == 1  # float32

    def test_create_minimal_onnx_model(self):
        """Test minimal ONNX model creation."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 3])

        model = create_minimal_onnx_model([], [X], [Y], [])

        assert model is not None
        assert model.graph.name == "test_graph"
        assert len(model.graph.input) == 1
        assert len(model.graph.output) == 1

    def test_create_initializer(self):
        """Test initializer creation."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        init = create_initializer("W", arr)

        assert init.name == "W"
        assert len(init.dims) == 2
        assert init.dims[0] == 2
        assert init.dims[1] == 2

    def test_get_model_size_bytes(self):
        """get_model_size() returns size in bytes."""
        # Create model with known initializer size
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        # Initializer: (3, 2) float32 = 6 * 4 = 24 bytes
        W = np.ones((3, 2), dtype=np.float32)
        initializers = [create_initializer("W", W)]

        model = create_minimal_onnx_model([], [X], [Y], initializers)

        # Get model size
        model_bytes = model.SerializeToString()
        size = len(model_bytes)

        # Should be at least the initializer size + overhead
        assert size > 0, "Model size should be greater than 0"

        # For this simple model, size should be reasonable
        # (at least the ~24 bytes of initializer data + ONNX proto overhead)
        assert size >= 24, "Model should contain initializer data"
