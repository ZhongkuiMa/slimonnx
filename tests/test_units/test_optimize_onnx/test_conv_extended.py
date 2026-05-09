"""Extended Conv-simplification tests.

Covers scenarios not in ``test_conv_simplification.py``: multi-conv chains,
list/5D shapes, batch edge cases. Duplicates of tests in the canonical file
were removed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx._conv import (
    _can_simplify_conv,
    _simplify_conv_to_flatten_gemm,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import create_initializer


class TestCanSimplifyConvAdditional:
    """Edge cases for _can_simplify_conv not covered in test_conv_simplification."""

    def test_can_simplify_large_batch(self):
        """Conv with a large batch dimension is still simplifiable."""
        weight = np.ones((32, 16, 1, 1), dtype=np.float32)
        result = _can_simplify_conv(
            weight,
            pre_output_shape=[128, 16, 1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
            auto_pad="NOTSET",
        )
        assert result is True

    @pytest.mark.parametrize(("h", "w"), [(1, 1), (2, 2), (5, 10), (10, 5)])
    def test_can_simplify_various_input_shapes(self, h, w):
        """Simplifiable across various spatial shapes when kernel matches."""
        weight = np.ones((16, 3, h, w), dtype=np.float32)
        result = _can_simplify_conv(
            weight,
            pre_output_shape=[1, 3, h, w],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
            auto_pad="NOTSET",
        )
        assert result is True


class TestSimplifyConvAdditional:
    """Multi-node-graph scenarios for _simplify_conv_to_flatten_gemm."""

    def test_non_consecutive_convs_not_simplified(self):
        """Convs separated by a Relu are not paired for simplification."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_0")
        conv2 = helper.make_node("Conv", inputs=["Z", "W2", "B2"], outputs=["Out"], name="conv_1")

        nodes = [conv1, relu, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1), dtype=np.float32)),
            "B1": create_initializer("B1", np.ones(16, dtype=np.float32)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1), dtype=np.float32)),
            "B2": create_initializer("B2", np.ones(32, dtype=np.float32)),
        }
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Z": [1, 16, 1, 1],
            "Out": [1, 32, 1, 1],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) == 3

    def test_5d_weight_raises(self):
        """A 5D conv weight is unsupported and must raise."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Out"], name="conv_1")

        nodes = [conv1, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1), dtype=np.float32)),
            "B1": create_initializer("B1", np.ones(16, dtype=np.float32)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1, 1), dtype=np.float32)),
            "B2": create_initializer("B2", np.ones(32, dtype=np.float32)),
        }
        data_shapes = {"X": [1, 3, 1, 1], "Y": [1, 16, 1, 1], "Out": [1, 32, 1, 1]}

        with pytest.raises((ValueError, NotImplementedError)):
            _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)

    def test_list_shape_values(self):
        """All-list data_shapes are accepted and produce valid nodes."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Out"], name="conv_1")

        nodes = [conv1, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1), dtype=np.float32)),
            "B1": create_initializer("B1", np.ones(16, dtype=np.float32)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1), dtype=np.float32)),
            "B2": create_initializer("B2", np.ones(32, dtype=np.float32)),
        }
        data_shapes = {"X": [1, 3, 1, 1], "Y": [1, 16, 1, 1], "Out": [1, 32, 1, 1]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) >= 2
        assert all(hasattr(n, "op_type") for n in result)

    def test_simplify_skipped_when_kernel_not_1x1(self):
        """A 3x3 follower conv blocks the simplification of the pair."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node(
            "Conv",
            inputs=["Y", "W2", "B2"],
            outputs=["Out"],
            name="conv_1",
            kernel_shape=[3, 3],
        )

        nodes = [conv1, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1), dtype=np.float32)),
            "B1": create_initializer("B1", np.ones(16, dtype=np.float32)),
            "W2": create_initializer("W2", np.ones((32, 16, 3, 3), dtype=np.float32)),
            "B2": create_initializer("B2", np.ones(32, dtype=np.float32)),
        }
        data_shapes = {"X": [1, 3, 1, 1], "Y": [1, 16, 1, 1], "Out": [1, 32, 3, 3]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) == 2

    def test_surrounding_nodes_preserved(self):
        """Non-Conv neighbours (Relu, Sigmoid) are preserved across the pass."""
        relu = helper.make_node("Relu", inputs=["X"], outputs=["A"], name="relu_0")
        conv1 = helper.make_node("Conv", inputs=["A", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Z"], name="conv_1")
        sigmoid = helper.make_node("Sigmoid", inputs=["Z"], outputs=["Out"], name="sigmoid_0")

        nodes = [relu, conv1, conv2, sigmoid]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1), dtype=np.float32)),
            "B1": create_initializer("B1", np.ones(16, dtype=np.float32)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1), dtype=np.float32)),
            "B2": create_initializer("B2", np.ones(32, dtype=np.float32)),
        }
        data_shapes = {
            "X": [1, 3, 1],
            "A": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Z": [1, 32, 1, 1],
            "Out": [1, 32, 1, 1],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert result[0].op_type == "Relu"
        assert result[-1].op_type == "Sigmoid"

    def test_multiple_consecutive_convs(self):
        """A chain of three 1x1 convs produces a valid (possibly fused) graph."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Z"], name="conv_1")
        conv3 = helper.make_node("Conv", inputs=["Z", "W3", "B3"], outputs=["Out"], name="conv_2")

        nodes = [conv1, conv2, conv3]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1), dtype=np.float32)),
            "B1": create_initializer("B1", np.ones(16, dtype=np.float32)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1), dtype=np.float32)),
            "B2": create_initializer("B2", np.ones(32, dtype=np.float32)),
            "W3": create_initializer("W3", np.ones((64, 32, 1, 1), dtype=np.float32)),
            "B3": create_initializer("B3", np.ones(64, dtype=np.float32)),
        }
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Z": [1, 32, 1, 1],
            "Out": [1, 64, 1, 1],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) >= 3
        assert all(hasattr(n, "op_type") for n in result)
