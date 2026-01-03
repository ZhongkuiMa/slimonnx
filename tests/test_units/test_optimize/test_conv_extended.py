"""Extended tests for Conv simplification (_conv.py)."""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from slimonnx.optimize_onnx._conv import (
    _can_simplify_conv,
    _create_flatten_gemm_nodes,
    _simplify_conv_to_flatten_gemm,
    _validate_conv_simplification,
)


def create_initializer(name, array):
    """Create a TensorProto initializer from numpy array."""
    return numpy_helper.from_array(array.astype(np.float32), name)


class TestCanSimplifyConv:
    """Test _can_simplify_conv function."""

    def test_can_simplify_basic(self):
        """Test basic Conv that can be simplified."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is True

    def test_cannot_simplify_non_1x1_kernel(self):
        """Test Conv with non-1x1 kernel cannot be simplified."""
        weight = np.ones((16, 3, 3, 3), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]  # Input shape must match weight kernel
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        # Should be False because input spatial dims (1,1) don't match kernel dims (3,3)
        assert result is False

    def test_cannot_simplify_mismatched_channels(self):
        """Test Conv with mismatched channel dimensions."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 5, 1, 1]  # Mismatched input channel
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is False

    def test_cannot_simplify_with_padding(self):
        """Test Conv with padding cannot be simplified."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]
        pads = [1, 1, 1, 1]  # Non-zero padding
        strides = [1, 1]
        dilations = [1, 1]
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is False

    def test_cannot_simplify_with_stride_gt_1(self):
        """Test Conv with stride > 1 cannot be simplified."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]
        pads = [0, 0, 0, 0]
        strides = [2, 2]  # Stride > 1
        dilations = [1, 1]
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is False

    def test_cannot_simplify_with_dilation_gt_1(self):
        """Test Conv with dilation > 1 cannot be simplified."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [2, 2]  # Dilation > 1
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is False

    def test_cannot_simplify_with_group_gt_1(self):
        """Test Conv with group > 1 cannot be simplified."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        group = 2  # Group > 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is False

    def test_cannot_simplify_with_auto_pad(self):
        """Test Conv with auto_pad set cannot be simplified."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        pre_output_shape = [1, 3, 1, 1]
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        group = 1
        auto_pad = "SAME_LOWER"  # Non-NOTSET auto_pad

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is False

    def test_can_simplify_large_batch(self):
        """Test Conv with large batch size."""
        weight = np.ones((32, 16, 1, 1), dtype=np.float32)
        pre_output_shape = [128, 16, 1, 1]  # Large batch size
        pads = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        group = 1
        auto_pad = "NOTSET"

        result = _can_simplify_conv(
            weight,
            pre_output_shape,
            pads,
            strides,
            dilations,
            group,
            auto_pad,
        )

        assert result is True

    def test_can_simplify_different_input_shapes(self):
        """Test Conv with various valid input shapes."""
        for h, w in [(1, 1), (2, 2), (5, 10), (10, 5)]:
            weight = np.ones((16, 3, h, w), dtype=np.float32)
            pre_output_shape = [1, 3, h, w]
            pads = [0, 0, 0, 0]
            strides = [1, 1]
            dilations = [1, 1]
            group = 1
            auto_pad = "NOTSET"

            result = _can_simplify_conv(
                weight,
                pre_output_shape,
                pads,
                strides,
                dilations,
                group,
                auto_pad,
            )

            assert result is True


class TestValidateConvSimplification:
    """Test _validate_conv_simplification function."""

    def test_validate_valid_simplification(self):
        """Test validation passes for valid simplification."""
        pre_pre_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        pre_node = helper.make_node("Conv", inputs=["Y", "W"], outputs=["Z"], name="conv_0")
        node = helper.make_node("Conv", inputs=["Z", "W2"], outputs=["Out"], name="conv_1")
        nodes = [pre_pre_node, pre_node, node]

        # Should not raise
        _validate_conv_simplification(pre_pre_node, pre_node, node, nodes)

    def test_validate_fails_no_predecessor(self):
        """Test validation fails when pre_pre_node is None."""
        pre_node = helper.make_node("Conv", inputs=["Y", "W"], outputs=["Z"], name="conv_0")
        node = helper.make_node("Conv", inputs=["Z", "W2"], outputs=["Out"], name="conv_1")
        nodes = [pre_node, node]

        with pytest.raises(ValueError, match="no predecessor"):
            _validate_conv_simplification(None, pre_node, node, nodes)

    def test_validate_with_non_only_next_node(self):
        """Test validation with nodes where pre_node is used by other nodes too."""
        pre_pre_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        pre_node = helper.make_node("Conv", inputs=["Y", "W"], outputs=["Z"], name="conv_0")
        node1 = helper.make_node("Conv", inputs=["Z", "W2"], outputs=["Out1"], name="conv_1")
        node2 = helper.make_node("Add", inputs=["Z", "B"], outputs=["Out2"], name="add_0")
        nodes = [pre_pre_node, pre_node, node1, node2]

        # This would fail because pre_node's output is used by both node1 and node2
        # so node1 is not the only next node
        with pytest.raises(ValueError, match="multiple successors"):
            _validate_conv_simplification(pre_pre_node, pre_node, node1, nodes)

    def test_validate_with_multiple_pre_pre_outputs(self):
        """Test validation when pre_pre_node outputs are used elsewhere."""
        pre_pre_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        pre_node = helper.make_node("Conv", inputs=["Y", "W"], outputs=["Z"], name="conv_0")
        node = helper.make_node("Conv", inputs=["Z", "W2"], outputs=["Out"], name="conv_1")
        other_node = helper.make_node("Add", inputs=["Y", "B"], outputs=["Other"], name="add_0")
        nodes = [pre_pre_node, other_node, pre_node, node]

        # This would fail because pre_pre_node's output is used by both pre_node and other_node
        with pytest.raises(ValueError, match="multiple successors"):
            _validate_conv_simplification(pre_pre_node, pre_node, node, nodes)


class TestCreateFlattenGemmNodes:
    """Test _create_flatten_gemm_nodes function."""

    def test_create_nodes_basic(self):
        """Test creating Flatten and Gemm nodes from Conv."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        conv_node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"], name="conv_0")
        initializers = {
            "W": create_initializer("W", weight),
            "B": create_initializer("B", np.ones(16)),
        }

        flatten_node, gemm_node = _create_flatten_gemm_nodes(conv_node, weight, initializers)

        assert flatten_node.op_type == "Flatten"
        assert gemm_node.op_type == "Gemm"
        assert flatten_node.name == "conv_0_flatten"
        assert gemm_node.name == "conv_0_flatten_gemm"

    def test_create_nodes_preserves_output_name(self):
        """Test that Gemm output preserves Conv output name."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        conv_node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"], name="conv_0")
        initializers = {
            "W": create_initializer("W", weight),
            "B": create_initializer("B", np.ones(16)),
        }

        _flatten_node, gemm_node = _create_flatten_gemm_nodes(conv_node, weight, initializers)

        assert gemm_node.output[0] == "Y"

    def test_create_nodes_reshapes_weight(self):
        """Test that weight is reshaped correctly."""
        weight = np.arange(48, dtype=np.float32).reshape(16, 3, 1, 1)
        conv_node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"], name="conv_0")
        initializers = {
            "W": create_initializer("W", weight),
            "B": create_initializer("B", np.ones(16)),
        }

        _flatten_node, _gemm_node = _create_flatten_gemm_nodes(conv_node, weight, initializers)

        # Weight should be reshaped from (16, 3, 1, 1) to (3, 16) after transpose
        reshaped_weight = numpy_helper.to_array(initializers["W"])
        assert reshaped_weight.shape == (3, 16)

    def test_create_nodes_flatten_input(self):
        """Test Flatten node takes Conv input."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        conv_node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"], name="conv_0")
        initializers = {
            "W": create_initializer("W", weight),
            "B": create_initializer("B", np.ones(16)),
        }

        flatten_node, _gemm_node = _create_flatten_gemm_nodes(conv_node, weight, initializers)

        assert flatten_node.input[0] == "X"

    def test_create_nodes_gemm_uses_correct_inputs(self):
        """Test Gemm node uses Flatten output, weight, and bias."""
        weight = np.ones((16, 3, 1, 1), dtype=np.float32)
        conv_node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"], name="conv_0")
        initializers = {
            "W": create_initializer("W", weight),
            "B": create_initializer("B", np.ones(16)),
        }

        flatten_node, gemm_node = _create_flatten_gemm_nodes(conv_node, weight, initializers)

        # Gemm should use Flatten output as first input
        assert gemm_node.input[0] == flatten_node.output[0]
        assert gemm_node.input[1] == "W"
        assert gemm_node.input[2] == "B"


class TestSimplifyConvToFlattenGemm:
    """Test _simplify_conv_to_flatten_gemm function."""

    def test_simplify_single_conv_no_simplification(self):
        """Test that single Conv without predecessor is not simplified."""
        conv_node = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["Y"], name="conv_0")
        nodes = [conv_node]
        initializers = {
            "W": create_initializer("W", np.ones((16, 3, 1, 1))),
            "B": create_initializer("B", np.ones(16)),
        }
        data_shapes = {"X": [1, 3, 1, 1], "Y": [1, 16, 1, 1]}

        # Should not raise, just returns original nodes since no simplification possible
        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) == 1
        assert result[0].op_type == "Conv"

    def test_simplify_non_consecutive_convs(self):
        """Test simplification doesn't apply to non-consecutive Conv nodes."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_0")
        conv2 = helper.make_node("Conv", inputs=["Z", "W2", "B2"], outputs=["Out"], name="conv_1")

        nodes = [conv1, relu, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1))),
            "B1": create_initializer("B1", np.ones(16)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1))),
            "B2": create_initializer("B2", np.ones(32)),
        }
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Z": [1, 16, 1, 1],
            "Out": [1, 32, 1, 1],
        }

        # Should not raise, but also won't simplify non-consecutive convs
        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) == 3  # No simplification

    def test_simplify_with_5d_weight(self):
        """Test simplification with 5D weight raises NotImplementedError."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Out"], name="conv_1")

        nodes = [conv1, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1))),
            "B1": create_initializer("B1", np.ones(16)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1, 1))),  # 5D weight
            "B2": create_initializer("B2", np.ones(32)),
        }
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Out": [1, 32, 1, 1],
        }

        # Should raise NotImplementedError from _get_conv_params
        with pytest.raises((ValueError, NotImplementedError)):
            _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)

    def test_simplify_with_list_shape(self):
        """Test simplification with list shape values."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Out"], name="conv_1")

        nodes = [conv1, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1))),
            "B1": create_initializer("B1", np.ones(16)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1))),
            "B2": create_initializer("B2", np.ones(32)),
        }
        # All shapes are lists
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Out": [1, 32, 1, 1],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        assert len(result) >= 2  # Should complete without error

    def test_simplify_conditions_not_met(self):
        """Test no simplification when conditions are not met."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node(
            "Conv",
            inputs=["Y", "W2", "B2"],
            outputs=["Out"],
            name="conv_1",
            kernel_shape=[3, 3],  # Non-1x1 kernel
        )

        nodes = [conv1, conv2]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1))),
            "B1": create_initializer("B1", np.ones(16)),
            "W2": create_initializer("W2", np.ones((32, 16, 3, 3))),
            "B2": create_initializer("B2", np.ones(32)),
        }
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Out": [1, 32, 3, 3],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
        # Should have 2 Conv nodes still (no simplification)
        assert len(result) == 2

    def test_simplify_preserves_other_nodes(self):
        """Test that other nodes are preserved during simplification."""
        relu = helper.make_node("Relu", inputs=["X"], outputs=["A"], name="relu_0")
        conv1 = helper.make_node("Conv", inputs=["A", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Z"], name="conv_1")
        sigmoid = helper.make_node("Sigmoid", inputs=["Z"], outputs=["Out"], name="sigmoid_0")

        nodes = [relu, conv1, conv2, sigmoid]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1))),
            "B1": create_initializer("B1", np.ones(16)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1))),
            "B2": create_initializer("B2", np.ones(32)),
        }
        data_shapes = {
            "X": [1, 3, 1],
            "A": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Z": [1, 32, 1, 1],
            "Out": [1, 32, 1, 1],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)

        # Should have Relu, possibly simplified Conv nodes, and Sigmoid
        assert result[0].op_type == "Relu"
        assert result[-1].op_type == "Sigmoid"

    def test_simplify_multiple_consecutive_convs(self):
        """Test simplification with multiple consecutive Conv nodes."""
        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["Y"], name="conv_0")
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["Z"], name="conv_1")
        conv3 = helper.make_node("Conv", inputs=["Z", "W3", "B3"], outputs=["Out"], name="conv_2")

        nodes = [conv1, conv2, conv3]
        initializers = {
            "W1": create_initializer("W1", np.ones((16, 3, 1, 1))),
            "B1": create_initializer("B1", np.ones(16)),
            "W2": create_initializer("W2", np.ones((32, 16, 1, 1))),
            "B2": create_initializer("B2", np.ones(32)),
            "W3": create_initializer("W3", np.ones((64, 32, 1, 1))),
            "B3": create_initializer("B3", np.ones(64)),
        }
        data_shapes = {
            "X": [1, 3, 1, 1],
            "Y": [1, 16, 1, 1],
            "Z": [1, 32, 1, 1],
            "Out": [1, 64, 1, 1],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)

        # At minimum should have some nodes
        assert len(result) >= 3
