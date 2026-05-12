"""Tests for pattern detection utilities."""

__docformat__ = "restructuredtext"

import numpy as np
import pytest
from onnx import helper, numpy_helper

from slimonnx.pattern_detect.utils import (
    has_constant_weight,
    is_consecutive_nodes,
    validate_bn_inputs,
)


class TestIsConsecutiveNodes:
    """Test is_consecutive_nodes function."""

    @pytest.mark.parametrize(
        ("output", "input2", "expected"),
        [
            ("conv_out", "conv_out", True),  # valid
            ("other_output", "other_input", False),  # non-matching
            ([], "conv_out", False),  # empty output
        ],
    )
    def test_connectivity_checks(self, output, input2, expected):
        """Test connectivity checks for consecutive nodes."""
        node1 = helper.make_node("Conv", ["X", "W"], [output] if output else [])
        node2 = helper.make_node("Relu", [input2], ["relu_out"])
        nodes = [node1, node2]

        assert is_consecutive_nodes(node1, node2, nodes) is expected

    def test_handles_empty_input(self):
        """Test defensive check for empty input."""
        node1 = helper.make_node("Conv", ["X", "W"], ["conv_out"])
        # Create node2 with no inputs manually (can't do with helper)
        node2 = helper.make_node("Constant", [], ["const_out"])
        nodes = [node1, node2]

        assert is_consecutive_nodes(node1, node2, nodes) is False

    def test_rejects_self_loop(self):
        """Test that node is not its own consumer."""
        node1 = helper.make_node("Conv", ["X", "W"], ["conv_out"])
        nodes = [node1]

        assert is_consecutive_nodes(node1, node1, nodes) is False

    def test_identifies_direct_vs_indirect_chains(self):
        """Test chain: node1 -> node2 -> node3."""
        node1 = helper.make_node("Conv", ["X", "W"], ["out1"])
        node2 = helper.make_node("Relu", ["out1"], ["out2"])
        node3 = helper.make_node("Sigmoid", ["out2"], ["out3"])
        nodes = [node1, node2, node3]

        # node1 -> node2 should be consecutive
        assert is_consecutive_nodes(node1, node2, nodes) is True
        # node2 -> node3 should be consecutive
        assert is_consecutive_nodes(node2, node3, nodes) is True
        # node1 -> node3 should not be consecutive (not direct)
        assert is_consecutive_nodes(node1, node3, nodes) is False


class TestValidateBnInputs:
    """Test validate_bn_inputs function."""

    def test_valid_bn_node(self):
        """Test valid BN node with all parameters."""
        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bn_bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        bn_mean = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "mean")
        bn_var = numpy_helper.from_array(np.ones(3, dtype=np.float32), "var")

        initializers = {
            "scale": bn_scale,
            "bias": bn_bias,
            "mean": bn_mean,
            "var": bn_var,
        }

        bn_node = helper.make_node(
            "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"]
        )

        assert validate_bn_inputs(bn_node, initializers) is True

    def test_rejects_insufficient_inputs(self):
        """Test BN node with insufficient inputs."""
        bn_node = helper.make_node(
            "BatchNormalization",
            ["X", "scale", "bias"],  # Only 3 inputs
            ["Y"],
        )

        assert validate_bn_inputs(bn_node, {}) is False

    def test_bn_missing_initializer(self):
        """Test BN node with parameter not in initializers."""
        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        # Missing other parameters

        initializers = {"scale": bn_scale}

        bn_node = helper.make_node(
            "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"]
        )

        assert validate_bn_inputs(bn_node, initializers) is False

    def test_accepts_exactly_5_inputs(self):
        """Test BN with exactly 5 inputs (minimum required)."""
        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bn_bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        bn_mean = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "mean")
        bn_var = numpy_helper.from_array(np.ones(3, dtype=np.float32), "var")

        initializers = {
            "scale": bn_scale,
            "bias": bn_bias,
            "mean": bn_mean,
            "var": bn_var,
        }

        bn_node = helper.make_node(
            "BatchNormalization",
            ["X", "scale", "bias", "mean", "var"],
            ["Y", "mean_out", "var_out"],
        )

        assert validate_bn_inputs(bn_node, initializers) is True

    def test_accepts_more_than_5_inputs(self):
        """Test BN with more than 5 inputs (only first 5 checked)."""
        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bn_bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        bn_mean = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "mean")
        bn_var = numpy_helper.from_array(np.ones(3, dtype=np.float32), "var")
        extra = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "extra")

        initializers = {
            "scale": bn_scale,
            "bias": bn_bias,
            "mean": bn_mean,
            "var": bn_var,
            "extra": extra,
        }

        bn_node = helper.make_node(
            "BatchNormalization", ["X", "scale", "bias", "mean", "var", "extra"], ["Y"]
        )

        # Only first 5 inputs are checked
        assert validate_bn_inputs(bn_node, initializers) is True

    def test_bn_partial_missing_parameters(self):
        """Test BN missing some (but not all) parameters."""
        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bn_bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        # Missing mean and var

        initializers = {
            "scale": bn_scale,
            "bias": bn_bias,
        }

        bn_node = helper.make_node(
            "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"]
        )

        assert validate_bn_inputs(bn_node, initializers) is False


class TestHasConstantWeight:
    """Test has_constant_weight function."""

    @pytest.mark.parametrize(
        ("inputs", "initializers", "weight_index", "expected"),
        [
            (
                ["X", "W"],
                {"W": numpy_helper.from_array(np.ones((3, 3), dtype=np.float32), "W")},
                1,
                True,
            ),
            (["X"], {}, 1, False),
            (["X", "W"], {}, 1, False),
            (
                ["X", "Y", "W"],
                {"W": numpy_helper.from_array(np.ones((3, 3), dtype=np.float32), "W")},
                2,
                True,
            ),
            (["X", "W"], {}, 5, False),
        ],
    )
    def test_weight_lookup(self, inputs, initializers, weight_index, expected):
        """Test weight presence at various indices."""
        op = "Conv" if len(inputs) >= 2 else "Relu"
        node = helper.make_node(op, inputs, ["Y"])
        assert has_constant_weight(node, initializers, weight_index=weight_index) is expected


class TestUtilitiesIntegration:
    """Integration tests combining multiple utilities."""

    def test_detect_conv_bn_pattern(self):
        """Test typical Conv -> BN pattern using utilities."""
        # Create Conv node with weight
        weight = numpy_helper.from_array(np.ones((3, 3, 3, 3), dtype=np.float32), "W")
        initializers = {"W": weight}

        node1 = helper.make_node("Conv", ["X", "W"], ["conv_out"], kernel_shape=[3, 3])

        # Create BN node with parameters
        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bn_bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        bn_mean = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "mean")
        bn_var = numpy_helper.from_array(np.ones(3, dtype=np.float32), "var")

        initializers.update(
            {
                "scale": bn_scale,
                "bias": bn_bias,
                "mean": bn_mean,
                "var": bn_var,
            }
        )

        node2 = helper.make_node(
            "BatchNormalization", ["conv_out", "scale", "bias", "mean", "var"], ["Y"]
        )

        nodes = [node1, node2]

        # Verify pattern detection conditions
        assert is_consecutive_nodes(node1, node2, nodes) is True
        assert has_constant_weight(node1, initializers) is True
        assert validate_bn_inputs(node2, initializers) is True

    def test_branching_prevents_fusion(self):
        """Test that branching prevents fusion detection."""
        weight = numpy_helper.from_array(np.ones((3, 3, 3, 3), dtype=np.float32), "W")
        initializers = {"W": weight}

        node1 = helper.make_node("Conv", ["X", "W"], ["conv_out"])
        node2 = helper.make_node("Relu", ["conv_out"], ["relu_out"])
        node3 = helper.make_node("Add", ["conv_out", "Y"], ["add_out"])  # Branch!

        bn_scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bn_bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        bn_mean = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "mean")
        bn_var = numpy_helper.from_array(np.ones(3, dtype=np.float32), "var")

        initializers.update(
            {
                "scale": bn_scale,
                "bias": bn_bias,
                "mean": bn_mean,
                "var": bn_var,
            }
        )

        node4 = helper.make_node(
            "BatchNormalization", ["relu_out", "scale", "bias", "mean", "var"], ["BN_out"]
        )

        nodes = [node1, node2, node3, node4]

        # Branching is detected
        assert is_consecutive_nodes(node1, node2, nodes) is False
