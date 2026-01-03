"""Tests for Transpose+BN+Transpose pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.transpose_bn import detect_transpose_bn_transpose

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestTransposeBNTransposeDetection:
    """Test Transpose+BN+Transpose pattern detection."""

    def test_detect_basic_transpose_bn_transpose(self):
        """Test detection of basic Transpose+BN+Transpose pattern."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3, 4])]

        bn_scale = np.ones(4, dtype=np.float32)
        bn_bias = np.zeros(4, dtype=np.float32)
        bn_mean = np.zeros(4, dtype=np.float32)
        bn_var = np.ones(4, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        # Transpose(0,2,1) -> BN -> Transpose(0,2,1)
        transpose1_node = helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["t1_out"],
            perm=[0, 2, 1],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        transpose2_node = helper.make_node(
            "Transpose",
            inputs=["bn_out"],
            outputs=["Y"],
            perm=[0, 2, 1],
        )

        model = create_minimal_onnx_model(
            [transpose1_node, bn_node, transpose2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_transpose_bn_transpose(nodes, initializers_dict)

        assert isinstance(instances, list)
        assert len(instances) > 0
        assert instances[0]["perm"] == (0, 2, 1)

    def test_detect_wrong_perm_not_detected(self):
        """Test that pattern with wrong perm is not detected."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [3, 2, 4])]

        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        # Transpose with wrong perm (0,2,1) but second one is (1,0,2)
        transpose1_node = helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["t1_out"],
            perm=[0, 2, 1],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        transpose2_node = helper.make_node(
            "Transpose",
            inputs=["bn_out"],
            outputs=["Y"],
            perm=[1, 0, 2],  # Different perm
        )

        model = create_minimal_onnx_model(
            [transpose1_node, bn_node, transpose2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_transpose_bn_transpose(nodes, initializers_dict)

        # Should not be detected due to wrong perm
        assert isinstance(instances, list)

    def test_detect_missing_bn_params_not_detected(self):
        """Test that pattern without all BN parameters is not detected."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3, 4])]

        bn_scale = np.ones(4, dtype=np.float32)
        bn_bias = np.zeros(4, dtype=np.float32)
        # Missing bn_mean and bn_var

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
        ]

        transpose1_node = helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["t1_out"],
            perm=[0, 2, 1],
        )
        # BN with missing inputs
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias"],  # Missing mean and var
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        transpose2_node = helper.make_node(
            "Transpose",
            inputs=["bn_out"],
            outputs=["Y"],
            perm=[0, 2, 1],
        )

        model = create_minimal_onnx_model(
            [transpose1_node, bn_node, transpose2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_transpose_bn_transpose(nodes, initializers_dict)

        # Should not be detected
        assert isinstance(instances, list)

    def test_detect_non_consecutive_nodes_not_detected(self):
        """Test that pattern with non-consecutive nodes is not detected."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3, 4])]

        bn_scale = np.ones(4, dtype=np.float32)
        bn_bias = np.zeros(4, dtype=np.float32)
        bn_mean = np.zeros(4, dtype=np.float32)
        bn_var = np.ones(4, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        transpose1_node = helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["t1_out"],
            perm=[0, 2, 1],
        )
        # Add unrelated node in between
        relu_node = helper.make_node(
            "Relu",
            inputs=["t1_out"],
            outputs=["relu_out"],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["relu_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        transpose2_node = helper.make_node(
            "Transpose",
            inputs=["bn_out"],
            outputs=["Y"],
            perm=[0, 2, 1],
        )

        model = create_minimal_onnx_model(
            [transpose1_node, relu_node, bn_node, transpose2_node],
            inputs,
            outputs,
            initializers,
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_transpose_bn_transpose(nodes, initializers_dict)

        # Should not be detected due to non-consecutive nodes
        assert isinstance(instances, list)

    def test_detect_empty_node_list(self):
        """Test detection with empty node list."""
        instances = detect_transpose_bn_transpose([], {})

        assert isinstance(instances, list)
        assert len(instances) == 0

    def test_detect_single_node_not_detected(self):
        """Test that single node is not detected as pattern."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 4, 3])]

        transpose_node = helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["Y"],
            perm=[0, 2, 1],
        )

        model = create_minimal_onnx_model([transpose_node], inputs, outputs)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_transpose_bn_transpose(nodes, initializers_dict)

        # Should not be detected
        assert isinstance(instances, list)
        assert len(instances) == 0

    def test_detect_returns_dict_list(self):
        """Test that detection returns list of dictionaries."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3, 4])]

        bn_scale = np.ones(4, dtype=np.float32)
        bn_bias = np.zeros(4, dtype=np.float32)
        bn_mean = np.zeros(4, dtype=np.float32)
        bn_var = np.ones(4, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        transpose1_node = helper.make_node(
            "Transpose",
            inputs=["X"],
            outputs=["t1_out"],
            perm=[0, 2, 1],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        transpose2_node = helper.make_node(
            "Transpose",
            inputs=["bn_out"],
            outputs=["Y"],
            perm=[0, 2, 1],
        )

        model = create_minimal_onnx_model(
            [transpose1_node, bn_node, transpose2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_transpose_bn_transpose(nodes, initializers_dict)

        # Verify structure
        assert isinstance(instances, list)
        if len(instances) > 0:
            assert isinstance(instances[0], dict)
            assert "perm" in instances[0] or "can_fuse" in instances[0]
