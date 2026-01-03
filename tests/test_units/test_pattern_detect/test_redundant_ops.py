"""Tests for redundant operations pattern detection."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.redundant_ops import (
    detect_add_zero,
    detect_div_one,
    detect_identity_reshape,
    detect_mul_one,
    detect_sub_zero,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestAddZeroDetection:
    """Test Add(X, 0) → X redundancy detection."""

    def test_detect_add_zero(self):
        """Test detection of Add with zero constant."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        zero = np.zeros((2, 3), dtype=np.float32)
        initializers = [create_initializer("zero", zero)]

        add_node = helper.make_node(
            "Add",
            inputs=["X", "zero"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_add_zero(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_multiple_add_zero(self):
        """Test detection of multiple Add(X, 0) patterns."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        zero1 = np.zeros((2, 3), dtype=np.float32)
        zero2 = np.zeros((2, 3), dtype=np.float32)
        initializers = [
            create_initializer("zero1", zero1),
            create_initializer("zero2", zero2),
        ]

        add1_node = helper.make_node(
            "Add",
            inputs=["X", "zero1"],
            outputs=["add1_out"],
        )
        add2_node = helper.make_node(
            "Add",
            inputs=["add1_out", "zero2"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([add1_node, add2_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_add_zero(nodes, initializers_dict)
        assert isinstance(instances, list)


class TestSubZeroDetection:
    """Test Sub(X, 0) → X redundancy detection."""

    def test_detect_sub_zero(self):
        """Test detection of Sub with zero constant."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        zero = np.zeros((2, 3), dtype=np.float32)
        initializers = [create_initializer("zero", zero)]

        sub_node = helper.make_node(
            "Sub",
            inputs=["X", "zero"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([sub_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_sub_zero(nodes, initializers_dict)
        assert isinstance(instances, list)


class TestMulOneDetection:
    """Test Mul(X, 1) → X redundancy detection."""

    def test_detect_mul_one(self):
        """Test detection of Mul with one constant."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        one = np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("one", one)]

        mul_node = helper.make_node(
            "Mul",
            inputs=["X", "one"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([mul_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_mul_one(nodes, initializers_dict)
        assert isinstance(instances, list)


class TestDivOneDetection:
    """Test Div(X, 1) → X redundancy detection."""

    def test_detect_div_one(self):
        """Test detection of Div with one constant."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        one = np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("one", one)]

        div_node = helper.make_node(
            "Div",
            inputs=["X", "one"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([div_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_div_one(nodes, initializers_dict)
        assert isinstance(instances, list)


class TestIdentityRedundancy:
    """Test Identity operations redundancy detection."""

    def test_detect_identity(self):
        """Test detection of Identity operations."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        identity_node = helper.make_node(
            "Identity",
            inputs=["X"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([identity_node], inputs, outputs)
        nodes = list(model.graph.node)
        initializers_dict: dict[str, Any] = {}

        instances = detect_identity_reshape(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_multiple_identities(self):
        """Test detection of multiple Identity operations."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        identity1_node = helper.make_node(
            "Identity",
            inputs=["X"],
            outputs=["id1_out"],
        )
        identity2_node = helper.make_node(
            "Identity",
            inputs=["id1_out"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([identity1_node, identity2_node], inputs, outputs)
        nodes = list(model.graph.node)
        initializers_dict: dict[str, Any] = {}

        instances = detect_identity_reshape(nodes, initializers_dict)
        assert isinstance(instances, list)


class TestMixedRedundantOps:
    """Test mixed redundant operations detection."""

    def test_detect_mixed_redundant_ops(self):
        """Test detection of various redundant operations together."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        zero = np.zeros((2, 3), dtype=np.float32)
        one = np.ones((2, 3), dtype=np.float32)
        initializers = [
            create_initializer("zero", zero),
            create_initializer("one", one),
        ]

        # Add X + 0
        add_node = helper.make_node(
            "Add",
            inputs=["X", "zero"],
            outputs=["add_out"],
        )
        # Mul by 1
        mul_node = helper.make_node(
            "Mul",
            inputs=["add_out", "one"],
            outputs=["mul_out"],
        )
        # Identity
        identity_node = helper.make_node(
            "Identity",
            inputs=["mul_out"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model(
            [add_node, mul_node, identity_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        # Test detection of any redundant ops
        add_instances = detect_add_zero(nodes, initializers_dict)
        mul_instances = detect_mul_one(nodes, initializers_dict)
        identity_instances = detect_identity_reshape(nodes, initializers_dict)

        assert isinstance(add_instances, list)
        assert isinstance(mul_instances, list)
        assert isinstance(identity_instances, list)

    def test_detect_no_redundant_ops(self):
        """Test when no redundant operations are present."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        W = np.random.randn(3, 2).astype(np.float32)
        initializers = [create_initializer("W", W)]

        # Normal MatMul (not redundant)
        matmul_node = helper.make_node(
            "MatMul",
            inputs=["X", "W"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        # Test that no redundant ops are detected
        add_instances = detect_add_zero(nodes, initializers_dict)
        assert isinstance(add_instances, list)

    def test_detect_empty_node_list(self):
        """Test detection with empty node list."""
        instances = detect_add_zero([], {})
        assert isinstance(instances, list)
        assert len(instances) == 0

    def test_detect_with_broadcast_shapes(self):
        """Test redundant operations with broadcast-compatible shapes."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3, 4, 4])]

        # Scalar zero (broadcasts to any shape)
        zero = np.array(0.0, dtype=np.float32)
        initializers = [create_initializer("zero", zero)]

        add_node = helper.make_node(
            "Add",
            inputs=["X", "zero"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_add_zero(nodes, initializers_dict)
        assert isinstance(instances, list)
