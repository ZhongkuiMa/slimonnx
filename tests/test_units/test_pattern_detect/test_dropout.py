"""Tests for Dropout pattern detection."""

import sys
from pathlib import Path
from typing import Any

from onnx import helper

from slimonnx.pattern_detect.dropout import detect_dropout

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestDropoutDetection:
    """Test Dropout pattern detection."""

    def test_detect_single_dropout(self, create_dropout_model):
        """Test detection of single Dropout node."""
        model = create_dropout_model(dropout_ratio=0.5)

        # Extract nodes and initializers
        nodes = list(model.graph.node)
        initializers: dict[str, Any] = {}

        # Detect dropout patterns
        instances = detect_dropout(nodes, initializers)

        # Verify detection
        assert len(instances) == 1
        assert instances[0]["input"] == "X"
        assert instances[0]["output"] == "Y"
        # ratio can be None if it's passed as an input instead of attribute
        assert instances[0]["ratio"] is None or instances[0]["ratio"] == 0.5
        assert instances[0]["should_remove"] is True

    def test_detect_multiple_dropout(self):
        """Test detection of multiple Dropout nodes."""
        # Create model with two Dropout nodes
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Two Dropout nodes in sequence
        dropout1 = helper.make_node(
            "Dropout",
            inputs=["X"],
            outputs=["dropout1_out"],
            name="dropout1",
            ratio=0.5,
        )
        dropout2 = helper.make_node(
            "Dropout",
            inputs=["dropout1_out"],
            outputs=["Y"],
            name="dropout2",
            ratio=0.3,
        )
        nodes = [dropout1, dropout2]

        model = create_minimal_onnx_model(nodes, inputs, outputs)

        # Detect dropout patterns
        instances = detect_dropout(list(model.graph.node), {})

        # Verify detection
        assert len(instances) == 2
        # ratio can be None if it's passed as an input instead of attribute
        assert instances[0]["ratio"] is None or abs(instances[0]["ratio"] - 0.5) < 1e-6
        assert instances[1]["ratio"] is None or abs(instances[1]["ratio"] - 0.3) < 1e-6

    def test_detect_dropout_no_ratio(self):
        """Test detection of Dropout without ratio attribute."""
        # Create model with Dropout node without ratio
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        dropout = helper.make_node(
            "Dropout",
            inputs=["X"],
            outputs=["Y"],
            name="dropout_no_ratio",
        )
        nodes = [dropout]

        model = create_minimal_onnx_model(nodes, inputs, outputs)

        # Detect dropout patterns
        instances = detect_dropout(list(model.graph.node), {})

        # Verify detection
        assert len(instances) == 1
        assert instances[0]["node"] == "dropout_no_ratio"
        assert instances[0]["ratio"] is None
        assert instances[0]["should_remove"] is True

    def test_detect_no_dropout(self):
        """Test detection with no Dropout nodes."""
        # Create model with Identity node (no dropout)
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        nodes = [identity]

        model = create_minimal_onnx_model(nodes, inputs, outputs)

        # Detect dropout patterns
        instances = detect_dropout(list(model.graph.node), {})

        # Verify no detection
        assert len(instances) == 0

    def test_detect_dropout_input_output_names(self, create_dropout_model):
        """Test that detected Dropout has correct input/output names."""
        model = create_dropout_model(dropout_ratio=0.7)

        nodes = list(model.graph.node)
        instances = detect_dropout(nodes, {})

        assert len(instances) == 1
        instance = instances[0]

        # Verify the instance is correctly extracted
        assert instance["input"] is not None
        assert instance["output"] is not None
        assert instance["input"] == "X"
        assert instance["output"] == "Y"

    def test_detect_empty_node_list(self):
        """Test detection with empty node list."""
        instances = detect_dropout([], {})
        assert len(instances) == 0

    def test_dropout_always_removable(self, create_dropout_model):
        """Test that detected Dropout is always marked as removable."""
        model = create_dropout_model(dropout_ratio=0.1)

        nodes = list(model.graph.node)
        instances = detect_dropout(nodes, {})

        assert len(instances) == 1
        assert instances[0]["should_remove"] is True
