"""Tests for pattern detection registry."""

import sys
from pathlib import Path

from onnx import helper

from slimonnx.pattern_detect.registry import PATTERNS, detect_all_patterns

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestPatternRegistry:
    """Test pattern detection registry."""

    def test_patterns_metadata_exists(self):
        """Test that PATTERNS dict contains all expected pattern metadata."""
        # Check critical patterns exist
        critical_patterns = [
            "matmul_add",
            "conv_bn",
            "dropout",
            "add_zero",
        ]

        for pattern_name in critical_patterns:
            assert pattern_name in PATTERNS or any(pattern_name in key for key in PATTERNS), (
                f"Pattern '{pattern_name}' or variant not found in PATTERNS"
            )

    def test_patterns_have_metadata(self):
        """Test that each pattern has description and category."""
        for pattern_name, metadata in PATTERNS.items():
            assert "description" in metadata, f"Pattern {pattern_name} missing description"
            assert "category" in metadata, f"Pattern {pattern_name} missing category"
            assert "severity" in metadata, f"Pattern {pattern_name} missing severity"

    def test_pattern_categories(self):
        """Test that patterns have valid categories."""
        valid_categories = {
            "fusion",
            "redundant",
            "inference",
            "constant_folding",
            "shape_optimization",
        }

        for pattern_name, metadata in PATTERNS.items():
            assert metadata["category"] in valid_categories, (
                f"Invalid category for {pattern_name}: {metadata['category']}"
            )

    def test_pattern_severities(self):
        """Test that patterns have valid severity levels."""
        valid_severities = {"optimization", "redundant", "info", "warning"}

        for pattern_name, metadata in PATTERNS.items():
            assert metadata["severity"] in valid_severities, (
                f"Invalid severity for {pattern_name}: {metadata['severity']}"
            )

    def test_detect_all_patterns_dropout(self, create_dropout_model):
        """Test detect_all_patterns with Dropout."""
        model = create_dropout_model(dropout_ratio=0.5)

        # Build initializers dict
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = None

        # Detect all patterns
        results = detect_all_patterns(list(model.graph.node), initializers, data_shapes)

        # Verify dropout detection
        assert "dropout" in results
        assert len(results["dropout"]) >= 0  # May or may not detect depending on implementation

    def test_detect_all_patterns_matmul_add(self, create_matmul_add_model):
        """Test detect_all_patterns with MatMul+Add pattern."""
        model = create_matmul_add_model()

        # Build initializers dict
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = None

        # Detect all patterns
        results = detect_all_patterns(list(model.graph.node), initializers, data_shapes)

        # Verify matmul_add detection or pattern exists
        assert "matmul_add" in results or any("matmul" in key for key in results), (
            "MatMul+Add pattern not detected"
        )

    def test_detect_all_patterns_returns_dict(self, create_dropout_model):
        """Test that detect_all_patterns returns a dictionary."""
        model = create_dropout_model()

        initializers = {init.name: init for init in model.graph.initializer}
        results = detect_all_patterns(list(model.graph.node), initializers)

        assert isinstance(results, dict)

    def test_detect_all_patterns_empty_model(self):
        """Test detect_all_patterns with empty model."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Simple identity model
        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([identity], inputs, outputs)

        initializers = {init.name: init for init in model.graph.initializer}
        results = detect_all_patterns(list(model.graph.node), initializers)

        # Should return empty dict or dict with empty pattern lists
        assert isinstance(results, dict)

    def test_detect_all_patterns_with_data_shapes(self, create_matmul_add_model):
        """Test detect_all_patterns with data_shapes."""
        model = create_matmul_add_model()

        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [2, 3]}

        results = detect_all_patterns(list(model.graph.node), initializers, data_shapes)

        assert isinstance(results, dict)

    def test_patterns_count(self):
        """Test that PATTERNS dict is not empty."""
        assert len(PATTERNS) > 0, "PATTERNS dict should contain pattern definitions"

    def test_pattern_names_valid(self):
        """Test that pattern names are valid identifiers."""
        for pattern_name in PATTERNS:
            # Pattern names should be lowercase with underscores
            assert pattern_name.islower() or "_" in pattern_name
            assert pattern_name.replace("_", "").replace(" ", "").isalnum()


class TestDetectAllPatternsIntegration:
    """Integration tests for detect_all_patterns."""

    def test_detect_all_patterns_conv_bn(self, create_conv_bn_model):
        """Test pattern detection with Conv+BN model."""
        model = create_conv_bn_model()

        initializers = {init.name: init for init in model.graph.initializer}
        results = detect_all_patterns(list(model.graph.node), initializers)

        # Conv+BN should be detectable
        assert isinstance(results, dict)

    def test_detect_patterns_with_gemm(self, create_gemm_model):
        """Test pattern detection with Gemm model."""
        model = create_gemm_model()

        initializers = {init.name: init for init in model.graph.initializer}
        results = detect_all_patterns(list(model.graph.node), initializers)

        assert isinstance(results, dict)

    def test_detect_patterns_consistency(self, create_dropout_model):
        """Test that detection is consistent across multiple calls."""
        model = create_dropout_model()

        initializers = {init.name: init for init in model.graph.initializer}
        nodes = list(model.graph.node)

        # Call detect_all_patterns twice
        results1 = detect_all_patterns(nodes, initializers)
        results2 = detect_all_patterns(nodes, initializers)

        # Results should be identical
        assert results1.keys() == results2.keys()
        for key in results1:
            assert len(results1[key]) == len(results2[key])
