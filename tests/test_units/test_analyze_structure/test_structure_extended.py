"""Extended tests for structure analysis module."""

__docformat__ = "restructuredtext"

import tempfile
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest

from slimonnx.analyze_structure import analyze_model, compare_models


class TestAnalyzeModelReportStructure:
    """Test report structure and contents from analyze_model."""

    @pytest.mark.parametrize(
        ("model_fixture", "expected_keys"),
        [
            (
                "create_dropout_model",
                [
                    "model_path",
                ],
            ),
            (
                "create_conv_bn_model",
                [
                    "model_path",
                ],
            ),
        ],
    )
    def test_report_contains_required_keys(self, request, model_fixture, expected_keys):
        """Test that analyze_model report contains all required keys."""
        model = request.getfixturevalue(model_fixture)()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            for key in expected_keys:
                assert key in result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocessing_contains_required_info(self, create_conv_bn_model):
        """Test preprocessing information in report."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert "preprocessing" in result
            preprocessing = result["preprocessing"]
            assert "original_opset" in preprocessing
            assert "original_ir_version" in preprocessing
            assert "shape_inference" in preprocessing
            assert preprocessing["shape_inference"] in ("success", "failed")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_optimization_opportunities_types_valid(self, create_dropout_model):
        """Test optimization_opportunities structure."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            opp = result["optimization_opportunities"]
            assert "total_fusible" in opp
            assert "total_redundant" in opp
            assert "estimated_reduction" in opp
            assert isinstance(opp["total_fusible"], int)
            assert isinstance(opp["total_redundant"], int)
            assert isinstance(opp["estimated_reduction"], int)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_patterns_has_required_structure(self, create_matmul_add_model):
        """Test patterns dictionary structure."""
        model = create_matmul_add_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            patterns = result["patterns"]
            assert isinstance(patterns, dict)
            for pattern_info in patterns.values():
                assert "count" in pattern_info
                assert "category" in pattern_info
                assert isinstance(pattern_info["count"], int)
                assert isinstance(pattern_info["category"], str)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_structure_contains_node_count(self, create_conv_bn_model):
        """Test structure information in report."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            structure = result["structure"]
            assert isinstance(structure, dict)
            assert "node_count" in structure
            assert isinstance(structure["node_count"], int)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAnalyzeModelErrorHandling:
    """Test error handling in analyze_model."""

    @pytest.mark.parametrize(
        ("invalid_path", "exception_types"),
        [
            ("/nonexistent/path/model.onnx", (FileNotFoundError, Exception)),
            ("", (FileNotFoundError, ValueError, OSError)),
        ],
    )
    def test_raises_on_invalid_path(self, invalid_path, exception_types):
        """Test analyze_model with invalid paths."""
        with pytest.raises(exception_types, match=r"No such file or directory"):
            analyze_model(invalid_path)

    @pytest.mark.parametrize("target_opset", [-1, 0])
    def test_handles_edge_case_opsets(self, create_dropout_model, target_opset):
        """Test analyze_model with invalid opset values."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, target_opset=target_opset)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAnalyzeModelEdgeCases:
    """Test edge cases for analyze_model."""

    @pytest.mark.parametrize(
        ("model_fixture", "has_batch_dim", "target_opset"),
        [
            ("create_dropout_model", False, None),
            ("create_conv_bn_model", True, 99),
        ],
    )
    def test_handles_edge_case_options(self, request, model_fixture, has_batch_dim, target_opset):
        """Test analyze_model with various edge case options."""
        model = request.getfixturevalue(model_fixture)()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            kwargs = {"has_batch_dim": has_batch_dim}
            if target_opset is not None:
                kwargs["target_opset"] = target_opset
            result = analyze_model(temp_path, **kwargs)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        ("export_json", "export_topology"),
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_export_files_with_options(self, create_dropout_model, export_json, export_topology):
        """Test analyze_model with various export options."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(
                temp_path, export_json=export_json, export_topology=export_topology
            )
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_path.replace(".onnx", "_analysis.json")).unlink(missing_ok=True)
            Path(temp_path.replace(".onnx", "_topology.json")).unlink(missing_ok=True)


class TestAnalyzeModelShapeInference:
    """Test shape inference in analyze_model."""

    @pytest.mark.parametrize(
        ("mock_behavior", "expected_status"),
        [
            ({"return_value": {"output": [1, 10]}}, "success"),
            ({"side_effect": ValueError("Shape inference failed")}, "failed"),
            ({"side_effect": ImportError("Module not found")}, "failed"),
        ],
    )
    @patch("slimonnx.analyze_structure.infer_onnx_shape")
    def test_shape_inference_status(
        self, mock_infer, create_dropout_model, mock_behavior, expected_status
    ):
        """Test shape inference status with various outcomes."""
        for key, value in mock_behavior.items():
            setattr(mock_infer, key, value)

        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert "preprocessing" in result
            assert result["preprocessing"]["shape_inference"] == expected_status
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCompareModelsReportStructure:
    """Test compare_models report structure."""

    def test_report_contains_required_sections(self, create_dropout_model):
        """Test compare_models report contains required keys."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)
            assert "original" in result
            assert "optimized" in result
            assert "changes" in result
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_changes_section_has_required_fields(
        self, create_dropout_model, create_matmul_add_model
    ):
        """Test changes structure in comparison report."""
        model1 = create_dropout_model()
        model2 = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        onnx.save(model1, path1)
        onnx.save(model2, path2)

        try:
            result = compare_models(path1, path2)
            assert "changes" in result
            changes = result["changes"]
            assert "node_reduction" in changes
            assert "node_reduction_pct" in changes
            assert "patterns_resolved" in changes
            assert isinstance(changes["node_reduction"], int)
            assert isinstance(changes["node_reduction_pct"], float)
            assert isinstance(changes["patterns_resolved"], dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_original_and_optimized_are_full_analyses(self, create_dropout_model):
        """Test original and optimized analysis in comparison."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)
            assert isinstance(result["original"], dict)
            assert isinstance(result["optimized"], dict)
            assert "patterns" in result["original"]
            assert "patterns" in result["optimized"]
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestCompareModelsEdgeCases:
    """Test edge cases for compare_models."""

    def test_identical_models_show_zero_reduction(self, create_dropout_model):
        """Test comparison when node counts are identical."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)
            assert "changes" in result
            assert result["changes"]["node_reduction"] == 0
            assert result["changes"]["node_reduction_pct"] == 0.0
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_empty_patterns_resolved_is_valid(self, create_dropout_model):
        """Test comparison when no patterns are resolved."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)
            assert "changes" in result
            assert "patterns_resolved" in result["changes"]
            assert isinstance(result["changes"]["patterns_resolved"], dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestAnalyzeAndCompareIntegration:
    """Integration tests for full analyze and compare workflows."""

    def test_analyze_returns_correct_model_path(self, create_dropout_model):
        """Test that analyze_model report contains correct model path."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert result["model_path"] == temp_path
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocessing_preserves_original_opset(self, create_conv_bn_model):
        """Test that original opset is preserved in report."""
        model = create_conv_bn_model()
        original_opset = model.opset_import[0].version if model.opset_import else 0

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert "preprocessing" in result
            assert result["preprocessing"]["original_opset"] == original_opset
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_compare_produces_valid_results_with_same_model(self, create_conv_bn_model):
        """Test comparing models with different opsets."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)
            assert isinstance(result, dict)
            assert "changes" in result
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_analyze_multiple_models_consistency(
        self, create_dropout_model, create_matmul_add_model
    ):
        """Test analyzing multiple different models."""
        models = [create_dropout_model(), create_matmul_add_model()]
        results = []

        for model in models:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                temp_path = f.name

            onnx.save(model, temp_path)

            try:
                result = analyze_model(temp_path)
                results.append(result)
            finally:
                Path(temp_path).unlink(missing_ok=True)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)
            assert "patterns" in result
            assert "structure" in result


class TestAnalyzeModelPatternCategories:
    """Test pattern categorization in analysis."""

    def test_patterns_have_category_field(self, create_conv_bn_model):
        """Test that patterns have category field."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            patterns = result["patterns"]
            for pattern_info in patterns.values():
                assert "category" in pattern_info
                assert isinstance(pattern_info.get("category"), str)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_opportunities_calculated_from_patterns(self, create_conv_bn_model):
        """Test that optimization opportunities are calculated correctly."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            patterns = result["patterns"]
            opp = result["optimization_opportunities"]

            expected_fusible = sum(
                p["count"] for p in patterns.values() if p.get("category") == "fusion"
            )
            expected_redundant = sum(
                p["count"] for p in patterns.values() if p.get("category") == "redundant"
            )
            expected_total = expected_fusible + expected_redundant

            assert opp["total_fusible"] == expected_fusible
            assert opp["total_redundant"] == expected_redundant
            assert opp["estimated_reduction"] == expected_total
        finally:
            Path(temp_path).unlink(missing_ok=True)
