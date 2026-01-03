"""Extended tests for structure analysis module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from slimonnx.analyze_structure import analyze_model, compare_models


class TestAnalyzeModelReportStructure:
    """Test report structure and contents from analyze_model."""

    def test_analyze_model_report_keys(self, create_dropout_model):
        """Test that analyze_model report contains all required keys."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)

            # Check top-level keys
            assert "model_path" in result
            assert "preprocessing" in result
            assert "validation" in result
            assert "patterns" in result
            assert "structure" in result
            assert "optimization_opportunities" in result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_preprocessing_info(self, create_conv_bn_model):
        """Test preprocessing information in report."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)

            # Check preprocessing keys
            preprocessing = result["preprocessing"]
            assert "original_opset" in preprocessing
            assert "original_ir_version" in preprocessing
            assert "shape_inference" in preprocessing
            assert preprocessing["shape_inference"] in ("success", "failed")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_optimization_opportunities(self, create_dropout_model):
        """Test optimization_opportunities structure."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)

            opp = result["optimization_opportunities"]
            assert "total_fusible" in opp
            assert "total_redundant" in opp
            assert "estimated_reduction" in opp
            # Should be integers or zero
            assert isinstance(opp["total_fusible"], int)
            assert isinstance(opp["total_redundant"], int)
            assert isinstance(opp["estimated_reduction"], int)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_patterns_structure(self, create_matmul_add_model):
        """Test patterns dictionary structure."""
        model = create_matmul_add_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)

            patterns = result["patterns"]
            assert isinstance(patterns, dict)
            # Each pattern should have count and category
            for pattern_info in patterns.values():
                assert "count" in pattern_info
                assert "category" in pattern_info
                assert isinstance(pattern_info["count"], int)
                # Category should be a string for valid patterns
                assert isinstance(pattern_info["category"], str)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_structure_info(self, create_conv_bn_model):
        """Test structure information in report."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

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

    def test_analyze_model_nonexistent_file(self):
        """Test analyze_model with nonexistent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            analyze_model("/nonexistent/path/model.onnx")

    def test_analyze_model_invalid_onnx_path(self):
        """Test analyze_model with invalid path."""
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            analyze_model("")

    def test_analyze_model_with_invalid_target_opset(self, create_dropout_model):
        """Test analyze_model with invalid target_opset."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            # Negative opset should still work (clamped or ignored)
            result = analyze_model(temp_path, target_opset=-1)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_with_zero_opset(self, create_dropout_model):
        """Test analyze_model with opset=0."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, target_opset=0)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAnalyzeModelEdgeCases:
    """Test edge cases for analyze_model."""

    def test_analyze_model_no_batch_dim(self, create_dropout_model):
        """Test analyze_model with has_batch_dim=False."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, has_batch_dim=False)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_with_high_opset(self, create_conv_bn_model):
        """Test analyze_model with very high target_opset."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, target_opset=99)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_json_export_default_path(self, create_dropout_model):
        """Test JSON export with default path (replaces .onnx with _analysis.json)."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, export_json=True)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)
            # Clean up any generated analysis.json
            analysis_path = temp_path.replace(".onnx", "_analysis.json")
            Path(analysis_path).unlink(missing_ok=True)

    def test_analyze_model_topology_export_default_path(self, create_matmul_add_model):
        """Test topology export with default path."""
        model = create_matmul_add_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, export_topology=True)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)
            # Clean up any generated topology.json
            topo_path = temp_path.replace(".onnx", "_topology.json")
            Path(topo_path).unlink(missing_ok=True)

    def test_analyze_model_both_exports(self, create_dropout_model):
        """Test analyze_model with both JSON and topology exports."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, export_json=True, export_topology=True)
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)
            # Clean up generated files
            Path(temp_path.replace(".onnx", "_analysis.json")).unlink(missing_ok=True)
            Path(temp_path.replace(".onnx", "_topology.json")).unlink(missing_ok=True)


class TestAnalyzeModelShapeInference:
    """Test shape inference in analyze_model."""

    @patch("slimonnx.analyze_structure.infer_onnx_shape")
    def test_analyze_model_shape_inference_success(self, mock_infer, create_dropout_model):
        """Test successful shape inference."""
        mock_infer.return_value = {"output": [1, 10]}
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert result["preprocessing"]["shape_inference"] == "success"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("slimonnx.analyze_structure.infer_onnx_shape")
    def test_analyze_model_shape_inference_failure(self, mock_infer, create_dropout_model):
        """Test shape inference failure handling."""
        mock_infer.side_effect = ValueError("Shape inference failed")
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert result["preprocessing"]["shape_inference"] == "failed"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("slimonnx.analyze_structure.infer_onnx_shape")
    def test_analyze_model_shape_inference_import_error(self, mock_infer, create_dropout_model):
        """Test shape inference with ImportError."""
        mock_infer.side_effect = ImportError("Module not found")
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert result["preprocessing"]["shape_inference"] == "failed"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCompareModelsReportStructure:
    """Test compare_models report structure."""

    def test_compare_models_report_keys(self, create_dropout_model):
        """Test compare_models report contains required keys."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

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

    def test_compare_models_changes_structure(self, create_dropout_model, create_matmul_add_model):
        """Test changes structure in comparison report."""
        model1 = create_dropout_model()
        model2 = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model1, path1)
        onnx.save(model2, path2)

        try:
            result = compare_models(path1, path2)

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

    def test_compare_models_original_optimized_analysis(self, create_dropout_model):
        """Test original and optimized analysis in comparison."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)

            # Both should have full analysis reports
            assert isinstance(result["original"], dict)
            assert isinstance(result["optimized"], dict)
            assert "patterns" in result["original"]
            assert "patterns" in result["optimized"]
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestCompareModelsEdgeCases:
    """Test edge cases for compare_models."""

    def test_compare_identical_node_count(self, create_dropout_model):
        """Test comparison when node counts are identical."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)

            # Node reduction should be 0
            assert result["changes"]["node_reduction"] == 0
            # Percentage should be 0
            assert result["changes"]["node_reduction_pct"] == 0.0
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_compare_zero_patterns_resolved(self, create_dropout_model):
        """Test comparison when no patterns are resolved."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)

            # Should have patterns_resolved dict (possibly empty)
            assert isinstance(result["changes"]["patterns_resolved"], dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestAnalyzeAndCompareIntegration:
    """Integration tests for full analyze and compare workflows."""

    def test_analyze_returns_valid_model_path(self, create_dropout_model):
        """Test that analyze_model report contains correct model path."""
        model = create_dropout_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            assert result["model_path"] == temp_path
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_opset_preservation(self, create_conv_bn_model):
        """Test that original opset is preserved in report."""
        model = create_conv_bn_model()
        original_opset = model.opset_import[0].version if model.opset_import else 0

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            # Original opset should be preserved
            assert result["preprocessing"]["original_opset"] == original_opset
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_compare_different_opsets(self, create_conv_bn_model):
        """Test comparing models with different opsets."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            result = compare_models(path1, path2)
            # Should still produce valid comparison
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

            import onnx

            onnx.save(model, temp_path)

            try:
                result = analyze_model(temp_path)
                results.append(result)
            finally:
                Path(temp_path).unlink(missing_ok=True)

        # Both results should be valid
        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)
            assert "patterns" in result
            assert "structure" in result


class TestAnalyzeModelPatternCategories:
    """Test pattern categorization in analysis."""

    def test_pattern_categories_present(self, create_conv_bn_model):
        """Test that patterns have category field."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            patterns = result["patterns"]

            # Check pattern categories exist
            for pattern_info in patterns.values():
                assert "category" in pattern_info
                # Category should be a string
                assert isinstance(pattern_info.get("category"), str)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_optimization_opportunities_calculation(self, create_conv_bn_model):
        """Test that optimization opportunities are calculated correctly."""
        model = create_conv_bn_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)
            patterns = result["patterns"]
            opp = result["optimization_opportunities"]

            # Calculate expected totals
            expected_fusible = sum(
                p["count"] for p in patterns.values() if p.get("category") == "fusion"
            )
            expected_redundant = sum(
                p["count"] for p in patterns.values() if p.get("category") == "redundant"
            )
            expected_total = expected_fusible + expected_redundant

            # Verify calculations
            assert opp["total_fusible"] == expected_fusible
            assert opp["total_redundant"] == expected_redundant
            assert opp["estimated_reduction"] == expected_total
        finally:
            Path(temp_path).unlink(missing_ok=True)
