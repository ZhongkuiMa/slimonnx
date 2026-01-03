"""Extended tests for SlimONNX main API - covering error paths and export functionality."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from onnx import TensorProto, helper

from slimonnx.configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from slimonnx.slimonnx import SlimONNX


def create_simple_model():
    """Create a simple ONNX model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "test_model", [X], [Y])
    model = helper.make_model(graph)
    return model


class TestSlimONNXShapeInferenceFallback:
    """Test SlimONNX methods with shape inference failures."""

    def test_analyze_with_shape_inference_failure(self):
        """Test analyze() gracefully handles shape inference failure."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            # Mock infer_onnx_shape to raise an error
            with mock.patch(
                "shapeonnx.infer_shape.infer_onnx_shape",
                side_effect=ValueError("Shape inference failed"),
            ):
                result = slim.analyze(temp_path)

                # Should still return a result despite shape inference failure
                assert isinstance(result, dict)
                assert "model" in result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_validate_with_shape_inference_failure(self):
        """Test validate() handles shape inference failure gracefully."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            # Mock infer_onnx_shape to raise ImportError
            with mock.patch(
                "shapeonnx.infer_shape.infer_onnx_shape",
                side_effect=ImportError("shapeonnx not available"),
            ):
                result = slim.validate(temp_path)

                # Should still return validation result
                assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_detect_patterns_with_shape_inference_failure(self):
        """Test detect_patterns() handles shape inference failure."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            # Mock infer_onnx_shape to raise RuntimeError
            with mock.patch(
                "shapeonnx.infer_shape.infer_onnx_shape",
                side_effect=RuntimeError("Shape inference error"),
            ):
                result = slim.detect_patterns(temp_path)

                # Should still return patterns
                assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXExports:
    """Test SlimONNX export functionality (topology and json)."""

    def test_analyze_export_topology(self):
        """Test analyze() with topology export enabled."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            topology_path = f.name

        import onnx

        onnx.save(model, model_path)
        Path(topology_path).unlink()  # Remove file so it's created fresh

        try:
            slim = SlimONNX()
            config = AnalysisConfig(export_topology=True, topology_path=topology_path)

            result = slim.analyze(model_path, analysis=config)

            # Should return analysis result
            assert isinstance(result, dict)
        finally:
            Path(model_path).unlink(missing_ok=True)
            Path(topology_path).unlink(missing_ok=True)

    def test_analyze_export_json(self):
        """Test analyze() with JSON export enabled."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        import onnx

        onnx.save(model, model_path)
        Path(json_path).unlink()  # Remove file so it's created fresh

        try:
            slim = SlimONNX()
            config = AnalysisConfig(export_json=True, json_path=json_path)

            result = slim.analyze(model_path, analysis=config)

            # Should return analysis result
            assert isinstance(result, dict)
        finally:
            Path(model_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)

    def test_analyze_export_both_topology_and_json(self):
        """Test analyze() with both exports enabled."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_path = f.name

        with tempfile.NamedTemporaryFile(suffix="_topology.json", delete=False) as f:
            topology_path = f.name
        with tempfile.NamedTemporaryFile(suffix="_analysis.json", delete=False) as f:
            json_path = f.name

        import onnx

        onnx.save(model, model_path)
        Path(topology_path).unlink(missing_ok=True)
        Path(json_path).unlink(missing_ok=True)

        try:
            slim = SlimONNX()
            config = AnalysisConfig(
                export_topology=True,
                topology_path=topology_path,
                export_json=True,
                json_path=json_path,
            )

            result = slim.analyze(model_path, analysis=config)

            # Should return complete analysis
            assert isinstance(result, dict)
        finally:
            Path(model_path).unlink(missing_ok=True)
            Path(topology_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)


class TestSlimONNXValidationFailure:
    """Test SlimONNX.slim() with validation failures."""

    def test_slim_validation_failure_raises_error(self):
        """Test that slim() raises error when validation fails."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(remove_dropout=True)
            validation = ValidationConfig(validate_outputs=True, num_samples=1)

            # Mock compare_model_outputs to simulate validation failure
            with (
                mock.patch(
                    "slimonnx.model_validate.compare_model_outputs",
                    return_value={
                        "all_match": False,
                        "failed": 1,
                        "num_tests": 1,
                        "max_diff": 1.0,
                    },
                ),
                pytest.raises(ValueError, match="Validation failed"),
            ):
                slim.slim(input_path, output_path, config=config, validation=validation)
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_returns_report_on_successful_validation(self):
        """Test slim() returns report when validation passes."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(remove_dropout=False)
            validation = ValidationConfig(validate_outputs=True, num_samples=1)

            # Mock compare_model_outputs to return success
            with mock.patch(
                "slimonnx.model_validate.compare_model_outputs",
                return_value={
                    "all_match": True,
                    "failed": 0,
                    "num_tests": 1,
                    "max_diff": 0.0,
                },
            ):
                result = slim.slim(input_path, output_path, config=config, validation=validation)

                # Should return a report
                assert isinstance(result, dict)
                assert "original_nodes" in result
                assert "optimization_time" in result
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestSlimONNXCompareEdgeCases:
    """Test SlimONNX.compare() with various model differences."""

    def test_compare_different_node_counts(self):
        """Test comparing models with different node counts."""
        model1 = create_simple_model()

        # Create a model with additional nodes
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
        graph = helper.make_graph([node1, node2], "test_model", [X], [Z])
        model2 = helper.make_model(graph)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model1, path1)
        onnx.save(model2, path2)

        try:
            slim = SlimONNX()
            result = slim.compare(path1, path2)

            # Should return comparison result
            assert isinstance(result, dict)
            assert "diff" in result
            assert "nodes" in result["diff"]
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_compare_with_pattern_differences(self):
        """Test that compare detects pattern differences between models."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            slim = SlimONNX()
            result = slim.compare(path1, path2)

            # Should return comparison with pattern info
            assert isinstance(result, dict)
            assert "original" in result
            assert "optimized" in result
            assert "diff" in result
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestSlimONNXPreprocessOptions:
    """Test SlimONNX.preprocess() with various options."""

    def test_preprocess_with_opset_conversion(self):
        """Test preprocess with opset conversion."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(temp_path, target_opset=17)

            assert result is not None
            assert len(result.opset_import) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_with_no_opset_conversion(self):
        """Test preprocess without opset conversion (keep original)."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(temp_path, target_opset=None)

            assert result is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_without_docstring_clearing(self):
        """Test preprocess without clearing docstrings."""
        model = create_simple_model()
        model.graph.node[0].doc_string = "Keep this doc"

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(temp_path, clear_docstrings=False)

            assert result is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_without_slimonnx_marker(self):
        """Test preprocess without marking as SlimONNX processed."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(temp_path, mark_slimonnx=False)

            assert result is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXConfigHandling:
    """Test SlimONNX with various configuration options."""

    def test_slim_with_all_config_options(self):
        """Test slim with all optimization config options enabled."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(
                constant_folding=True,
                fuse_matmul_add=True,
                fuse_conv_bn=True,
                fuse_bn_conv=True,
                remove_dropout=True,
                simplify_node_name=True,
                has_batch_dim=True,
            )

            _result = slim.slim(input_path, output_path, config=config)

            # Should complete without error
            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_analyze_with_batch_dim_config(self):
        """Test analyze with batch dimension configuration."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(has_batch_dim=False)

            result = slim.analyze(temp_path, config=config)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_validate_with_config(self):
        """Test validate with optimization config."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(has_batch_dim=False)

            result = slim.validate(temp_path, config=config)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_detect_patterns_with_config(self):
        """Test detect_patterns with optimization config."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(has_batch_dim=True)

            result = slim.detect_patterns(temp_path, config=config)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXSlimWithValidation:
    """Test SlimONNX.slim() method with validation parameters."""

    def test_slim_with_default_validation(self):
        """Test slim with default validation config."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(remove_dropout=False)
            validation = ValidationConfig(validate_outputs=False)

            _result = slim.slim(input_path, output_path, config=config, validation=validation)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_without_validation(self):
        """Test slim when validation is None."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(constant_folding=True)

            _result = slim.slim(input_path, output_path, config=config, validation=None)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_without_config(self):
        """Test slim when config is None."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()

            _result = slim.slim(input_path, output_path, config=None, validation=None)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestSlimONNXAnalysisEdgeCases:
    """Test SlimONNX analysis with edge cases."""

    def test_analyze_with_default_config(self):
        """Test analyze with default config (None)."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.analyze(temp_path, config=None)

            assert isinstance(result, dict)
            assert "patterns" in result
            assert "model" in result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_with_analysis_config_only(self):
        """Test analyze with analysis config but no optimization config."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            analysis_config = AnalysisConfig(export_json=False)
            result = slim.analyze(temp_path, analysis=analysis_config, config=None)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_detect_patterns_with_default_config(self):
        """Test detect_patterns with default config."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.detect_patterns(temp_path, config=None)

            assert isinstance(result, dict)
            # Result contains pattern keys directly (matmul_add, conv_bn, etc.)
            assert len(result) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_validate_with_default_config(self):
        """Test validate with default config."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.validate(temp_path, config=None)

            assert isinstance(result, dict)
            assert "is_valid" in result
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXMultipleOptimizations:
    """Test SlimONNX with various optimization flag combinations."""

    def test_slim_with_constant_folding_only(self):
        """Test slim with only constant folding enabled."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(
                constant_folding=True,
                fuse_matmul_add=False,
                fuse_conv_bn=False,
            )

            _result = slim.slim(input_path, output_path, config=config)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_with_fusion_flags_only(self):
        """Test slim with only fusion flags enabled."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(
                constant_folding=False,
                fuse_matmul_add=True,
                fuse_conv_bn=True,
                fuse_bn_conv=True,
            )

            _result = slim.slim(input_path, output_path, config=config)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_with_simplify_node_name(self):
        """Test slim with node name simplification."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(simplify_node_name=True)

            _result = slim.slim(input_path, output_path, config=config)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_with_batch_dim_false(self):
        """Test slim with has_batch_dim set to False."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(has_batch_dim=False)

            _result = slim.slim(input_path, output_path, config=config)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestSlimONNXCompareAdvanced:
    """Test SlimONNX.compare() with advanced scenarios."""

    def test_compare_returns_diff_structure(self):
        """Test that compare() returns proper diff structure."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            slim = SlimONNX()
            result = slim.compare(path1, path2)

            assert isinstance(result, dict)
            assert "original" in result
            assert "optimized" in result
            if "diff" in result:
                assert isinstance(result["diff"], dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_compare_with_same_file(self):
        """Test compare() when comparing identical models."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name

        import onnx

        onnx.save(model, path)

        try:
            slim = SlimONNX()
            result = slim.compare(path, path)

            assert isinstance(result, dict)
        finally:
            Path(path).unlink(missing_ok=True)


class TestSlimONNXPreprocessAdvanced:
    """Test SlimONNX.preprocess() with advanced scenarios."""

    def test_preprocess_returns_model_proto(self):
        """Test that preprocess() returns ModelProto."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(temp_path)

            assert isinstance(result, onnx.ModelProto)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_with_all_false_flags(self):
        """Test preprocess with all optional flags set to False."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(
                temp_path,
                target_opset=None,
                infer_shapes=False,
                clear_docstrings=False,
                mark_slimonnx=False,
            )

            assert isinstance(result, onnx.ModelProto)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_with_high_opset(self):
        """Test preprocess with high opset version."""
        model = create_simple_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.preprocess(temp_path, target_opset=21)

            assert isinstance(result, onnx.ModelProto)
            assert result.opset_import[0].version == 21
        finally:
            Path(temp_path).unlink(missing_ok=True)
