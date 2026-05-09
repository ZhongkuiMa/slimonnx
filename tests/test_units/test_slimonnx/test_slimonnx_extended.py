"""Extended tests for SlimONNX main API - covering error paths and export functionality."""

import tempfile
from pathlib import Path
from unittest import mock

import onnx
import pytest
from onnx import TensorProto, helper

from slimonnx.configs import AnalysisConfig, OptimizationConfig, ValidationConfig


def temp_model_files():
    """Create temporary model file paths."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    return path


@pytest.fixture
def temp_input_output_paths():
    """Create temporary input/output paths for slim operations."""
    input_path = temp_model_files()
    output_path = temp_model_files()
    return input_path, output_path


class TestSlimONNXShapeInferenceFallback:
    """Test SlimONNX methods with shape inference failures."""

    @pytest.mark.parametrize(
        ("method_name", "exception_side_effect"),
        [
            ("analyze", ValueError("Shape inference failed")),
            ("validate", ImportError("shapeonnx not available")),
            ("detect_patterns", RuntimeError("Shape inference error")),
        ],
    )
    def test_handles_shape_inference_errors(
        self, method_name, exception_side_effect, slim_onnx, temp_onnx_file
    ):
        """Test API methods handle shape inference failures gracefully."""
        with mock.patch(
            "shapeonnx.infer_shape.infer_onnx_shape",
            side_effect=exception_side_effect,
        ):
            result = getattr(slim_onnx, method_name)(temp_onnx_file)
            assert isinstance(result, dict)


class TestSlimONNXConfigDefaults:
    """Test SlimONNX methods with default configuration options."""

    @pytest.mark.parametrize("use_analysis_config", [False, True])
    def test_analyze_with_optional_analysis_config(
        self, temp_onnx_file, slim_onnx, use_analysis_config
    ):
        """Test analyze with and without AnalysisConfig."""
        if use_analysis_config:
            analysis_config = AnalysisConfig(export_json=False)
            result = slim_onnx.analyze(temp_onnx_file, analysis=analysis_config, config=None)
        else:
            result = slim_onnx.analyze(temp_onnx_file, config=None)

        assert isinstance(result, dict)
        assert "patterns" in result
        assert "model" in result


class TestSlimONNXExports:
    """Test SlimONNX export functionality (topology and json)."""

    @pytest.mark.parametrize(
        ("export_topology", "export_json"),
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_supports_topology_and_json_export_options(
        self, export_topology, export_json, simple_model, slim_onnx
    ):
        """Test analyze() respects export_topology and export_json flags."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            model_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            topology_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        onnx.save(simple_model, model_path)
        Path(topology_path).unlink(missing_ok=True)
        Path(json_path).unlink(missing_ok=True)

        try:
            config = AnalysisConfig(
                export_topology=export_topology,
                topology_path=topology_path,
                export_json=export_json,
                json_path=json_path,
            )
            result = slim_onnx.analyze(model_path, analysis=config)
            assert isinstance(result, dict)
        finally:
            Path(model_path).unlink(missing_ok=True)
            Path(topology_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)


class TestSlimONNXValidationFailure:
    """Test SlimONNX.slim() with validation failures."""

    def test_raises_error_when_validation_fails(self, simple_model, slim_onnx):
        """Test slim() raises ValueError when validation fails."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        onnx.save(simple_model, input_path)

        try:
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
                slim_onnx.slim(input_path, output_path, config=config, validation=validation)
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_returns_report_dict_on_successful_validation(self, simple_model, slim_onnx):
        """Test slim() returns report dict when validation passes."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        onnx.save(simple_model, input_path)

        try:
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
                result = slim_onnx.slim(
                    input_path, output_path, config=config, validation=validation
                )

                # Should return a report
                assert isinstance(result, dict)
                assert "original_nodes" in result
                assert "optimization_time" in result
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestSlimONNXCompareEdgeCases:
    """Test SlimONNX.compare() with various model differences."""

    def test_compare_different_node_counts(self, slim_onnx):
        """Test comparing models with different node counts."""
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

        # Create simple model for path1
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        graph = helper.make_graph([node], "test_model", [X], [Y])
        model1 = helper.make_model(graph)

        onnx.save(model1, path1)
        onnx.save(model2, path2)

        try:
            result = slim_onnx.compare(path1, path2)

            # Should return comparison result
            assert isinstance(result, dict)
            assert "diff" in result
            assert "nodes" in result["diff"]
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_detects_pattern_differences_in_models(self, temp_model_pair, slim_onnx):
        """Test compare() detects pattern differences between models."""
        path1, path2 = temp_model_pair

        result = slim_onnx.compare(path1, path2)

        # Should return comparison with pattern info
        assert isinstance(result, dict)
        assert "original" in result
        assert "optimized" in result
        assert "diff" in result


class TestSlimONNXPreprocessing:
    """Test SlimONNX.preprocess() with various options."""

    @pytest.mark.parametrize(
        ("target_opset", "clear_docstrings", "mark_slimonnx"),
        [
            (17, True, True),
            (None, False, True),
            (None, False, False),
        ],
    )
    def test_preprocess_with_flag_combinations(
        self, target_opset, clear_docstrings, mark_slimonnx, simple_model, slim_onnx
    ):
        """Test preprocess respects target_opset, clear_docstrings, and mark_slimonnx flags."""
        if not clear_docstrings:
            simple_model.graph.node[0].doc_string = "Keep this doc"

        temp_path = temp_model_files()

        onnx.save(simple_model, temp_path)

        try:
            kwargs = {
                "clear_docstrings": clear_docstrings,
                "mark_slimonnx": mark_slimonnx,
            }
            if target_opset is not None:
                kwargs["target_opset"] = target_opset
            result = slim_onnx.preprocess(temp_path, **kwargs)
            assert result
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXConfigHandling:
    """Test SlimONNX with various configuration options."""

    @pytest.mark.parametrize("batch_dim", [True, False])
    def test_configures_batch_dimension_flag(self, temp_onnx_file, slim_onnx, batch_dim):
        """Test analyze with has_batch_dim configuration."""
        config = OptimizationConfig(has_batch_dim=batch_dim)
        result = slim_onnx.analyze(temp_onnx_file, config=config)
        assert isinstance(result, dict)


class TestSlimONNXSlimWithValidation:
    """Test SlimONNX.slim() method with validation parameters."""

    def test_slim_succeeds_with_various_validation_configs(self, simple_model, slim_onnx):
        """Test slim() with different validation parameter combinations."""
        test_cases = [
            (
                OptimizationConfig(remove_dropout=False),
                ValidationConfig(validate_outputs=False),
                "default validation",
            ),
            (OptimizationConfig(constant_folding=True), None, "no validation"),
            (None, None, "no config or validation"),
        ]

        for config, validation, description in test_cases:
            input_path = temp_model_files()
            output_path = temp_model_files()

            onnx.save(simple_model, input_path)

            try:
                _result = slim_onnx.slim(
                    input_path, output_path, config=config, validation=validation
                )
                assert Path(output_path).exists(), f"Failed for case: {description}"
            finally:
                Path(input_path).unlink(missing_ok=True)
                Path(output_path).unlink(missing_ok=True)


class TestSlimONNXPatternDetection:
    """Test SlimONNX pattern detection with various configurations."""

    @pytest.mark.parametrize("use_config", [False, True])
    def test_detects_patterns_with_optional_config(self, temp_onnx_file, slim_onnx, use_config):
        """Test detect_patterns with and without OptimizationConfig."""
        if use_config:
            config = OptimizationConfig(has_batch_dim=True)
            result = slim_onnx.detect_patterns(temp_onnx_file, config=config)
        else:
            result = slim_onnx.detect_patterns(temp_onnx_file, config=None)

        assert isinstance(result, dict)
        # Result contains pattern keys directly
        assert len(result) > 0

    @pytest.mark.parametrize("use_config", [False, True])
    def test_validates_model_with_optional_config(self, temp_onnx_file, slim_onnx, use_config):
        """Test validate with and without OptimizationConfig."""
        if use_config:
            config = OptimizationConfig(has_batch_dim=False)
            result = slim_onnx.validate(temp_onnx_file, config=config)
        else:
            result = slim_onnx.validate(temp_onnx_file, config=None)

        assert isinstance(result, dict)
        assert "is_valid" in result


class TestSlimONNXOptimizationFlags:
    """Test SlimONNX with various optimization flag combinations."""

    @pytest.mark.parametrize(
        ("constant_folding", "fuse_matmul_add", "fuse_conv_bn", "simplify_name", "batch_dim"),
        [
            (True, False, False, False, True),
            (False, True, True, False, True),
            (False, False, False, True, True),
            (True, True, True, True, False),
        ],
    )
    def test_slim_with_optimization_flag_combinations(
        self,
        simple_model,
        slim_onnx,
        constant_folding,
        fuse_matmul_add,
        fuse_conv_bn,
        simplify_name,
        batch_dim,
    ):
        """Test slim() with various optimization flag combinations."""
        input_path = temp_model_files()
        output_path = temp_model_files()

        onnx.save(simple_model, input_path)

        try:
            config = OptimizationConfig(
                constant_folding=constant_folding,
                fuse_matmul_add=fuse_matmul_add,
                fuse_conv_bn=fuse_conv_bn,
                fuse_bn_conv=fuse_conv_bn,
                simplify_node_name=simplify_name,
                has_batch_dim=batch_dim,
            )

            _result = slim_onnx.slim(input_path, output_path, config=config)

            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestSlimONNXCompareAdvanced:
    """Test SlimONNX.compare() with advanced scenarios."""

    def test_returns_structured_diff_with_original_and_optimized_keys(
        self, temp_model_pair, slim_onnx
    ):
        """Test compare() returns dict with original, optimized, and diff keys."""
        path1, path2 = temp_model_pair

        result = slim_onnx.compare(path1, path2)

        assert isinstance(result, dict)
        assert "original" in result
        assert "optimized" in result
        if "diff" in result:
            assert isinstance(result["diff"], dict)

    def test_compare_with_same_file(self, temp_onnx_file, slim_onnx):
        """Test compare() when comparing identical models."""
        result = slim_onnx.compare(temp_onnx_file, temp_onnx_file)

        assert isinstance(result, dict)


class TestSlimONNXPreprocessOutput:
    """Test SlimONNX.preprocess() output and return types."""

    @pytest.mark.parametrize("target_opset", [None, 17, 21])
    def test_preprocess_returns_model_proto_with_opset_versions(
        self, temp_onnx_file, slim_onnx, target_opset
    ):
        """Test that preprocess() returns ModelProto with correct opset version."""
        if target_opset is not None:
            result = slim_onnx.preprocess(temp_onnx_file, target_opset=target_opset)
            assert isinstance(result, onnx.ModelProto)
            assert result.opset_import[0].version == target_opset
        else:
            result = slim_onnx.preprocess(
                temp_onnx_file,
                infer_shapes=False,
                clear_docstrings=False,
                mark_slimonnx=False,
            )
            assert isinstance(result, onnx.ModelProto)
