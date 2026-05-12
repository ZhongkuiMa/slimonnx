"""Tests for structure analysis module."""

__docformat__ = "restructuredtext"

import tempfile
from pathlib import Path

import onnx
import pytest

from slimonnx.analyze_structure import analyze_model, compare_models


class TestAnalyzeModel:
    """Test analyze_model() function."""

    def test_returns_dict(self, create_dropout_model, temp_onnx_path):
        """Test that analyze_model returns a dictionary."""
        model = create_dropout_model()
        onnx.save(model, temp_onnx_path)

        result = analyze_model(temp_onnx_path)

        assert isinstance(result, dict)

    def test_returns_dict_with_different_models(self, create_matmul_add_model, temp_onnx_path):
        """Test that analyze_model returns a dictionary for different model types."""
        model = create_matmul_add_model()
        onnx.save(model, temp_onnx_path)

        result = analyze_model(temp_onnx_path)

        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        ("model_fixture", "kwargs"),
        [
            pytest.param("create_conv_bn_model", {"target_opset": 17}, id="with_target_opset"),
            pytest.param("create_conv_bn_model", {"has_batch_dim": True}, id="with_batch_dim"),
        ],
    )
    def test_returns_dict_with_option(self, model_fixture, kwargs, temp_onnx_path, request):
        """Test analyze_model with various options returns a dictionary."""
        model = request.getfixturevalue(model_fixture)()
        onnx.save(model, temp_onnx_path)

        result = analyze_model(temp_onnx_path, **kwargs)

        assert isinstance(result, dict)

    def test_returns_dict_with_json_export(
        self, create_dropout_model, temp_onnx_path, temp_json_path
    ):
        """Test analyze_model with JSON export."""
        model = create_dropout_model()
        onnx.save(model, temp_onnx_path)

        result = analyze_model(temp_onnx_path, export_json=True, json_output_path=temp_json_path)

        assert isinstance(result, dict)

    def test_returns_dict_with_topology_export(
        self, create_matmul_add_model, temp_onnx_path, temp_topo_path
    ):
        """Test analyze_model with topology export."""
        model = create_matmul_add_model()
        onnx.save(model, temp_onnx_path)

        result = analyze_model(
            temp_onnx_path, export_topology=True, topology_output_path=temp_topo_path
        )

        assert isinstance(result, dict)


class TestCompareModels:
    """Test compare_models() function."""

    def test_returns_dict_for_identical(self, create_dropout_model, temp_onnx_path):
        """Test comparing identical models."""
        model = create_dropout_model()
        onnx.save(model, temp_onnx_path)

        # Create second temp file via fixture context
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        try:
            onnx.save(model, path2)
            result = compare_models(temp_onnx_path, path2)

            assert isinstance(result, dict)
        finally:
            Path(path2).unlink(missing_ok=True)

    def test_returns_dict_for_same_specs(self, create_matmul_add_model, temp_onnx_path):
        """Test that compare_models returns a dictionary for models with same specs."""
        model1 = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)
        model2 = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)

        onnx.save(model1, temp_onnx_path)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        try:
            onnx.save(model2, path2)
            result = compare_models(temp_onnx_path, path2)

            assert isinstance(result, dict)
        finally:
            Path(path2).unlink(missing_ok=True)

    def test_returns_dict_for_different_types(
        self, create_dropout_model, create_matmul_add_model, temp_onnx_path
    ):
        """Test comparing different model types."""
        model1 = create_dropout_model()
        model2 = create_matmul_add_model()

        onnx.save(model1, temp_onnx_path)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        try:
            onnx.save(model2, path2)
            result = compare_models(temp_onnx_path, path2)

            assert isinstance(result, dict)
        finally:
            Path(path2).unlink(missing_ok=True)


class TestAnalysisStructureIntegration:
    """Integration tests for structure analysis."""

    def test_analyze_and_compare_workflow(self, create_matmul_add_model, temp_onnx_path):
        """Test analyze then compare workflow."""
        original_model = create_matmul_add_model()
        onnx.save(original_model, temp_onnx_path)

        analysis = analyze_model(temp_onnx_path)
        assert isinstance(analysis, dict)

        comparison = compare_models(temp_onnx_path, temp_onnx_path)
        assert isinstance(comparison, dict)

    def test_analyze_various_models(
        self, create_dropout_model, create_conv_bn_model, create_matmul_add_model, temp_onnx_path
    ):
        """Test analyzing various model types."""
        models = [
            create_dropout_model(),
            create_conv_bn_model(),
            create_matmul_add_model(),
        ]

        for model in models:
            onnx.save(model, temp_onnx_path)
            result = analyze_model(temp_onnx_path)
            assert isinstance(result, dict)
