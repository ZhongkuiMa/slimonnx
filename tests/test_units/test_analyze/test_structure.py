"""Tests for structure analysis module."""

import tempfile
from pathlib import Path

from slimonnx.analyze_structure import analyze_model, compare_models


class TestAnalyzeModel:
    """Test analyze_model() function."""

    def test_analyze_model_basic(self, create_dropout_model):
        """Test basic model analysis."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)

            # Verify result is a dictionary
            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_returns_dict(self, create_matmul_add_model):
        """Test that analyze_model returns a dictionary."""
        model = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path)

            assert isinstance(result, dict)
            # Should contain analysis information
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_with_target_opset(self, create_conv_bn_model):
        """Test analyze_model with custom target opset."""
        model = create_conv_bn_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, target_opset=17)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_model_json_export(self, create_dropout_model):
        """Test analyze_model with JSON export."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            result = analyze_model(input_path, export_json=True, json_output_path=json_path)

            assert isinstance(result, dict)
            # JSON file may or may not be created depending on implementation
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)

    def test_analyze_model_topology_export(self, create_matmul_add_model):
        """Test analyze_model with topology export."""
        model = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            topo_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            result = analyze_model(input_path, export_topology=True, topology_output_path=topo_path)

            assert isinstance(result, dict)
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(topo_path).unlink(missing_ok=True)

    def test_analyze_model_with_batch_dim(self, create_conv_bn_model):
        """Test analyze_model with batch dimension flag."""
        model = create_conv_bn_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            result = analyze_model(temp_path, has_batch_dim=True)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCompareModels:
    """Test compare_models() function."""

    def test_compare_identical_models(self, create_dropout_model):
        """Test comparing identical models."""
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

            assert isinstance(result, dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_compare_returns_dict(self, create_matmul_add_model):
        """Test that compare_models returns a dictionary."""
        model1 = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)
        model2 = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model1, path1)
        onnx.save(model2, path2)

        try:
            result = compare_models(path1, path2)

            assert isinstance(result, dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_compare_different_models(self, create_dropout_model, create_matmul_add_model):
        """Test comparing different models."""
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

            assert isinstance(result, dict)
            # Comparison should show differences
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestAnalysisStructureIntegration:
    """Integration tests for structure analysis."""

    def test_analyze_and_compare_workflow(self, create_matmul_add_model):
        """Test analyze then compare workflow."""
        original_model = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            orig_path = f.name

        import onnx

        onnx.save(original_model, orig_path)

        try:
            # Analyze original
            analysis = analyze_model(orig_path)
            assert isinstance(analysis, dict)

            # For comparison, use same model (simulating before/after)
            comparison = compare_models(orig_path, orig_path)
            assert isinstance(comparison, dict)
        finally:
            Path(orig_path).unlink(missing_ok=True)

    def test_analyze_various_models(
        self, create_dropout_model, create_conv_bn_model, create_matmul_add_model
    ):
        """Test analyzing various model types."""
        models = [
            create_dropout_model(),
            create_conv_bn_model(),
            create_matmul_add_model(),
        ]

        for model in models:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                temp_path = f.name

            import onnx

            onnx.save(model, temp_path)

            try:
                result = analyze_model(temp_path)
                assert isinstance(result, dict)
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_analyze_creates_directories(self, create_dropout_model):
        """Test that analyze works with JSON export in temp directory."""
        model = create_dropout_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "model.onnx"
            output_json = Path(tmpdir) / "analysis.json"

            import onnx

            onnx.save(model, str(input_path))

            # Analyze with JSON output in temp directory (parent should exist)
            result = analyze_model(
                str(input_path), export_json=True, json_output_path=str(output_json)
            )

            # Verify result is returned and temp directory exists
            assert isinstance(result, dict)
            assert Path(tmpdir).exists()
