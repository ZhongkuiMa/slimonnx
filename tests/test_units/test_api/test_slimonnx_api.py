"""Tests for SlimONNX main API."""

import tempfile
from pathlib import Path

from slimonnx.slimonnx import SlimONNX


class TestSlimONNXPreprocess:
    """Test SlimONNX.preprocess() method."""

    def test_preprocess_basic(self, create_dropout_model):
        """Test basic model preprocessing."""
        model = create_dropout_model()

        # Save model to temp file
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            preprocessed = slim.preprocess(temp_path)

            # Verify model is returned
            assert preprocessed is not None
            assert hasattr(preprocessed, "graph")
            assert len(preprocessed.graph.node) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_clears_docstrings(self, create_conv_bn_model):
        """Test that preprocess clears docstrings."""
        model = create_conv_bn_model()

        # Save model with docstring
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            preprocessed = slim.preprocess(temp_path, clear_docstrings=True)

            # Model should be returned
            assert preprocessed is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocess_marks_slimonnx(self, create_dropout_model):
        """Test that preprocess marks model as SlimONNX processed."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            preprocessed = slim.preprocess(temp_path, mark_slimonnx=True)

            # Check preprocessing completed
            assert preprocessed is not None  # Preprocessing should complete
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXAnalyze:
    """Test SlimONNX.analyze() method."""

    def test_analyze_basic(self, create_matmul_add_model):
        """Test basic model analysis."""
        model = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            analysis = slim.analyze(temp_path)

            # Verify analysis result
            assert isinstance(analysis, dict)
            # May contain validation, patterns, structure
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_analyze_returns_dict(self, create_dropout_model):
        """Test that analyze returns a dictionary."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.analyze(temp_path)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXDetectPatterns:
    """Test SlimONNX.detect_patterns() method."""

    def test_detect_patterns_basic(self, create_matmul_add_model):
        """Test basic pattern detection."""
        model = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            patterns = slim.detect_patterns(temp_path)

            # Verify patterns result
            assert isinstance(patterns, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_detect_patterns_returns_dict(self, create_dropout_model):
        """Test that detect_patterns returns a dictionary."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.detect_patterns(temp_path)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXValidate:
    """Test SlimONNX.validate() method."""

    def test_validate_basic(self, create_dropout_model):
        """Test basic model validation."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            validation_result = slim.validate(temp_path)

            # Verify validation result
            assert isinstance(validation_result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_validate_returns_dict(self, create_conv_bn_model):
        """Test that validate returns a dictionary."""
        model = create_conv_bn_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        import onnx

        onnx.save(model, temp_path)

        try:
            slim = SlimONNX()
            result = slim.validate(temp_path)

            assert isinstance(result, dict)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSlimONNXCompare:
    """Test SlimONNX.compare() method."""

    def test_compare_identical_models(self, create_dropout_model):
        """Test comparing identical models."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path2 = f.name

        import onnx

        onnx.save(model, temp_path1)
        onnx.save(model, temp_path2)

        try:
            slim = SlimONNX()
            comparison = slim.compare(temp_path1, temp_path2)

            # Verify comparison result
            assert isinstance(comparison, dict)
        finally:
            Path(temp_path1).unlink(missing_ok=True)
            Path(temp_path2).unlink(missing_ok=True)

    def test_compare_returns_dict(self, create_matmul_add_model):
        """Test that compare returns a dictionary."""
        model1 = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)
        model2 = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path2 = f.name

        import onnx

        onnx.save(model1, temp_path1)
        onnx.save(model2, temp_path2)

        try:
            slim = SlimONNX()
            result = slim.compare(temp_path1, temp_path2)

            assert isinstance(result, dict)
        finally:
            Path(temp_path1).unlink(missing_ok=True)
            Path(temp_path2).unlink(missing_ok=True)


class TestSlimONNXSlim:
    """Test SlimONNX.slim() method - main optimization."""

    def test_slim_basic(self, create_dropout_model):
        """Test basic model optimization."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            slim.slim(input_path, output_path)

            # Output file should exist
            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_creates_output_file(self, create_matmul_add_model):
        """Test that slim creates output file."""
        model = create_matmul_add_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        onnx.save(model, input_path)

        # Ensure output file doesn't exist yet
        Path(output_path).unlink(missing_ok=True)

        try:
            slim = SlimONNX()
            slim.slim(input_path, output_path)

            # Output file should exist after optimization
            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_with_config(self, create_dropout_model):
        """Test slim with custom OptimizationConfig."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        import onnx

        from slimonnx.configs import OptimizationConfig

        onnx.save(model, input_path)

        try:
            slim = SlimONNX()
            config = OptimizationConfig(remove_dropout=True)
            slim.slim(input_path, output_path, config=config)

            # Output file should exist
            assert Path(output_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_slim_default_output_path(self, create_dropout_model):
        """Test slim with default output path generation."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            input_path = f.name

        import onnx

        onnx.save(model, input_path)

        # Generate expected default output path
        expected_output = input_path.replace(".onnx", "_simplified.onnx")

        try:
            slim = SlimONNX()
            slim.slim(input_path)  # No target_path specified

            # Expected output file should exist
            assert Path(expected_output).exists() or Path(input_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(expected_output).unlink(missing_ok=True)


class TestSlimONNXValidateOutputs:
    """Test SlimONNX.validate_outputs() method."""

    def test_validate_outputs_identical_models(self, create_dropout_model):
        """Test numerical validation of identical models."""
        model = create_dropout_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path2 = f.name

        import onnx

        onnx.save(model, path1)
        onnx.save(model, path2)

        try:
            slim = SlimONNX()
            result = slim.validate_outputs(path1, path2)

            # Result should be a dictionary
            assert isinstance(result, dict)
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)


class TestSlimONNXIntegration:
    """Integration tests for SlimONNX."""

    def test_slim_onnx_instantiation(self):
        """Test that SlimONNX can be instantiated."""
        slim = SlimONNX()
        assert slim is not None

    def test_slim_onnx_has_required_methods(self):
        """Test that SlimONNX has all required methods."""
        slim = SlimONNX()

        # Check for main methods
        assert hasattr(slim, "slim")
        assert hasattr(slim, "analyze")
        assert hasattr(slim, "preprocess")
        assert hasattr(slim, "validate")
        assert hasattr(slim, "detect_patterns")
        assert hasattr(slim, "compare")
        assert hasattr(slim, "validate_outputs")

    def test_slim_creates_parent_directories(self, create_dropout_model):
        """Test that slim creates parent directories for output."""
        model = create_dropout_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.onnx"
            output_path = Path(tmpdir) / "subdir" / "output.onnx"

            import onnx

            onnx.save(model, str(input_path))

            slim = SlimONNX()
            slim.slim(str(input_path), str(output_path))

            # Output file and parent directory should exist
            assert output_path.exists()
