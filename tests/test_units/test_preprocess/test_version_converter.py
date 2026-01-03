"""Tests for ONNX version conversion utilities."""

import tempfile
import warnings
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from slimonnx.preprocess.version_converter import (
    MAX_TESTED_OPSET,
    MIN_TESTED_OPSET,
    RECOMMENDED_OPSET,
    SLIMONNX_VERSION,
    convert_model_version,
    load_and_preprocess,
)


def create_simple_model():
    """Create a simple ONNX model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

    graph = helper.make_graph([node], "test_model", [X], [Y])
    model = helper.make_model(graph)
    return model


def save_model_to_temp(model):
    """Save model to temporary file."""
    tmpdir = tempfile.gettempdir()
    model_path = Path(tmpdir) / "test_model_version.onnx"
    onnx.save(model, str(model_path))
    return str(model_path)


class TestVersionConstants:
    """Test version conversion constants."""

    def test_opset_constants_defined(self):
        """Test that opset constants are defined."""
        assert RECOMMENDED_OPSET is not None
        assert MIN_TESTED_OPSET is not None
        assert MAX_TESTED_OPSET is not None

    def test_opset_constants_range(self):
        """Test that opset constants are in valid range."""
        assert MIN_TESTED_OPSET > 0
        assert MAX_TESTED_OPSET > MIN_TESTED_OPSET
        assert MIN_TESTED_OPSET <= RECOMMENDED_OPSET <= MAX_TESTED_OPSET

    def test_slimonnx_version_defined(self):
        """Test that SlimONNX version is defined."""
        assert SLIMONNX_VERSION is not None
        assert isinstance(SLIMONNX_VERSION, str)


class TestConvertModelVersion:
    """Test convert_model_version function."""

    def test_convert_with_default_opset(self):
        """Test model conversion with default opset."""
        model = create_simple_model()

        result = convert_model_version(model)

        assert isinstance(result, onnx.ModelProto)
        assert result.opset_import[0].version == RECOMMENDED_OPSET

    def test_convert_to_specific_opset(self):
        """Test model conversion to specific opset."""
        model = create_simple_model()
        target_opset = 19

        result = convert_model_version(model, target_opset=target_opset)

        assert isinstance(result, onnx.ModelProto)
        assert result.opset_import[0].version == target_opset

    def test_convert_same_opset_no_change(self):
        """Test conversion with same opset returns similar model."""
        model = create_simple_model()
        original_opset = model.opset_import[0].version

        result = convert_model_version(model, target_opset=original_opset)

        assert isinstance(result, onnx.ModelProto)
        # Should maintain same opset
        assert result.opset_import[0].version == original_opset

    def test_convert_warns_outside_range(self):
        """Test warning for opset outside tested range."""
        model = create_simple_model()
        # Use opset below minimum tested range but still valid
        out_of_range_opset = 15  # Below MIN_TESTED_OPSET (17)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _result = convert_model_version(
                model, target_opset=out_of_range_opset, warn_on_diff=True
            )

            # Should have warning about outside range
            assert len(w) > 0
            # Check for either the range warning or conversion failed warning
            warning_messages = [str(x.message).lower() for x in w]
            assert any(
                "outside tested range" in msg or "version conversion failed" in msg
                for msg in warning_messages
            )

    def test_convert_no_warn_when_disabled(self):
        """Test no warning when warn_on_diff=False."""
        model = create_simple_model()
        out_of_range_opset = 100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _result = convert_model_version(
                model, target_opset=out_of_range_opset, warn_on_diff=False
            )

            # Should not warn about range when disabled
            range_warnings = [x for x in w if "outside tested range" in str(x.message).lower()]
            assert len(range_warnings) == 0

    def test_convert_recommended_opset_no_warn(self):
        """Test no warning for recommended opset."""
        model = create_simple_model()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _result = convert_model_version(model, target_opset=RECOMMENDED_OPSET)

            # Should not warn for recommended opset
            range_warnings = [x for x in w if "outside tested range" in str(x.message).lower()]
            assert len(range_warnings) == 0

    def test_convert_within_range_no_warn(self):
        """Test no warning for opsets within tested range."""
        model = create_simple_model()
        test_opset = MIN_TESTED_OPSET

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _result = convert_model_version(model, target_opset=test_opset)

            # Should not warn for opsets in range
            range_warnings = [x for x in w if "outside tested range" in str(x.message).lower()]
            assert len(range_warnings) == 0

    def test_convert_returns_model_proto(self):
        """Test that convert returns ModelProto."""
        model = create_simple_model()
        result = convert_model_version(model)
        assert isinstance(result, onnx.ModelProto)


class TestLoadAndPreprocess:
    """Test load_and_preprocess function."""

    def test_load_and_preprocess_basic(self):
        """Test basic load and preprocess with all defaults."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path)

            assert isinstance(result, onnx.ModelProto)
            # Should have SlimONNX marker
            assert "SlimONNX" in result.producer_name
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_with_opset_conversion(self):
        """Test load and preprocess with opset conversion."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)
        target_opset = 19

        try:
            result = load_and_preprocess(model_path, target_opset=target_opset, infer_shapes=False)

            assert isinstance(result, onnx.ModelProto)
            assert result.opset_import[0].version == target_opset
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_skip_docstring_clear(self):
        """Test load and preprocess skipping docstring clearing."""
        model = create_simple_model()
        # Add docstring
        model.graph.node[0].doc_string = "Test doc"
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, clear_docstrings=False, infer_shapes=False)

            # Docstring should still be there
            assert result.graph.node[0].doc_string == "Test doc"
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_clear_docstring(self):
        """Test load and preprocess with docstring clearing."""
        model = create_simple_model()
        # Add docstring
        model.graph.node[0].doc_string = "Test doc"
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, clear_docstrings=True, infer_shapes=False)

            # Docstring should be cleared
            assert result.graph.node[0].doc_string == ""
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_skip_check(self):
        """Test load and preprocess skipping model check."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, check_model=False, infer_shapes=False)

            assert isinstance(result, onnx.ModelProto)
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_skip_shape_inference(self):
        """Test load and preprocess skipping shape inference."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, infer_shapes=False, clear_docstrings=False)

            assert isinstance(result, onnx.ModelProto)
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_skip_slimonnx_marker(self):
        """Test load and preprocess skipping SlimONNX marker."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, mark_slimonnx=False, infer_shapes=False)

            # Should not have SlimONNX marker
            assert "SlimONNX" not in result.producer_name
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_marks_slimonnx(self):
        """Test load and preprocess marks with SlimONNX."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, mark_slimonnx=True, infer_shapes=False)

            # Should have SlimONNX marker
            assert f"SlimONNX-{SLIMONNX_VERSION}" in result.producer_name
            assert f"Processed by SlimONNX v{SLIMONNX_VERSION}" in result.doc_string
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_all_options(self):
        """Test load and preprocess with all options enabled."""
        model = create_simple_model()
        model.graph.node[0].doc_string = "Test doc"
        model_path = save_model_to_temp(model)
        target_opset = 20

        try:
            result = load_and_preprocess(
                model_path,
                target_opset=target_opset,
                infer_shapes=True,
                check_model=True,
                clear_docstrings=True,
                mark_slimonnx=True,
            )

            assert isinstance(result, onnx.ModelProto)
            assert result.opset_import[0].version == target_opset
            assert result.graph.node[0].doc_string == ""
            assert "SlimONNX" in result.producer_name
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_returns_model_proto(self):
        """Test that load_and_preprocess returns ModelProto."""
        model = create_simple_model()
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, infer_shapes=False)
            assert isinstance(result, onnx.ModelProto)
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_load_and_preprocess_nonexistent_file(self):
        """Test load_and_preprocess with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess("/nonexistent/path/model.onnx")

    def test_load_and_preprocess_none_opset(self):
        """Test load_and_preprocess with None target_opset skips conversion."""
        model = create_simple_model()
        original_opset = model.opset_import[0].version
        model_path = save_model_to_temp(model)

        try:
            result = load_and_preprocess(model_path, target_opset=None, infer_shapes=False)

            # Should keep original opset
            assert result.opset_import[0].version == original_opset
        finally:
            Path(model_path).unlink(missing_ok=True)
