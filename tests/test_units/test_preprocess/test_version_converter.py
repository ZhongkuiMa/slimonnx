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
        assert RECOMMENDED_OPSET
        assert MIN_TESTED_OPSET
        assert MAX_TESTED_OPSET

    def test_opset_constants_range(self):
        """Test that opset constants are in valid range."""
        assert MIN_TESTED_OPSET > 0
        assert MAX_TESTED_OPSET > MIN_TESTED_OPSET
        assert MIN_TESTED_OPSET <= RECOMMENDED_OPSET <= MAX_TESTED_OPSET

    def test_slimonnx_version_defined(self):
        """Test that SlimONNX version is defined."""
        assert isinstance(SLIMONNX_VERSION, str)
        assert len(SLIMONNX_VERSION) > 0


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

    def test_returns_model_proto(self):
        """Test that convert returns ModelProto."""
        model = create_simple_model()
        result = convert_model_version(model)
        assert isinstance(result, onnx.ModelProto)


class TestLoadAndPreprocess:
    """Test load_and_preprocess function."""

    @pytest.fixture(autouse=True)
    def cleanup_temp(self):
        """Cleanup temp files after each test."""
        self._temp_files = []
        yield
        for path in self._temp_files:
            Path(path).unlink(missing_ok=True)

    def _save_and_track(self, model):
        """Save model and track for cleanup."""
        path = save_model_to_temp(model)
        self._temp_files.append(path)
        return path

    def test_loads_and_preprocesses_model_with_defaults(self):
        """Test basic load and preprocess with all defaults."""
        model = create_simple_model()
        model_path = self._save_and_track(model)

        result = load_and_preprocess(model_path)

        assert isinstance(result, onnx.ModelProto)
        assert "SlimONNX" in result.producer_name

    def test_with_opset_conversion(self):
        """Test load and preprocess with opset conversion."""
        model = create_simple_model()
        model_path = self._save_and_track(model)
        target_opset = 19

        result = load_and_preprocess(model_path, target_opset=target_opset, infer_shapes=False)

        assert isinstance(result, onnx.ModelProto)
        assert result.opset_import[0].version == target_opset

    @pytest.mark.parametrize(
        ("clear_docstrings", "expected_doc"),
        [
            pytest.param(False, "Test doc", id="preserves_when_disabled"),
            pytest.param(True, "", id="clears_when_enabled"),
        ],
    )
    def test_docstring_handling(self, clear_docstrings, expected_doc):
        """Test docstring clearing behavior with different flag values."""
        model = create_simple_model()
        model.graph.node[0].doc_string = "Test doc"
        model_path = self._save_and_track(model)

        result = load_and_preprocess(
            model_path, clear_docstrings=clear_docstrings, infer_shapes=False
        )

        assert result.graph.node[0].doc_string == expected_doc

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"check_model": False, "infer_shapes": False}, id="skip_model_check"),
            pytest.param(
                {"infer_shapes": False, "clear_docstrings": False}, id="skip_shape_inference"
            ),
        ],
    )
    def test_skips_optional_processing(self, kwargs):
        """Test load and preprocess with optional processing disabled."""
        model = create_simple_model()
        model_path = self._save_and_track(model)

        result = load_and_preprocess(model_path, **kwargs)

        assert isinstance(result, onnx.ModelProto)

    @pytest.mark.parametrize(
        ("mark_slimonnx", "should_be_present"),
        [
            pytest.param(False, False, id="omits_when_disabled"),
            pytest.param(True, True, id="adds_when_enabled"),
        ],
    )
    def test_slimonnx_marker_handling(self, mark_slimonnx, should_be_present):
        """Test SlimONNX marker presence based on flag value."""
        model = create_simple_model()
        model_path = self._save_and_track(model)

        result = load_and_preprocess(model_path, mark_slimonnx=mark_slimonnx, infer_shapes=False)

        if should_be_present:
            assert f"SlimONNX-{SLIMONNX_VERSION}" in result.producer_name
            assert f"Processed by SlimONNX v{SLIMONNX_VERSION}" in result.doc_string
        else:
            assert "SlimONNX" not in result.producer_name

    def test_with_all_options_enabled(self):
        """Test load and preprocess with all options enabled."""
        model = create_simple_model()
        model.graph.node[0].doc_string = "Test doc"
        model_path = self._save_and_track(model)
        target_opset = 20

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

    def test_returns_model_proto(self):
        """Test that load_and_preprocess returns ModelProto."""
        model = create_simple_model()
        model_path = self._save_and_track(model)

        result = load_and_preprocess(model_path, infer_shapes=False)
        assert isinstance(result, onnx.ModelProto)

    def test_raises_on_nonexistent_file(self):
        """Test load_and_preprocess with nonexistent file."""
        with pytest.raises(FileNotFoundError, match=r".*"):
            load_and_preprocess("/nonexistent/path/model.onnx")

    def test_preserves_opset_when_none(self):
        """Test load_and_preprocess with None target_opset skips conversion."""
        model = create_simple_model()
        original_opset = model.opset_import[0].version
        model_path = self._save_and_track(model)

        result = load_and_preprocess(model_path, target_opset=None, infer_shapes=False)

        assert result.opset_import[0].version == original_opset
