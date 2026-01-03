"""Tests for ONNX attribute extraction and validation."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import TensorProto, helper

from slimonnx.optimize_onnx._onnx_attrs import (
    check_pads_symmetric,
    get_attrs_argmax,
    get_attrs_batchnorm,
    get_attrs_cast,
    get_attrs_concat,
    get_attrs_conv,
    get_attrs_conv_transpose,
    get_attrs_maxpool,
    get_attrs_reshape,
    get_attrs_transpose,
    get_onnx_attrs,
    infer_kernel_defaults,
    scan_attrs,
    validate_auto_pad,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import create_initializer


class TestScanAttrs:
    """Test scan_attrs function."""

    def test_scan_attrs_empty_attributes(self):
        """Test scanning with no attributes (use defaults)."""
        defaults = {"alpha": 1.0, "beta": 2.0}
        attrs: list[Any] = []
        result = scan_attrs(defaults, attrs)
        assert result == defaults

    def test_scan_attrs_float_attribute(self):
        """Test extracting float attribute."""
        defaults = {"alpha": 1.0}
        attr = helper.make_attribute("alpha", 0.5)
        result = scan_attrs(defaults, [attr])
        assert result["alpha"] == 0.5

    def test_scan_attrs_int_attribute(self):
        """Test extracting int attribute."""
        defaults = {"axis": 0}
        attr = helper.make_attribute("axis", 2)
        result = scan_attrs(defaults, [attr])
        assert result["axis"] == 2

    def test_scan_attrs_floats_attribute(self):
        """Test extracting floats (repeated) attribute."""
        defaults = {"strides": None}
        attr = helper.make_attribute("strides", [2, 3])
        result = scan_attrs(defaults, [attr])
        assert result["strides"] == (2, 3)

    def test_scan_attrs_ints_attribute(self):
        """Test extracting ints (repeated) attribute."""
        defaults = {"kernel_shape": None}
        attr = helper.make_attribute("kernel_shape", [3, 3])
        result = scan_attrs(defaults, [attr])
        assert result["kernel_shape"] == (3, 3)

    def test_scan_attrs_undefined_type(self):
        """Test undefined attribute type returns None."""
        defaults: dict[str, Any] = {}
        attr = helper.AttributeProto()
        attr.name = "undefined_attr"
        attr.type = 0  # UNDEFINED
        result = scan_attrs(defaults, [attr])
        assert result["undefined_attr"] is None


class TestCheckPadsSymmetric:
    """Test check_pads_symmetric function."""

    def test_check_pads_symmetric_valid_1d(self):
        """Test symmetric 1D padding."""
        check_pads_symmetric((1, 1))

    def test_check_pads_symmetric_valid_2d(self):
        """Test symmetric 2D padding."""
        check_pads_symmetric((1, 2, 1, 2))

    def test_check_pads_symmetric_asymmetric_1d(self):
        """Test error on asymmetric 1D padding."""
        with pytest.raises(ValueError, match="Asymmetric padding"):
            check_pads_symmetric((1, 2))

    def test_check_pads_symmetric_asymmetric_2d(self):
        """Test error on asymmetric 2D padding."""
        with pytest.raises(ValueError, match="Asymmetric padding"):
            check_pads_symmetric((1, 2, 3, 4))


class TestInferKernelDefaults:
    """Test infer_kernel_defaults function."""

    def test_infer_kernel_defaults_1d_all_none(self):
        """Test inferring all defaults for 1D kernel."""
        attrs = {"kernel_shape": (3,), "dilations": None, "strides": None, "pads": None}
        result = infer_kernel_defaults(attrs, (3,))
        assert result["dilations"] == (1,)
        assert result["strides"] == (1,)
        assert result["pads"] == (0, 0)

    def test_infer_kernel_defaults_2d_all_none(self):
        """Test inferring all defaults for 2D kernel."""
        attrs = {"kernel_shape": (3, 3), "dilations": None, "strides": None, "pads": None}
        result = infer_kernel_defaults(attrs, (3, 3))
        assert result["dilations"] == (1, 1)
        assert result["strides"] == (1, 1)
        assert result["pads"] == (0, 0, 0, 0)

    def test_infer_kernel_defaults_partial(self):
        """Test inferring only missing defaults."""
        attrs = {
            "kernel_shape": (3, 3),
            "dilations": (2, 2),
            "strides": None,
            "pads": None,
        }
        result = infer_kernel_defaults(attrs, (3, 3))
        assert result["dilations"] == (2, 2)
        assert result["strides"] == (1, 1)


class TestValidateAutoPad:
    """Test validate_auto_pad function."""

    def test_validate_auto_pad_notset(self):
        """Test NOTSET auto_pad (valid)."""
        validate_auto_pad("NOTSET", "Conv")

    def test_validate_auto_pad_same_upper_error(self):
        """Test SAME_UPPER auto_pad (invalid)."""
        with pytest.raises(ValueError, match="auto_pad=SAME_UPPER"):
            validate_auto_pad("SAME_UPPER", "Conv")

    def test_validate_auto_pad_same_lower_error(self):
        """Test SAME_LOWER auto_pad (invalid)."""
        with pytest.raises(ValueError, match="auto_pad=SAME_LOWER"):
            validate_auto_pad("SAME_LOWER", "Conv")


class TestGetAttrsArgmax:
    """Test get_attrs_argmax function."""

    def test_get_attrs_argmax_defaults(self):
        """Test ArgMax with default attributes."""
        node = helper.make_node("ArgMax", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_argmax(node, initializers)
        assert result["axis"] == 0
        assert result["keepdims"] == 1
        assert result["select_last_index"] == 0

    def test_get_attrs_argmax_custom_axis(self):
        """Test ArgMax with custom axis."""
        node = helper.make_node("ArgMax", inputs=["X"], outputs=["Y"], axis=2)
        initializers: dict[str, Any] = {}
        result = get_attrs_argmax(node, initializers)
        assert result["axis"] == 2

    def test_get_attrs_argmax_select_last_index_error(self):
        """Test ArgMax with select_last_index=1 (unsupported)."""
        node = helper.make_node("ArgMax", inputs=["X"], outputs=["Y"], select_last_index=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="select_last_index"):
            get_attrs_argmax(node, initializers)


class TestGetAttrsBatchNorm:
    """Test get_attrs_batchnorm function."""

    def test_get_attrs_batchnorm_defaults(self):
        """Test BatchNormalization with default attributes."""
        node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_batchnorm(node, initializers)
        assert result["epsilon"] == 1e-5
        assert result["momentum"] == 0.9

    def test_get_attrs_batchnorm_training_mode_error(self):
        """Test BatchNormalization with training_mode=1 (unsupported)."""
        node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
            training_mode=1,
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="training_mode"):
            get_attrs_batchnorm(node, initializers)

    def test_get_attrs_batchnorm_multiple_outputs_error(self):
        """Test BatchNormalization with multiple outputs (unsupported)."""
        node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y", "mean_out", "var_out"],
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="outputs"):
            get_attrs_batchnorm(node, initializers)


class TestGetAttrsCast:
    """Test get_attrs_cast function."""

    def test_get_attrs_cast_to_float(self):
        """Test Cast to float."""
        node = helper.make_node("Cast", inputs=["X"], outputs=["Y"], to=TensorProto.FLOAT)
        initializers: dict[str, Any] = {}
        result = get_attrs_cast(node, initializers)
        assert result["to"] == TensorProto.FLOAT

    def test_get_attrs_cast_missing_to(self):
        """Test Cast without to attribute (required)."""
        node = helper.make_node("Cast", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="to"):
            get_attrs_cast(node, initializers)

    def test_get_attrs_cast_saturate_unsupported(self):
        """Test Cast with saturate=0 (unsupported)."""
        node = helper.make_node(
            "Cast", inputs=["X"], outputs=["Y"], to=TensorProto.FLOAT, saturate=0
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="saturate"):
            get_attrs_cast(node, initializers)


class TestGetAttrsConcat:
    """Test get_attrs_concat function."""

    def test_get_attrs_concat_axis_0(self):
        """Test Concat with axis=0."""
        node = helper.make_node("Concat", inputs=["X1", "X2"], outputs=["Y"], axis=0)
        initializers: dict[str, Any] = {}
        result = get_attrs_concat(node, initializers)
        assert result["axis"] == 0

    def test_get_attrs_concat_missing_axis(self):
        """Test Concat without axis attribute (required)."""
        node = helper.make_node("Concat", inputs=["X1", "X2"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="axis"):
            get_attrs_concat(node, initializers)


class TestGetAttrsConv:
    """Test get_attrs_conv function."""

    def test_get_attrs_conv_with_kernel_shape(self):
        """Test Conv with explicit kernel_shape."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_attrs_conv(node, initializers)
        assert result["kernel_shape"] == (3, 3)
        assert result["group"] == 1

    def test_get_attrs_conv_infer_kernel_from_weight(self):
        """Test Conv inferring kernel_shape from weight."""
        weight = create_initializer("W", np.random.randn(32, 3, 3, 3).astype(np.float32))
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        initializers = {"W": weight}
        result = get_attrs_conv(node, initializers)
        assert result["kernel_shape"] == (3, 3)

    def test_get_attrs_conv_auto_pad_error(self):
        """Test Conv with auto_pad=SAME_UPPER (unsupported)."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], auto_pad="SAME_UPPER")
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="auto_pad"):
            get_attrs_conv(node, initializers)

    def test_get_attrs_conv_asymmetric_pad_error(self):
        """Test Conv with asymmetric padding (unsupported)."""
        node = helper.make_node(
            "Conv",
            inputs=["X", "W"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[1, 2, 3, 4],
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="Asymmetric padding"):
            get_attrs_conv(node, initializers)


class TestGetAttrsConvTranspose:
    """Test get_attrs_convtranspose function."""

    def test_get_attrs_convtranspose_defaults(self):
        """Test ConvTranspose with defaults."""
        node = helper.make_node(
            "ConvTranspose", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_conv_transpose(node, initializers)
        assert result["kernel_shape"] == (3, 3)
        assert result["group"] == 1

    def test_get_attrs_conv_transpose_group_error(self):
        """Test ConvTranspose with group>1 (unsupported)."""
        node = helper.make_node(
            "ConvTranspose",
            inputs=["X", "W"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            group=2,
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="group"):
            get_attrs_conv_transpose(node, initializers)


class TestGetAttrsMaxPool:
    """Test get_attrs_maxpool function."""

    def test_get_attrs_maxpool_defaults(self):
        """Test MaxPool with defaults."""
        node = helper.make_node("MaxPool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_attrs_maxpool(node, initializers)
        assert result["kernel_shape"] == (3, 3)

    def test_get_attrs_maxpool_missing_kernel(self):
        """Test MaxPool without kernel_shape (required)."""
        node = helper.make_node("MaxPool", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="kernel_shape"):
            get_attrs_maxpool(node, initializers)

    def test_get_attrs_maxpool_storage_order_error(self):
        """Test MaxPool with storage_order=1 (unsupported)."""
        node = helper.make_node(
            "MaxPool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            storage_order=1,
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="storage_order"):
            get_attrs_maxpool(node, initializers)

    def test_get_attrs_maxpool_multiple_outputs_error(self):
        """Test MaxPool with multiple outputs (unsupported)."""
        node = helper.make_node(
            "MaxPool",
            inputs=["X"],
            outputs=["Y", "indices"],
            kernel_shape=[3, 3],
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="outputs"):
            get_attrs_maxpool(node, initializers)


class TestGetAttrsReshape:
    """Test get_attrs_reshape function."""

    def test_get_attrs_reshape_defaults(self):
        """Test Reshape with defaults."""
        node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_reshape(node, initializers)
        assert result["allowzero"] == 0

    def test_get_attrs_reshape_allowzero_error(self):
        """Test Reshape with allowzero=1 (unsupported)."""
        node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"], allowzero=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="allowzero"):
            get_attrs_reshape(node, initializers)


class TestGetAttrsTranspose:
    """Test get_attrs_transpose function."""

    def test_get_attrs_transpose_perm(self):
        """Test Transpose with perm."""
        node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[1, 0, 2])
        initializers: dict[str, Any] = {}
        result = get_attrs_transpose(node, initializers)
        assert result["perm"] == (1, 0, 2)

    def test_get_attrs_transpose_missing_perm(self):
        """Test Transpose without perm (required)."""
        node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="perm"):
            get_attrs_transpose(node, initializers)


class TestGetOnnxAttrs:
    """Test get_onnx_attrs main function."""

    def test_get_onnx_attrs_conv(self):
        """Test get_onnx_attrs for Conv."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert "kernel_shape" in result
        assert result["kernel_shape"] == (3, 3)

    def test_get_onnx_attrs_batchnorm(self):
        """Test get_onnx_attrs for BatchNormalization."""
        node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert "epsilon" in result

    def test_get_onnx_attrs_unsupported_op(self):
        """Test get_onnx_attrs with unsupported operator."""
        node = helper.make_node("UnsupportedOp", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(NotImplementedError, match="UnsupportedOp"):
            get_onnx_attrs(node, initializers)
