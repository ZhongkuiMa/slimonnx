"""Tests for ONNX attribute extraction and validation."""

__docformat__ = "restructuredtext"

from typing import Any

import numpy as np
import pytest
from _helpers import create_initializer  # type: ignore[import-not-found]
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


class TestScanAttrs:
    """Test scan_attrs function."""

    def test_empty_attributes_returns_defaults(self):
        """Test scanning with no attributes (use defaults)."""
        defaults = {"alpha": 1.0, "beta": 2.0}
        attrs: list[Any] = []
        result = scan_attrs(defaults, attrs)
        assert result == defaults

    @pytest.mark.parametrize(
        ("attr_name", "attr_value", "expected"),
        [
            pytest.param("alpha", 0.5, 0.5, id="float"),
            pytest.param("axis", 2, 2, id="int"),
            pytest.param("strides", [2, 3], (2, 3), id="floats_repeated"),
            pytest.param("kernel_shape", [3, 3], (3, 3), id="ints_repeated"),
        ],
    )
    def test_attribute_type_extraction(self, attr_name, attr_value, expected):
        """Test extracting scalar and repeated attributes by type."""
        defaults: dict[str, Any] = {attr_name: None}
        attr = helper.make_attribute(attr_name, attr_value)
        result = scan_attrs(defaults, [attr])
        assert result[attr_name] == expected

    def test_undefined_type_returns_none(self):
        """Test undefined attribute type returns None."""
        defaults: dict[str, Any] = {}
        attr = helper.AttributeProto()
        attr.name = "undefined_attr"
        attr.type = 0  # UNDEFINED
        result = scan_attrs(defaults, [attr])
        assert result["undefined_attr"] is None


class TestCheckPadsSymmetric:
    """Test check_pads_symmetric function."""

    @pytest.mark.parametrize(
        "pads",
        [
            pytest.param((1, 1), id="valid_1d"),
            pytest.param((1, 2, 1, 2), id="valid_2d"),
        ],
    )
    def test_accepts_symmetric_padding(self, pads):
        """Test symmetric padding does not raise."""
        check_pads_symmetric(pads)

    @pytest.mark.parametrize(
        "pads",
        [
            pytest.param((1, 2), id="asymmetric_1d"),
            pytest.param((1, 2, 3, 4), id="asymmetric_2d"),
        ],
    )
    def test_asymmetric_pads_raises(self, pads):
        """Test asymmetric padding raises ValueError."""
        with pytest.raises(ValueError, match="Asymmetric padding"):
            check_pads_symmetric(pads)


class TestInferKernelDefaults:
    """Test infer_kernel_defaults function."""

    @pytest.mark.parametrize(
        ("kernel_shape", "expected_dilations", "expected_strides", "expected_pads"),
        [
            pytest.param((3,), (1,), (1,), (0, 0), id="1d_all_none"),
            pytest.param((3, 3), (1, 1), (1, 1), (0, 0, 0, 0), id="2d_all_none"),
        ],
    )
    def test_all_none_defaults(
        self, kernel_shape, expected_dilations, expected_strides, expected_pads
    ):
        """Test inferring all defaults when dilations/strides/pads are None."""
        attrs = {
            "kernel_shape": kernel_shape,
            "dilations": None,
            "strides": None,
            "pads": None,
        }
        result = infer_kernel_defaults(attrs, kernel_shape)
        assert result["dilations"] == expected_dilations
        assert result["strides"] == expected_strides
        assert result["pads"] == expected_pads

    def test_partial_defaults(self):
        """Test inferring only missing defaults when dilations already set."""
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

    def test_notset_passes_validation(self):
        """Test NOTSET auto_pad (valid)."""
        validate_auto_pad("NOTSET", "Conv")

    @pytest.mark.parametrize(
        "auto_pad",
        [
            pytest.param("SAME_UPPER", id="same_upper"),
            pytest.param("SAME_LOWER", id="same_lower"),
        ],
    )
    def test_raises_for_unsupported_auto_pad_values(self, auto_pad):
        """Test unsupported auto_pad values raise ValueError."""
        with pytest.raises(ValueError, match=f"auto_pad={auto_pad}"):
            validate_auto_pad(auto_pad, "Conv")


class TestGetAttrsArgmax:
    """Test get_attrs_argmax function."""

    def test_returns_defaults_when_attributes_absent(self):
        """Test ArgMax with default attributes."""
        node = helper.make_node("ArgMax", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_argmax(node, initializers)
        assert result["axis"] == 0
        assert result["keepdims"] == 1
        assert result["select_last_index"] == 0

    def test_custom_axis(self):
        """Test ArgMax with custom axis."""
        node = helper.make_node("ArgMax", inputs=["X"], outputs=["Y"], axis=2)
        initializers: dict[str, Any] = {}
        result = get_attrs_argmax(node, initializers)
        assert result["axis"] == 2

    def test_select_last_index_raises(self):
        """Test ArgMax with select_last_index=1 (unsupported)."""
        node = helper.make_node("ArgMax", inputs=["X"], outputs=["Y"], select_last_index=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="select_last_index"):
            get_attrs_argmax(node, initializers)


class TestGetAttrsBatchNorm:
    """Test get_attrs_batchnorm function."""

    def test_returns_epsilon_and_momentum_defaults(self):
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

    def test_training_mode_raises(self):
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

    def test_multiple_outputs_raises(self):
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

    def test_to_float(self):
        """Test Cast to float."""
        node = helper.make_node("Cast", inputs=["X"], outputs=["Y"], to=TensorProto.FLOAT)
        initializers: dict[str, Any] = {}
        result = get_attrs_cast(node, initializers)
        assert result["to"] == TensorProto.FLOAT

    def test_missing_to_raises(self):
        """Test Cast without to attribute (required)."""
        node = helper.make_node("Cast", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="to"):
            get_attrs_cast(node, initializers)

    def test_saturate_unsupported_raises(self):
        """Test Cast with saturate=0 (unsupported)."""
        node = helper.make_node(
            "Cast", inputs=["X"], outputs=["Y"], to=TensorProto.FLOAT, saturate=0
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="saturate"):
            get_attrs_cast(node, initializers)


class TestGetAttrsConcat:
    """Test get_attrs_concat function."""

    def test_axis_0(self):
        """Test Concat with axis=0."""
        node = helper.make_node("Concat", inputs=["X1", "X2"], outputs=["Y"], axis=0)
        initializers: dict[str, Any] = {}
        result = get_attrs_concat(node, initializers)
        assert result["axis"] == 0

    def test_missing_axis_raises(self):
        """Test Concat without axis attribute (required)."""
        node = helper.make_node("Concat", inputs=["X1", "X2"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="axis"):
            get_attrs_concat(node, initializers)


class TestGetAttrsConv:
    """Test get_attrs_conv function."""

    def test_extracts_explicit_kernel_shape_and_defaults_group(self):
        """Test Conv with explicit kernel_shape."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_attrs_conv(node, initializers)
        assert result["kernel_shape"] == (3, 3)
        assert result["group"] == 1

    def test_infers_kernel_from_weight_tensor(self):
        """Test Conv inferring kernel_shape from weight."""
        weight = create_initializer("W", np.random.randn(32, 3, 3, 3).astype(np.float32))
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        initializers = {"W": weight}
        result = get_attrs_conv(node, initializers)
        assert result["kernel_shape"] == (3, 3)

    def test_auto_pad_raises(self):
        """Test Conv with auto_pad=SAME_UPPER (unsupported)."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], auto_pad="SAME_UPPER")
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="auto_pad"):
            get_attrs_conv(node, initializers)

    def test_asymmetric_pad_raises(self):
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
    """Test get_attrs_conv_transpose function."""

    def test_defaults(self):
        """Test ConvTranspose with defaults."""
        node = helper.make_node(
            "ConvTranspose", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_conv_transpose(node, initializers)
        assert result["kernel_shape"] == (3, 3)
        assert result["group"] == 1

    def test_group_greater_than_one_raises(self):
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

    def test_extracts_kernel_shape_and_defaults(self):
        """Test MaxPool with defaults."""
        node = helper.make_node("MaxPool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_attrs_maxpool(node, initializers)
        assert result["kernel_shape"] == (3, 3)

    def test_missing_kernel_raises(self):
        """Test MaxPool without kernel_shape (required)."""
        node = helper.make_node("MaxPool", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="kernel_shape"):
            get_attrs_maxpool(node, initializers)

    def test_storage_order_raises(self):
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

    def test_multiple_outputs_raises(self):
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

    def test_defaults(self):
        """Test Reshape with defaults."""
        node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_reshape(node, initializers)
        assert result["allowzero"] == 0

    def test_allowzero_raises(self):
        """Test Reshape with allowzero=1 (unsupported)."""
        node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"], allowzero=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="allowzero"):
            get_attrs_reshape(node, initializers)


class TestGetAttrsTranspose:
    """Test get_attrs_transpose function."""

    def test_extracts_perm_attribute(self):
        """Test Transpose with perm."""
        node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[1, 0, 2])
        initializers: dict[str, Any] = {}
        result = get_attrs_transpose(node, initializers)
        assert result["perm"] == (1, 0, 2)

    def test_missing_perm_raises(self):
        """Test Transpose without perm (required)."""
        node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="perm"):
            get_attrs_transpose(node, initializers)


class TestGetOnnxAttrs:
    """Test get_onnx_attrs main function."""

    def test_extracts_conv_attributes(self):
        """Test get_onnx_attrs for Conv."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert "kernel_shape" in result
        assert result["kernel_shape"] == (3, 3)

    def test_extracts_batchnorm_attributes(self):
        """Test get_onnx_attrs for BatchNormalization."""
        node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert "epsilon" in result

    def test_unsupported_op_raises(self):
        """Test get_onnx_attrs with unsupported operator."""
        node = helper.make_node("UnsupportedOp", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(NotImplementedError, match="UnsupportedOp"):
            get_onnx_attrs(node, initializers)
