"""Extended tests for ONNX attribute extraction and validation."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import helper, numpy_helper

from slimonnx.optimize_onnx._onnx_attrs import (
    get_attrs_avgpool,
    get_attrs_constant,
    get_attrs_constantofshape,
    get_attrs_resize,
    get_attrs_scatter,
    get_attrs_scatterelement,
    get_attrs_scatternd,
    get_attrs_shape,
    get_onnx_attrs,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import create_initializer


class TestGetAttrsAveragePool:
    """Test get_attrs_avgpool function."""

    @pytest.mark.parametrize(
        ("node_kwargs", "expected"),
        [
            pytest.param(
                {"kernel_shape": [3, 3]},
                {"kernel_shape": (3, 3), "auto_pad": "NOTSET", "ceil_mode": 0},
                id="with_kernel_shape",
            ),
            pytest.param(
                {"kernel_shape": [3, 3], "strides": [2, 2]},
                {"strides": (2, 2)},
                id="with_strides",
            ),
            pytest.param(
                {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
                {"pads": (1, 1, 1, 1)},
                id="with_pads",
            ),
            pytest.param(
                {"kernel_shape": [3, 3], "ceil_mode": 1},
                {"ceil_mode": 1},
                id="with_ceil_mode",
            ),
            pytest.param(
                {"kernel_shape": [3, 3], "count_include_pad": 1},
                {"count_include_pad": 1},
                id="with_count_include_pad",
            ),
        ],
    )
    def test_returns_expected_attrs(self, node_kwargs, expected):
        """Verify get_attrs_avgpool returns expected attribute values."""
        node = helper.make_node("AveragePool", inputs=["X"], outputs=["Y"], **node_kwargs)
        initializers: dict[str, Any] = {}
        result = get_attrs_avgpool(node, initializers)
        for key, value in expected.items():
            assert result[key] == value, f"attr {key}"

    def test_missing_kernel_shape(self):
        """Test AveragePool without kernel_shape (required)."""
        node = helper.make_node("AveragePool", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="kernel_shape"):
            get_attrs_avgpool(node, initializers)

    @pytest.mark.parametrize(
        ("node_kwargs", "match"),
        [
            pytest.param(
                {"kernel_shape": [3, 3], "pads": [1, 2, 3, 4]},
                "Asymmetric padding",
                id="asymmetric_pads",
            ),
            pytest.param(
                {"kernel_shape": [3, 3], "auto_pad": "SAME_UPPER"},
                "auto_pad",
                id="auto_pad",
            ),
        ],
    )
    def test_raises_on_unsupported_config(self, node_kwargs, match):
        """Verify get_attrs_avgpool raises ValueError for unsupported configurations."""
        node = helper.make_node("AveragePool", inputs=["X"], outputs=["Y"], **node_kwargs)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match=match):
            get_attrs_avgpool(node, initializers)


class TestGetAttrsConstant:
    """Test get_attrs_constant function."""

    def test_raises_runtime_error(self):
        """Test Constant node raises RuntimeError."""
        node = helper.make_node("Constant", inputs=[], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(RuntimeError, match="Constant nodes are not supported"):
            get_attrs_constant(node, initializers)

    def test_raises_with_initializers_present(self):
        """Test Constant node with various arguments."""
        node = helper.make_node("Constant", inputs=[], outputs=["Y"])
        initializers = {"dummy": create_initializer("dummy", np.array([1.0]))}
        with pytest.raises(RuntimeError, match="Constant nodes are not supported"):
            get_attrs_constant(node, initializers)


class TestGetAttrsConstantOfShape:
    """Test get_attrs_constantofshape function."""

    def test_value_present_in_result(self):
        """Test ConstantOfShape with value specified."""
        value_tensor = numpy_helper.from_array(np.array([5.0]), "value_tensor")
        node = helper.make_node("ConstantOfShape", inputs=["shape_input"], outputs=["Y"])
        node.attribute.append(helper.make_attribute("value", value_tensor))
        initializers: dict[str, Any] = {}
        result = get_attrs_constantofshape(node, initializers)
        assert "value" in result
        assert isinstance(result["value"], (int, float, np.ndarray, type(value_tensor)))

    def test_raises_on_missing_value(self):
        """Test ConstantOfShape without value (required)."""
        node = helper.make_node("ConstantOfShape", inputs=["shape_input"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="value"):
            get_attrs_constantofshape(node, initializers)


class TestGetAttrsResize:
    """Test get_attrs_resize function."""

    @pytest.mark.parametrize(
        ("node_kwargs", "expected"),
        [
            pytest.param(
                {},
                {"mode": "nearest", "coordinate_transformation_mode": "half_pixel"},
                id="defaults",
            ),
            pytest.param({"mode": "linear"}, {"mode": "linear"}, id="with_mode"),
            pytest.param({"cubic_coeff_a": -0.5}, {"cubic_coeff_a": -0.5}, id="with_cubic_coeff"),
            pytest.param(
                {"nearest_mode": "ceil"}, {"nearest_mode": "ceil"}, id="with_nearest_mode"
            ),
            pytest.param({"antialias": 1}, {"antialias": 1}, id="with_antialias"),
        ],
    )
    def test_returns_expected_attrs(self, node_kwargs, expected):
        """Verify get_attrs_resize returns expected attribute values."""
        node = helper.make_node(
            "Resize", inputs=["X", "roi", "scales"], outputs=["Y"], **node_kwargs
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_resize(node, initializers)
        for key, value in expected.items():
            if isinstance(value, float):
                assert pytest.approx(result[key]) == value, f"attr {key}"
            else:
                assert result[key] == value, f"attr {key}"


class TestGetAttrsScatter:
    """Test get_attrs_scatter function."""

    def test_defaults(self):
        """Test Scatter with default attributes."""
        node = helper.make_node(
            "Scatter", inputs=["data", "indices", "updates"], outputs=["output"]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatter(node, initializers)
        assert result["axis"] == 0
        assert result["reduction"] == "none"

    def test_with_axis(self):
        """Test Scatter with axis specified."""
        node = helper.make_node(
            "Scatter", inputs=["data", "indices", "updates"], outputs=["output"], axis=1
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatter(node, initializers)
        assert result["axis"] == 1

    def test_raises_on_unsupported_reduction(self):
        """Test Scatter with reduction=add (unsupported)."""
        node = helper.make_node(
            "Scatter",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            reduction="add",
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="reduction"):
            get_attrs_scatter(node, initializers)


class TestGetAttrsScatterElements:
    """Test get_attrs_scatterelement function."""

    def test_defaults(self):
        """Test ScatterElements with default attributes."""
        node = helper.make_node(
            "ScatterElements", inputs=["data", "indices", "updates"], outputs=["output"]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatterelement(node, initializers)
        assert result["axis"] == 0
        assert result["reduction"] == "none"

    def test_with_axis(self):
        """Test ScatterElements with axis specified."""
        node = helper.make_node(
            "ScatterElements",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            axis=2,
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatterelement(node, initializers)
        assert result["axis"] == 2

    def test_raises_on_unsupported_reduction(self):
        """Test ScatterElements with reduction=mul (unsupported)."""
        node = helper.make_node(
            "ScatterElements",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            reduction="mul",
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="reduction"):
            get_attrs_scatterelement(node, initializers)


class TestGetAttrsScatterND:
    """Test get_attrs_scatternd function."""

    def test_defaults(self):
        """Test ScatterND with default attributes."""
        node = helper.make_node(
            "ScatterND", inputs=["data", "indices", "updates"], outputs=["output"]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatternd(node, initializers)
        assert result["reduction"] == "none"

    def test_raises_on_unsupported_reduction(self):
        """Test ScatterND with reduction=add (unsupported)."""
        node = helper.make_node(
            "ScatterND",
            inputs=["data", "indices", "updates"],
            outputs=["output"],
            reduction="add",
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="reduction"):
            get_attrs_scatternd(node, initializers)


class TestGetAttrsShape:
    """Test get_attrs_shape function."""

    def test_defaults(self):
        """Test Shape with default attributes."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_shape(node, initializers)
        assert result["start"] == 0
        assert result["end"] == -1

    def test_with_start_zero(self):
        """Test Shape with start=0 (valid)."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"], start=0)
        initializers: dict[str, Any] = {}
        result = get_attrs_shape(node, initializers)
        assert result["start"] == 0

    @pytest.mark.parametrize(
        ("node_kwargs", "match"),
        [
            pytest.param({"start": 1}, "start", id="nonzero_start"),
            pytest.param({"end": 2}, "end", id="nondefault_end"),
        ],
    )
    def test_raises_on_unsupported_bounds(self, node_kwargs, match):
        """Verify get_attrs_shape raises ValueError for unsupported start/end values."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"], **node_kwargs)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match=match):
            get_attrs_shape(node, initializers)


class TestGetAttrsSimpleOperators:
    """Test get_attrs_simple factory and its operators."""

    @pytest.mark.parametrize(
        ("op_name", "inputs", "outputs", "attr_kwargs", "expected"),
        [
            pytest.param("Elu", ["X"], ["Y"], {}, {"alpha": 1.0}, id="elu_defaults"),
            pytest.param(
                "Elu", ["X"], ["Y"], {"alpha": 0.5}, {"alpha": 0.5}, id="elu_custom_alpha"
            ),
            pytest.param("Flatten", ["X"], ["Y"], {}, {"axis": 1}, id="flatten_defaults"),
            pytest.param(
                "Flatten", ["X"], ["Y"], {"axis": 0}, {"axis": 0}, id="flatten_custom_axis"
            ),
            pytest.param("Gather", ["X", "indices"], ["Y"], {}, {"axis": 0}, id="gather_defaults"),
            pytest.param(
                "Gather", ["X", "indices"], ["Y"], {"axis": 1}, {"axis": 1}, id="gather_custom_axis"
            ),
            pytest.param("Gelu", ["X"], ["Y"], {}, {"approximate": "none"}, id="gelu_defaults"),
            pytest.param(
                "Gelu",
                ["X"],
                ["Y"],
                {"approximate": "tanh"},
                {"approximate": "tanh"},
                id="gelu_tanh",
            ),
            pytest.param(
                "Gemm",
                ["A", "B", "C"],
                ["Y"],
                {},
                {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
                id="gemm_defaults",
            ),
            pytest.param(
                "Gemm",
                ["A", "B", "C"],
                ["Y"],
                {"transA": 1},
                {"transA": 1, "transB": 0},
                id="gemm_with_trans",
            ),
            pytest.param("LeakyRelu", ["X"], ["Y"], {}, {"alpha": 0.01}, id="leaky_relu_defaults"),
            pytest.param(
                "LeakyRelu",
                ["X"],
                ["Y"],
                {"alpha": 0.2},
                {"alpha": 0.2},
                id="leaky_relu_custom_alpha",
            ),
            pytest.param("Pad", ["X", "pads"], ["Y"], {}, {"mode": "constant"}, id="pad_defaults"),
            pytest.param(
                "Pad",
                ["X", "pads"],
                ["Y"],
                {"mode": "reflect"},
                {"mode": "reflect"},
                id="pad_reflect",
            ),
            pytest.param("Softmax", ["X"], ["Y"], {}, {"axis": -1}, id="softmax_defaults"),
            pytest.param(
                "Softmax", ["X"], ["Y"], {"axis": 1}, {"axis": 1}, id="softmax_custom_axis"
            ),
            pytest.param(
                "Split",
                ["X"],
                ["Y1", "Y2"],
                {},
                {"axis": 0, "num_outputs": None},
                id="split_defaults",
            ),
            pytest.param(
                "Split",
                ["X"],
                ["Y1", "Y2"],
                {"num_outputs": 2},
                {"num_outputs": 2},
                id="split_with_num_outputs",
            ),
            pytest.param("Unsqueeze", ["X"], ["Y"], {}, {"axes": None}, id="unsqueeze_defaults"),
            pytest.param(
                "Unsqueeze",
                ["X"],
                ["Y"],
                {"axes": [0, 2]},
                {"axes": (0, 2)},
                id="unsqueeze_with_axes",
            ),
            pytest.param(
                "Upsample", ["X", "scales"], ["Y"], {}, {"mode": "nearest"}, id="upsample_defaults"
            ),
            pytest.param(
                "Upsample",
                ["X", "scales"],
                ["Y"],
                {"mode": "linear"},
                {"mode": "linear"},
                id="upsample_linear",
            ),
        ],
    )
    def test_get_onnx_attrs_simple_operators(self, op_name, inputs, outputs, attr_kwargs, expected):
        """Verify get_onnx_attrs returns expected attribute values across simple operators."""
        node = helper.make_node(op_name, inputs=inputs, outputs=outputs, **attr_kwargs)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        for key, value in expected.items():
            if isinstance(value, float):
                assert pytest.approx(result[key]) == value, f"{op_name} attr {key}"
            else:
                assert result[key] == value, f"{op_name} attr {key}"


class TestGetAttrsReduceOperators:
    """Test get_attrs_reduce factory and reduce operators."""

    @pytest.mark.parametrize(
        ("op_name", "attr_kwargs", "expected"),
        [
            pytest.param(
                "ReduceMean",
                {},
                {"keepdims": 1, "noop_with_empty_axes": 0},
                id="reduce_mean_defaults",
            ),
            pytest.param(
                "ReduceMean", {"keepdims": 0}, {"keepdims": 0}, id="reduce_mean_keepdims_0"
            ),
            pytest.param(
                "ReduceSum",
                {},
                {"keepdims": 1, "noop_with_empty_axes": 0},
                id="reduce_sum_defaults",
            ),
            pytest.param("ReduceSum", {"keepdims": 0}, {"keepdims": 0}, id="reduce_sum_keepdims_0"),
        ],
    )
    def test_get_onnx_attrs_reduce_happy_path(self, op_name, attr_kwargs, expected):
        """Verify get_onnx_attrs returns expected attribute values for reduce operators."""
        node = helper.make_node(op_name, inputs=["X"], outputs=["Y"], **attr_kwargs)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        for key, value in expected.items():
            assert result[key] == value, f"{op_name} attr {key}"

    @pytest.mark.parametrize("op_name", ["ReduceMean", "ReduceSum"])
    def test_get_onnx_attrs_reduce_noop_with_empty_axes_unsupported(self, op_name):
        """Verify reduce operators raise ValueError when noop_with_empty_axes=1."""
        node = helper.make_node(op_name, inputs=["X"], outputs=["Y"], noop_with_empty_axes=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="not supported"):
            get_onnx_attrs(node, initializers)
