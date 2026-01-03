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

    def test_get_attrs_avgpool_with_kernel_shape(self):
        """Test AveragePool with kernel_shape specified."""
        node = helper.make_node("AveragePool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3])
        initializers: dict[str, Any] = {}
        result = get_attrs_avgpool(node, initializers)
        assert result["kernel_shape"] == (3, 3)
        assert result["auto_pad"] == "NOTSET"
        assert result["ceil_mode"] == 0

    def test_get_attrs_avgpool_missing_kernel_shape(self):
        """Test AveragePool without kernel_shape (required)."""
        node = helper.make_node("AveragePool", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="kernel_shape"):
            get_attrs_avgpool(node, initializers)

    def test_get_attrs_avgpool_with_strides(self):
        """Test AveragePool with strides specified."""
        node = helper.make_node(
            "AveragePool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3], strides=[2, 2]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_avgpool(node, initializers)
        assert result["strides"] == (2, 2)

    def test_get_attrs_avgpool_with_pads(self):
        """Test AveragePool with symmetric pads."""
        node = helper.make_node(
            "AveragePool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_avgpool(node, initializers)
        assert result["pads"] == (1, 1, 1, 1)

    def test_get_attrs_avgpool_asymmetric_pads_error(self):
        """Test AveragePool with asymmetric pads (unsupported)."""
        node = helper.make_node(
            "AveragePool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[1, 2, 3, 4],
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="Asymmetric padding"):
            get_attrs_avgpool(node, initializers)

    def test_get_attrs_avgpool_auto_pad_error(self):
        """Test AveragePool with auto_pad=SAME_UPPER (unsupported)."""
        node = helper.make_node(
            "AveragePool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            auto_pad="SAME_UPPER",
        )
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="auto_pad"):
            get_attrs_avgpool(node, initializers)

    def test_get_attrs_avgpool_with_ceil_mode(self):
        """Test AveragePool with ceil_mode=1."""
        node = helper.make_node(
            "AveragePool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3], ceil_mode=1
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_avgpool(node, initializers)
        assert result["ceil_mode"] == 1

    def test_get_attrs_avgpool_with_count_include_pad(self):
        """Test AveragePool with count_include_pad=1."""
        node = helper.make_node(
            "AveragePool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            count_include_pad=1,
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_avgpool(node, initializers)
        assert result["count_include_pad"] == 1


class TestGetAttrsConstant:
    """Test get_attrs_constant function."""

    def test_get_attrs_constant_raises_error(self):
        """Test Constant node raises RuntimeError."""
        node = helper.make_node("Constant", inputs=[], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(RuntimeError, match="Constant nodes are not supported"):
            get_attrs_constant(node, initializers)

    def test_get_attrs_constant_with_args(self):
        """Test Constant node with various arguments."""
        node = helper.make_node("Constant", inputs=[], outputs=["Y"])
        initializers = {"dummy": create_initializer("dummy", np.array([1.0]))}
        with pytest.raises(RuntimeError, match="Constant nodes are not supported"):
            get_attrs_constant(node, initializers)


class TestGetAttrsConstantOfShape:
    """Test get_attrs_constantofshape function."""

    def test_get_attrs_constantofshape_with_value(self):
        """Test ConstantOfShape with value specified."""
        value_tensor = numpy_helper.from_array(np.array([5.0]), "value_tensor")
        node = helper.make_node("ConstantOfShape", inputs=["shape_input"], outputs=["Y"])
        node.attribute.append(helper.make_attribute("value", value_tensor))
        initializers: dict[str, Any] = {}
        result = get_attrs_constantofshape(node, initializers)
        assert result["value"] is not None

    def test_get_attrs_constantofshape_missing_value(self):
        """Test ConstantOfShape without value (required)."""
        node = helper.make_node("ConstantOfShape", inputs=["shape_input"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="value"):
            get_attrs_constantofshape(node, initializers)


class TestGetAttrsResize:
    """Test get_attrs_resize function."""

    def test_get_attrs_resize_defaults(self):
        """Test Resize with default attributes."""
        node = helper.make_node("Resize", inputs=["X", "roi", "scales"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_resize(node, initializers)
        assert result["mode"] == "nearest"
        assert result["coordinate_transformation_mode"] == "half_pixel"

    def test_get_attrs_resize_with_mode(self):
        """Test Resize with mode=linear."""
        node = helper.make_node(
            "Resize", inputs=["X", "roi", "scales"], outputs=["Y"], mode="linear"
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_resize(node, initializers)
        assert result["mode"] == "linear"

    def test_get_attrs_resize_with_cubic_coeff(self):
        """Test Resize with cubic_coeff_a."""
        node = helper.make_node(
            "Resize",
            inputs=["X", "roi", "scales"],
            outputs=["Y"],
            cubic_coeff_a=-0.5,
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_resize(node, initializers)
        assert result["cubic_coeff_a"] == -0.5

    def test_get_attrs_resize_with_nearest_mode(self):
        """Test Resize with nearest_mode."""
        node = helper.make_node(
            "Resize",
            inputs=["X", "roi", "scales"],
            outputs=["Y"],
            nearest_mode="ceil",
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_resize(node, initializers)
        assert result["nearest_mode"] == "ceil"

    def test_get_attrs_resize_with_antialias(self):
        """Test Resize with antialias=1."""
        node = helper.make_node("Resize", inputs=["X", "roi", "scales"], outputs=["Y"], antialias=1)
        initializers: dict[str, Any] = {}
        result = get_attrs_resize(node, initializers)
        assert result["antialias"] == 1


class TestGetAttrsScatter:
    """Test get_attrs_scatter function."""

    def test_get_attrs_scatter_defaults(self):
        """Test Scatter with default attributes."""
        node = helper.make_node(
            "Scatter", inputs=["data", "indices", "updates"], outputs=["output"]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatter(node, initializers)
        assert result["axis"] == 0
        assert result["reduction"] == "none"

    def test_get_attrs_scatter_with_axis(self):
        """Test Scatter with axis specified."""
        node = helper.make_node(
            "Scatter", inputs=["data", "indices", "updates"], outputs=["output"], axis=1
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatter(node, initializers)
        assert result["axis"] == 1

    def test_get_attrs_scatter_with_unsupported_reduction(self):
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

    def test_get_attrs_scatterelement_defaults(self):
        """Test ScatterElements with default attributes."""
        node = helper.make_node(
            "ScatterElements", inputs=["data", "indices", "updates"], outputs=["output"]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatterelement(node, initializers)
        assert result["axis"] == 0
        assert result["reduction"] == "none"

    def test_get_attrs_scatterelement_with_axis(self):
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

    def test_get_attrs_scatterelement_with_unsupported_reduction(self):
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

    def test_get_attrs_scatternd_defaults(self):
        """Test ScatterND with default attributes."""
        node = helper.make_node(
            "ScatterND", inputs=["data", "indices", "updates"], outputs=["output"]
        )
        initializers: dict[str, Any] = {}
        result = get_attrs_scatternd(node, initializers)
        assert result["reduction"] == "none"

    def test_get_attrs_scatternd_with_unsupported_reduction(self):
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

    def test_get_attrs_shape_defaults(self):
        """Test Shape with default attributes."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_attrs_shape(node, initializers)
        assert result["start"] == 0
        assert result["end"] == -1

    def test_get_attrs_shape_with_start(self):
        """Test Shape with start=0 (valid)."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"], start=0)
        initializers: dict[str, Any] = {}
        result = get_attrs_shape(node, initializers)
        assert result["start"] == 0

    def test_get_attrs_shape_with_nonzero_start_error(self):
        """Test Shape with start!=0 (unsupported)."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"], start=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="start"):
            get_attrs_shape(node, initializers)

    def test_get_attrs_shape_with_nondefault_end_error(self):
        """Test Shape with end!=-1 (unsupported)."""
        node = helper.make_node("Shape", inputs=["X"], outputs=["Y"], end=2)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="end"):
            get_attrs_shape(node, initializers)


class TestGetAttrsSimpleOperators:
    """Test get_attrs_simple factory and its operators."""

    def test_get_onnx_attrs_elu_defaults(self):
        """Test Elu with default alpha."""
        node = helper.make_node("Elu", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["alpha"] == 1.0

    def test_get_onnx_attrs_elu_custom_alpha(self):
        """Test Elu with custom alpha."""
        node = helper.make_node("Elu", inputs=["X"], outputs=["Y"], alpha=0.5)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["alpha"] == 0.5

    def test_get_onnx_attrs_flatten_defaults(self):
        """Test Flatten with default axis."""
        node = helper.make_node("Flatten", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == 1

    def test_get_onnx_attrs_flatten_custom_axis(self):
        """Test Flatten with custom axis."""
        node = helper.make_node("Flatten", inputs=["X"], outputs=["Y"], axis=0)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == 0

    def test_get_onnx_attrs_gather_defaults(self):
        """Test Gather with default axis."""
        node = helper.make_node("Gather", inputs=["X", "indices"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == 0

    def test_get_onnx_attrs_gather_custom_axis(self):
        """Test Gather with custom axis."""
        node = helper.make_node("Gather", inputs=["X", "indices"], outputs=["Y"], axis=1)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == 1

    def test_get_onnx_attrs_gelu_defaults(self):
        """Test Gelu with default approximate."""
        node = helper.make_node("Gelu", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["approximate"] == "none"

    def test_get_onnx_attrs_gelu_tanh_approximate(self):
        """Test Gelu with approximate=tanh."""
        node = helper.make_node("Gelu", inputs=["X"], outputs=["Y"], approximate="tanh")
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["approximate"] == "tanh"

    def test_get_onnx_attrs_gemm_defaults(self):
        """Test Gemm with default attributes."""
        node = helper.make_node("Gemm", inputs=["A", "B", "C"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["alpha"] == 1.0
        assert result["beta"] == 1.0
        assert result["transA"] == 0
        assert result["transB"] == 0

    def test_get_onnx_attrs_gemm_with_trans(self):
        """Test Gemm with transA=1."""
        node = helper.make_node("Gemm", inputs=["A", "B", "C"], outputs=["Y"], transA=1)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["transA"] == 1
        assert result["transB"] == 0

    def test_get_onnx_attrs_leaky_relu_defaults(self):
        """Test LeakyRelu with default alpha."""
        node = helper.make_node("LeakyRelu", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["alpha"] == 0.01

    def test_get_onnx_attrs_leaky_relu_custom_alpha(self):
        """Test LeakyRelu with custom alpha."""
        node = helper.make_node("LeakyRelu", inputs=["X"], outputs=["Y"], alpha=0.2)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert pytest.approx(result["alpha"]) == 0.2

    def test_get_onnx_attrs_pad_defaults(self):
        """Test Pad with default mode."""
        node = helper.make_node("Pad", inputs=["X", "pads"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["mode"] == "constant"

    def test_get_onnx_attrs_pad_reflect_mode(self):
        """Test Pad with mode=reflect."""
        node = helper.make_node("Pad", inputs=["X", "pads"], outputs=["Y"], mode="reflect")
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["mode"] == "reflect"

    def test_get_onnx_attrs_softmax_defaults(self):
        """Test Softmax with default axis."""
        node = helper.make_node("Softmax", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == -1

    def test_get_onnx_attrs_softmax_custom_axis(self):
        """Test Softmax with custom axis."""
        node = helper.make_node("Softmax", inputs=["X"], outputs=["Y"], axis=1)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == 1

    def test_get_onnx_attrs_split_defaults(self):
        """Test Split with default attributes."""
        node = helper.make_node("Split", inputs=["X"], outputs=["Y1", "Y2"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axis"] == 0
        assert result["num_outputs"] is None

    def test_get_onnx_attrs_split_with_num_outputs(self):
        """Test Split with num_outputs."""
        node = helper.make_node("Split", inputs=["X"], outputs=["Y1", "Y2"], num_outputs=2)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["num_outputs"] == 2

    def test_get_onnx_attrs_unsqueeze_defaults(self):
        """Test Unsqueeze with default attributes."""
        node = helper.make_node("Unsqueeze", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axes"] is None

    def test_get_onnx_attrs_unsqueeze_with_axes(self):
        """Test Unsqueeze with axes."""
        node = helper.make_node("Unsqueeze", inputs=["X"], outputs=["Y"], axes=[0, 2])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["axes"] == (0, 2)

    def test_get_onnx_attrs_upsample_defaults(self):
        """Test Upsample with default mode."""
        node = helper.make_node("Upsample", inputs=["X", "scales"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["mode"] == "nearest"

    def test_get_onnx_attrs_upsample_linear_mode(self):
        """Test Upsample with mode=linear."""
        node = helper.make_node("Upsample", inputs=["X", "scales"], outputs=["Y"], mode="linear")
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["mode"] == "linear"


class TestGetAttrsReduceOperators:
    """Test get_attrs_reduce factory and reduce operators."""

    def test_get_onnx_attrs_reduce_mean_defaults(self):
        """Test ReduceMean with default attributes."""
        node = helper.make_node("ReduceMean", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["keepdims"] == 1
        assert result["noop_with_empty_axes"] == 0

    def test_get_onnx_attrs_reduce_mean_keepdims_0(self):
        """Test ReduceMean with keepdims=0."""
        node = helper.make_node("ReduceMean", inputs=["X"], outputs=["Y"], keepdims=0)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["keepdims"] == 0

    def test_get_onnx_attrs_reduce_mean_noop_error(self):
        """Test ReduceMean with noop_with_empty_axes=1 (unsupported)."""
        node = helper.make_node("ReduceMean", inputs=["X"], outputs=["Y"], noop_with_empty_axes=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="not supported"):
            get_onnx_attrs(node, initializers)

    def test_get_onnx_attrs_reduce_sum_defaults(self):
        """Test ReduceSum with default attributes."""
        node = helper.make_node("ReduceSum", inputs=["X"], outputs=["Y"])
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["keepdims"] == 1
        assert result["noop_with_empty_axes"] == 0

    def test_get_onnx_attrs_reduce_sum_keepdims_0(self):
        """Test ReduceSum with keepdims=0."""
        node = helper.make_node("ReduceSum", inputs=["X"], outputs=["Y"], keepdims=0)
        initializers: dict[str, Any] = {}
        result = get_onnx_attrs(node, initializers)
        assert result["keepdims"] == 0

    def test_get_onnx_attrs_reduce_sum_noop_error(self):
        """Test ReduceSum with noop_with_empty_axes=1 (unsupported)."""
        node = helper.make_node("ReduceSum", inputs=["X"], outputs=["Y"], noop_with_empty_axes=1)
        initializers: dict[str, Any] = {}
        with pytest.raises(ValueError, match="not supported"):
            get_onnx_attrs(node, initializers)
