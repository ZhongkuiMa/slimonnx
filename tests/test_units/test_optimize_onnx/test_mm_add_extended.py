"""Extended tests for MatMul+Add fusion (_mm_add.py)."""

__docformat__ = "restructuredtext"

import numpy as np
import pytest
from onnx import TensorProto, helper

from slimonnx.optimize_onnx._mm_add import (
    _can_fuse_to_gemm_matmul_add,
    _extract_matmul_add_params,
    _fuse_matmul_add,
)


class TestExtractMatmulAddParams:
    """Test _extract_matmul_add_params function."""

    @pytest.mark.parametrize(
        ("matmul_inputs", "add_inputs", "expected_trans_b"),
        [
            (["X", "W"], ["Y", "B"], 0),
            (["W", "X"], ["Y", "B"], 1),
        ],
        ids=["weight_second", "weight_first_transposed"],
    )
    def test_extracts_params_with_weight_position(
        self, matmul_inputs, add_inputs, expected_trans_b, make_initializer
    ):
        """Test extracting params handles weight in first or second position."""
        matmul = helper.make_node("MatMul", inputs=matmul_inputs, outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=add_inputs, outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        input_name, weight_name, bias_name, trans_b = _extract_matmul_add_params(
            matmul, add, initializers
        )

        assert input_name == "X"
        assert weight_name == "W"
        assert bias_name == "B"
        assert trans_b == expected_trans_b

    @pytest.mark.parametrize(
        "add_inputs",
        [
            ["Y", "B"],
            ["B", "Y"],
        ],
        ids=["bias_second", "bias_first"],
    )
    def test_extracts_params_with_bias_position(self, add_inputs, make_initializer):
        """Test extracting params handles bias in first or second position."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=add_inputs, outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        _input_name, _weight_name, bias_name, _trans_b = _extract_matmul_add_params(
            matmul, add, initializers
        )

        assert bias_name == "B"


class TestCanFuseToGemmMatmulAdd:
    """Test _can_fuse_to_gemm_matmul_add function."""

    @pytest.mark.parametrize(
        ("w_shape", "b_shape", "expected"),
        [
            pytest.param((3, 3), (3,), True, id="valid_2d_weight_1d_bias"),
            pytest.param((3, 3, 3), (3,), False, id="non_2d_weight_3d"),
            pytest.param((3,), (3,), False, id="non_2d_weight_1d"),
            pytest.param((3, 3), (3, 1), False, id="non_1d_bias_2d"),
            pytest.param((3, 3), (3, 1, 1), False, id="non_1d_bias_3d"),
        ],
    )
    def test_validates_shape_eligibility(self, w_shape, b_shape, expected, make_initializer):
        """Test fusion eligibility based on weight and bias tensor shapes."""
        initializers = {
            "W": make_initializer("W", np.ones(w_shape)),
            "B": make_initializer("B", np.ones(b_shape)),
        }
        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, None)
        assert result is expected

    @pytest.mark.parametrize(
        ("shape", "via_data_shapes", "expected"),
        [
            ([1, 3], True, True),
            ([1, 2, 3], True, False),
            ([1, 3], False, True),
            ([1, 2, 3], False, False),
        ],
        ids=[
            "rank2_via_data_shapes",
            "rank3_via_data_shapes",
            "rank2_via_input_nodes",
            "rank3_via_input_nodes",
        ],
    )
    def test_checks_input_rank_eligibility(
        self, shape, via_data_shapes, expected, make_initializer
    ):
        """Test fusion eligibility based on input tensor rank."""
        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        if via_data_shapes:
            data_shapes = {"X": shape}
            result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, data_shapes)
        else:
            input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
            input_nodes = [input_info]
            result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, input_nodes, None)

        assert result is expected

    def test_with_input_nodes_input_not_found(self, make_initializer):
        """Test fusion defaults to True when input not found in input_nodes."""
        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        input_info = helper.make_tensor_value_info("Other", TensorProto.FLOAT, [1, 2, 3])
        input_nodes = [input_info]

        # Should return True when input not found (optimization assumption)
        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, input_nodes, None)

        assert result is True

    def test_with_data_shapes_input_not_found(self, make_initializer):
        """Test fusion with data_shapes when input not found."""
        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }
        data_shapes = {"Other": [1, 3]}

        # Should return True when input not in data_shapes
        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, data_shapes)

        assert result is True


class TestFuseMatmulAdd:
    """Test _fuse_matmul_add function."""

    @pytest.mark.parametrize(
        ("extra_nodes", "expected_len", "expected_ops"),
        [
            ([], 1, ["Gemm"]),
            ([("Relu", ["Z"], ["Out"], "relu_0")], 2, ["Gemm", "Relu"]),
        ],
        ids=["matmul_add_only", "with_trailing_relu"],
    )
    def test_fuses_adjacent_matmul_add(
        self, extra_nodes, expected_len, expected_ops, make_initializer
    ):
        """Test MatMul+Add fusion to Gemm with various successor patterns."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]
        if extra_nodes:
            for op, inputs, outputs, name in extra_nodes:
                nodes.append(helper.make_node(op, inputs=inputs, outputs=outputs, name=name))

        result = _fuse_matmul_add(nodes, initializers)

        assert len(result) == expected_len
        assert [n.op_type for n in result] == expected_ops

        # When fused, Gemm should have X, W, B inputs
        if result and result[0].op_type == "Gemm":
            assert list(result[0].input) == ["X", "W", "B"]

    @pytest.mark.parametrize(
        ("matmul_inputs", "add_inputs", "has_all_initializers"),
        [
            (["X", "W"], ["Y", "B"], True),
            (["X", "W"], ["Y", "Z"], False),
            (["X", "Y"], ["Z", "B"], False),
        ],
        ids=["all_constant_inputs", "add_missing_constant", "matmul_missing_constant"],
    )
    def test_requires_constant_initializers(
        self, matmul_inputs, add_inputs, has_all_initializers, make_initializer
    ):
        """Test fusion fails when inputs lack constant initializers."""
        matmul = helper.make_node("MatMul", inputs=matmul_inputs, outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=add_inputs, outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers)

        if has_all_initializers:
            assert len(result) == 1
            assert result[0].op_type == "Gemm"
        else:
            assert len(result) == 2

    @pytest.mark.parametrize(
        ("shape", "should_fuse"),
        [
            ([1, 3], True),
            ([1, 2, 3], False),
        ],
        ids=["rank2_fusible", "rank3_not_fusible"],
    )
    def test_respects_input_shape_constraints(self, shape, should_fuse, make_initializer):
        """Test fusion eligibility with data_shapes parameter."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }
        data_shapes = {"X": shape}

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers, data_shapes=data_shapes)

        if should_fuse:
            assert len(result) == 1
            assert result[0].op_type == "Gemm"
        else:
            assert len(result) == 2

    @pytest.mark.parametrize(
        "via_data_shapes",
        [
            False,
            True,
        ],
        ids=["via_input_nodes", "via_data_shapes"],
    )
    def test_fuses_with_explicit_shape_info(self, via_data_shapes, make_initializer):
        """Test fusion with explicit shape information via input_nodes or data_shapes."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]

        if via_data_shapes:
            result = _fuse_matmul_add(nodes, initializers, data_shapes={"X": [1, 3]})
        else:
            input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
            result = _fuse_matmul_add(nodes, initializers, input_nodes=[input_info])

        assert len(result) == 1
        assert result[0].op_type == "Gemm"

    def test_fuses_with_weight_transposed(self, make_initializer):
        """Test fusion when weight is first MatMul input (transposed)."""
        matmul = helper.make_node("MatMul", inputs=["W", "X"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "B": make_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers)

        assert len(result) == 1
        assert result[0].op_type == "Gemm"
        # With trans_b=1, inputs should be W, X, B
        assert list(result[0].input) == ["W", "X", "B"]

    def test_preserves_other_nodes(self, make_initializer):
        """Test that other nodes are preserved."""
        relu1 = helper.make_node("Relu", inputs=["A"], outputs=["B"], name="relu_0")
        matmul = helper.make_node("MatMul", inputs=["B", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "Bias"], outputs=["Z"], name="add_0")
        relu2 = helper.make_node("Relu", inputs=["Z"], outputs=["Out"], name="relu_1")

        initializers = {
            "W": make_initializer("W", np.ones((3, 3))),
            "Bias": make_initializer("Bias", np.ones(3)),
        }

        nodes = [relu1, matmul, add, relu2]
        result = _fuse_matmul_add(nodes, initializers)

        # Should have Relu, Gemm, Relu
        assert len(result) == 3
        assert result[0].op_type == "Relu"
        assert result[1].op_type == "Gemm"
        assert result[2].op_type == "Relu"

    def test_fuses_multiple_adjacent_pairs(self, make_initializer):
        """Test multiple MatMul+Add pairs fused separately when adjacent."""
        matmul1 = helper.make_node("MatMul", inputs=["X", "W1"], outputs=["Y1"], name="mm_0")
        add1 = helper.make_node("Add", inputs=["Y1", "B1"], outputs=["Z1"], name="add_0")
        matmul2 = helper.make_node("MatMul", inputs=["Z1", "W2"], outputs=["Y2"], name="mm_1")
        add2 = helper.make_node("Add", inputs=["Y2", "B2"], outputs=["Z2"], name="add_1")

        initializers = {
            "W1": make_initializer("W1", np.ones((3, 3))),
            "B1": make_initializer("B1", np.ones(3)),
            "W2": make_initializer("W2", np.ones((3, 3))),
            "B2": make_initializer("B2", np.ones(3)),
        }

        nodes = [matmul1, add1, matmul2, add2]
        result = _fuse_matmul_add(nodes, initializers)

        # Should have 2 Gemm nodes
        assert len(result) == 2
        assert all(node.op_type == "Gemm" for node in result)
