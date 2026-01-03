"""Extended tests for MatMul+Add fusion (_mm_add.py)."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from slimonnx.optimize_onnx._mm_add import (
    _can_fuse_to_gemm_matmul_add,
    _extract_matmul_add_params,
    _fuse_matmul_add,
)


def create_initializer(name, array):
    """Create a TensorProto initializer from numpy array."""
    return numpy_helper.from_array(array.astype(np.float32), name)


class TestExtractMatmulAddParams:
    """Test _extract_matmul_add_params function."""

    def test_extract_params_standard_layout(self):
        """Test extracting params when weight is second input of MatMul."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        input_name, weight_name, bias_name, trans_b = _extract_matmul_add_params(
            matmul, add, initializers
        )

        assert input_name == "X"
        assert weight_name == "W"
        assert bias_name == "B"
        assert trans_b == 0

    def test_extract_params_transposed_weight(self):
        """Test extracting params when weight is first input of MatMul (transposed)."""
        matmul = helper.make_node("MatMul", inputs=["W", "X"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        input_name, weight_name, bias_name, trans_b = _extract_matmul_add_params(
            matmul, add, initializers
        )

        assert input_name == "X"
        assert weight_name == "W"
        assert bias_name == "B"
        assert trans_b == 1

    def test_extract_params_bias_second_input(self):
        """Test when bias is second input of Add."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        _input_name, _weight_name, bias_name, _trans_b = _extract_matmul_add_params(
            matmul, add, initializers
        )

        assert bias_name == "B"

    def test_extract_params_bias_first_input(self):
        """Test when bias is first input of Add."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["B", "Y"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        _input_name, _weight_name, bias_name, _trans_b = _extract_matmul_add_params(
            matmul, add, initializers
        )

        assert bias_name == "B"


class TestCanFuseToGemmMatmulAdd:
    """Test _can_fuse_to_gemm_matmul_add function."""

    def test_can_fuse_valid_2d_weight_1d_bias(self):
        """Test fusion with valid 2D weight and 1D bias."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, None)

        assert result is True

    def test_cannot_fuse_non_2d_weight(self):
        """Test fusion fails with non-2D weight."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3, 3))),  # 3D weight
            "B": create_initializer("B", np.ones(3)),
        }

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, None)

        assert result is False

    def test_cannot_fuse_1d_weight(self):
        """Test fusion fails with 1D weight."""
        initializers = {
            "W": create_initializer("W", np.ones(3)),  # 1D weight
            "B": create_initializer("B", np.ones(3)),
        }

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, None)

        assert result is False

    def test_cannot_fuse_non_1d_bias(self):
        """Test fusion fails with non-1D bias."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones((3, 1))),  # 2D bias
        }

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, None)

        assert result is False

    def test_cannot_fuse_3d_bias(self):
        """Test fusion fails with 3D bias."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones((3, 1, 1))),  # 3D bias
        }

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, None)

        assert result is False

    def test_can_fuse_with_data_shapes_rank_2(self):
        """Test fusion with data_shapes having rank 2 input."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }
        data_shapes = {"X": [1, 3]}

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, data_shapes)

        assert result is True

    def test_cannot_fuse_with_data_shapes_rank_3(self):
        """Test fusion fails with data_shapes having rank 3 input."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }
        data_shapes = {"X": [1, 2, 3]}

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, data_shapes)

        assert result is False

    def test_can_fuse_with_input_nodes_rank_2(self):
        """Test fusion with input_nodes having rank 2 input."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        input_nodes = [input_info]

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, input_nodes, None)

        assert result is True

    def test_cannot_fuse_with_input_nodes_rank_3(self):
        """Test fusion fails with input_nodes having rank 3 input."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3])
        input_nodes = [input_info]

        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, input_nodes, None)

        assert result is False

    def test_can_fuse_with_input_nodes_input_not_found(self):
        """Test fusion defaults to True when input not found in input_nodes."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        input_info = helper.make_tensor_value_info("Other", TensorProto.FLOAT, [1, 2, 3])
        input_nodes = [input_info]

        # Should return True when input not found (optimization assumption)
        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, input_nodes, None)

        assert result is True

    def test_can_fuse_with_data_shapes_input_not_found(self):
        """Test fusion with data_shapes when input not found."""
        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }
        data_shapes = {"Other": [1, 3]}

        # Should return True when input not in data_shapes
        result = _can_fuse_to_gemm_matmul_add("X", "W", "B", initializers, None, data_shapes)

        assert result is True


class TestFuseMatmulAdd:
    """Test _fuse_matmul_add function."""

    def test_fuse_matmul_add_basic(self):
        """Test basic MatMul+Add fusion to Gemm."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")
        relu = helper.make_node("Relu", inputs=["Z"], outputs=["Out"], name="relu_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add, relu]
        result = _fuse_matmul_add(nodes, initializers)

        # Should have 2 nodes (Gemm + Relu)
        assert len(result) == 2
        assert result[0].op_type == "Gemm"
        assert result[1].op_type == "Relu"

    def test_fuse_matmul_add_preserves_inputs(self):
        """Test that fused Gemm has correct inputs."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers)

        assert len(result) == 1
        assert result[0].op_type == "Gemm"
        # Inputs should be X, W, B
        assert list(result[0].input) == ["X", "W", "B"]

    def test_no_fuse_non_adjacent_matmul_add(self):
        """Test that non-adjacent MatMul and Add are not fused."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Y2"], name="relu_0")
        add = helper.make_node("Add", inputs=["Y2", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        nodes = [matmul, relu, add]
        result = _fuse_matmul_add(nodes, initializers)

        # Should not fuse - MatMul and Add not adjacent
        assert len(result) == 3

    def test_no_fuse_add_without_constant(self):
        """Test that Add without constant inputs is not fused."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "Z"], outputs=["Out"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
        }

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers)

        # Should not fuse - Add input Z is not in initializers
        assert len(result) == 2

    def test_no_fuse_matmul_without_constant(self):
        """Test that MatMul without constant inputs is not fused."""
        matmul = helper.make_node("MatMul", inputs=["X", "Y"], outputs=["Z"], name="mm_0")
        add = helper.make_node("Add", inputs=["Z", "B"], outputs=["Out"], name="add_0")

        initializers = {
            "B": create_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers)

        # Should not fuse - MatMul has no initializer input
        assert len(result) == 2

    def test_fuse_with_data_shapes(self):
        """Test fusion with data_shapes parameter."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }
        data_shapes = {"X": [1, 3]}

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers, data_shapes=data_shapes)

        # Should fuse
        assert len(result) == 1
        assert result[0].op_type == "Gemm"

    def test_no_fuse_invalid_input_shape(self):
        """Test no fusion when input shape is not rank 2."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }
        data_shapes = {"X": [1, 2, 3]}

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers, data_shapes=data_shapes)

        # Should not fuse - input rank is 3
        assert len(result) == 2

    def test_fuse_with_input_nodes(self):
        """Test fusion with input_nodes parameter."""
        matmul = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        input_nodes = [input_info]

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers, input_nodes=input_nodes)

        # Should fuse
        assert len(result) == 1
        assert result[0].op_type == "Gemm"

    def test_fuse_transposed_weight(self):
        """Test fusion when weight is transposed."""
        matmul = helper.make_node("MatMul", inputs=["W", "X"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "B"], outputs=["Z"], name="add_0")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "B": create_initializer("B", np.ones(3)),
        }

        nodes = [matmul, add]
        result = _fuse_matmul_add(nodes, initializers)

        assert len(result) == 1
        assert result[0].op_type == "Gemm"
        # With trans_b=1, inputs should be W, X, B
        assert list(result[0].input) == ["W", "X", "B"]

    def test_preserves_other_nodes(self):
        """Test that other nodes are preserved."""
        relu1 = helper.make_node("Relu", inputs=["A"], outputs=["B"], name="relu_0")
        matmul = helper.make_node("MatMul", inputs=["B", "W"], outputs=["Y"], name="mm_0")
        add = helper.make_node("Add", inputs=["Y", "Bias"], outputs=["Z"], name="add_0")
        relu2 = helper.make_node("Relu", inputs=["Z"], outputs=["Out"], name="relu_1")

        initializers = {
            "W": create_initializer("W", np.ones((3, 3))),
            "Bias": create_initializer("Bias", np.ones(3)),
        }

        nodes = [relu1, matmul, add, relu2]
        result = _fuse_matmul_add(nodes, initializers)

        # Should have Relu, Gemm, Relu
        assert len(result) == 3
        assert result[0].op_type == "Relu"
        assert result[1].op_type == "Gemm"
        assert result[2].op_type == "Relu"

    def test_multiple_matmul_add_pairs(self):
        """Test multiple MatMul+Add pairs in sequence."""
        matmul1 = helper.make_node("MatMul", inputs=["X", "W1"], outputs=["Y1"], name="mm_0")
        add1 = helper.make_node("Add", inputs=["Y1", "B1"], outputs=["Z1"], name="add_0")
        matmul2 = helper.make_node("MatMul", inputs=["Z1", "W2"], outputs=["Y2"], name="mm_1")
        add2 = helper.make_node("Add", inputs=["Y2", "B2"], outputs=["Z2"], name="add_1")

        initializers = {
            "W1": create_initializer("W1", np.ones((3, 3))),
            "B1": create_initializer("B1", np.ones(3)),
            "W2": create_initializer("W2", np.ones((3, 3))),
            "B2": create_initializer("B2", np.ones(3)),
        }

        nodes = [matmul1, add1, matmul2, add2]
        result = _fuse_matmul_add(nodes, initializers)

        # Should have 2 Gemm nodes
        assert len(result) == 2
        assert all(node.op_type == "Gemm" for node in result)
