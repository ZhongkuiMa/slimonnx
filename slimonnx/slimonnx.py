__docformat__ = "restructuredtext"
__all__ = ["SlimONNX"]

import os.path
import time

import onnx

from .optimize_onnx import optimize_onnx


class SlimONNX:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def slim(
        self,
        onnx_path: str,
        target_path: str | None = None,
        constant_to_initializer: bool = True,
        fuse_constant_nodes: bool = False,
        fuse_matmul_add: bool = False,
        fuse_gemm_reshape_bn: bool = False,
        fuse_bn_reshape_gemm: bool = False,
        fuse_bn_gemm: bool = False,
        fuse_transpose_bn_transpose: bool = False,
        fuse_gemm_gemm: bool = False,
        fuse_conv_bn: bool = False,
        fuse_bn_conv: bool = False,
        fuse_convtransposed_bn: bool = False,
        fuse_bn_convtransposed: bool = False,
        simplify_conv_to_flatten_gemm: bool = False,
        simplify_gemm: bool = True,
        remove_redundant_operations: bool = False,
        simplify_node_name: bool = True,
        reorder_by_strict_topological_order: bool = True,
        validate_model: bool = True,
    ):
        """
        Simplify the ONNX model by fusing some nodes.

        By default, all the node docstring will be removed from the ONNX model.

        :param onnx_path: The path to the ONNX model.
        :param target_path: The path to save the simplified ONNX model.
        :param constant_to_initializer: Convert the constant nodes to initializers.
        :param fuse_matmul_add: Fuse a MatMul and an Add node into a single Gemm node.
        :param fuse_gemm_reshape_bn: Fuse a Gemm, a Reshape, and a BatchNormalization
            node into a Gemm and a Reshape node.
        :param fuse_bn_reshape_gemm: Fuse a BatchNormalization, a Reshape, and a Gemm
            node into a Reshape and a Gemm node.
        :param fuse_bn_gemm: Fuse a BatchNormalization and a Gemm node into a Gemm node.
        :param fuse_transpose_bn_transpose: Fuse a Transpose, a BatchNormalization,
            and a Transpose node into a Gemm node.
        :param fuse_gemm_gemm: Fuse two Gemm nodes into a single Gemm node.
        :param fuse_conv_bn: Fuse a Conv and BatchNormalization node into a Conv node.
        :param fuse_bn_conv: Fuse a BatchNormalization and a Conv node into a Conv node.
        :param fuse_convtransposed_bn: Fuse a ConvTranspose and a BatchNormalization
            node into a ConvTranspose node.
        :param fuse_bn_convtransposed: Fuse a BatchNormalization and a ConvTranspose
            node into a ConvTranspose node.
        :param simplify_conv_to_flatten_gemm: Simplify the Conv node to a Flatten and
            a Gemm node if possible.
        :param simplify_gemm: Simplify the Gemm node by setting the alpha and beta to
            1.0 and transA and transB to False.
        :param remove_redundant_operations: Remove redundant nodes, such as redundant
            Reshape, Add, Sub, Mul, Div, Pad nodes.
        :param fuse_constant_nodes: Convert the shape nodes to initializers, or fuse
        fixed constant operations.
        :param simplify_node_name: Simplify the node name by topological order.
        :param reorder_by_strict_topological_order: Reorder the nodes by topological
            order and simplify their names.

        :return: The simplified ONNX model.
        """
        if self.verbose:
            print(f"Slim ONNX model {onnx_path}...")
            t = time.perf_counter()
        model = onnx.load(onnx_path)

        new_model = optimize_onnx(
            model,
            constant_to_initializer=constant_to_initializer,
            fuse_constant_nodes=fuse_constant_nodes,
            fuse_matmul_add=fuse_matmul_add,
            fuse_gemm_reshape_bn=fuse_gemm_reshape_bn,
            fuse_bn_reshape_gemm=fuse_bn_reshape_gemm,
            fuse_bn_gemm=fuse_bn_gemm,
            fuse_transpose_bn_transpose=fuse_transpose_bn_transpose,
            fuse_gemm_gemm=fuse_gemm_gemm,
            fuse_conv_bn=fuse_conv_bn,
            fuse_bn_conv=fuse_bn_conv,
            fuse_convtransposed_bn=fuse_convtransposed_bn,
            fuse_bn_convtransposed=fuse_bn_convtransposed,
            simplify_conv_to_flatten_gemm=simplify_conv_to_flatten_gemm,
            simplify_gemm=simplify_gemm,
            remove_redundant_operations=remove_redundant_operations,
            reorder_by_strict_topological_order=reorder_by_strict_topological_order,
            simplify_node_name=simplify_node_name,
            verbose=self.verbose,
        )

        if target_path is None:
            target_path = onnx_path.replace(".onnx", "_simplified.onnx")

        # Check if the directory exists
        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

        onnx.save(new_model, target_path)

        if self.verbose:
            t = time.perf_counter() - t
            print(f"Slimmed ONNX model saved to {target_path} ({t:.4f}s)")
