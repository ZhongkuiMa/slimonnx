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
        *,
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
        has_batch_dim: bool = True,
        validate_outputs: bool = False,
        validation_input_bounds: tuple[list[float], list[float]] | None = None,
        validation_test_data_path: str | None = None,
        num_validation_samples: int = 5,
        validation_rtol: float = 1e-5,
        validation_atol: float = 1e-6,
        return_report: bool = False,
    ) -> dict | None:
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
        :param validate_outputs: Whether to validate optimized outputs match original.
        :param validation_input_bounds: Input bounds (lower, upper) for validation.
        :param validation_test_data_path: Path to test data file (.pth, .npy, .npz).
        :param num_validation_samples: Number of validation samples to generate.
        :param validation_rtol: Relative tolerance for output validation.
        :param validation_atol: Absolute tolerance for output validation.
        :param return_report: Whether to return optimization report dict.

        :return: Report dict if return_report=True, else None.
        """
        t = time.perf_counter()
        if self.verbose:
            print(f"Load ONNX model from {onnx_path}...")
        model = onnx.load(onnx_path)

        # Convert to ONNX version 21 for compatibility with shapeonnx
        model = onnx.version_converter.convert_version(model, target_version=21)

        if self.verbose:
            print(f"Slim ONNX model...")
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
            has_batch_dim=has_batch_dim,
            verbose=self.verbose,
        )
        t = time.perf_counter() - t
        if self.verbose:
            print(f"Complete slimming ONNX model in {t:.4f}s")

        if target_path is None:
            target_path = onnx_path.replace(".onnx", "_simplified.onnx")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Downgrade model version if validation is requested for ONNX Runtime compatibility
        if validate_outputs:
            current_opset = (
                new_model.opset_import[0].version if new_model.opset_import else 0
            )
            # Downgrade to Opset 17 (widely supported) if current > 17
            if current_opset > 17:
                if self.verbose:
                    print(
                        f"Downgrading model from Opset {current_opset}/IR {new_model.ir_version} "
                        f"to Opset 17/IR 8 for ONNX Runtime compatibility..."
                    )
                import onnx.version_converter as version_converter

                new_model = version_converter.convert_version(new_model, 17)
                new_model.ir_version = 8

        onnx.save(new_model, target_path)

        if self.verbose:
            print(f"Slimmed ONNX model saved to {target_path}")

        # Optional: Validate outputs
        validation_result = None
        if validate_outputs:
            if self.verbose:
                print("Validating optimized outputs...")

            from .model_validate import compare_model_outputs

            validation_result = compare_model_outputs(
                onnx_path,
                target_path,
                input_bounds=validation_input_bounds,
                test_data_path=validation_test_data_path,
                num_samples=num_validation_samples,
                rtol=validation_rtol,
                atol=validation_atol,
            )

            if validation_result["all_match"]:
                if self.verbose:
                    print(f"Validation PASSED ({validation_result['num_tests']} tests)")
            else:
                print(
                    f"WARNING: Validation FAILED! "
                    f"{validation_result['failed']}/{validation_result['num_tests']} tests failed, "
                    f"max_diff={validation_result['max_diff']:.2e}"
                )

        # Optional: Return report
        if return_report or validate_outputs:
            original_node_count = len(model.graph.node)
            optimized_node_count = len(new_model.graph.node)
            reduction = original_node_count - optimized_node_count
            reduction_pct = (
                (reduction / original_node_count * 100)
                if original_node_count > 0
                else 0
            )

            return {
                "original_nodes": original_node_count,
                "optimized_nodes": optimized_node_count,
                "reduction": reduction,
                "reduction_pct": reduction_pct,
                "optimization_time": t,
                "validation": validation_result,
                "output_path": target_path,
            }

        return None
