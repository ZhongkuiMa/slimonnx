"""SlimONNX: ONNX model optimization and analysis toolkit."""

__docformat__ = "restructuredtext"
__all__ = ["SlimONNX"]

import os
import time

import onnx

from .configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from .optimize_onnx import optimize_onnx


class SlimONNX:
    """ONNX model optimization and analysis toolkit.

    Provides methods for optimizing, analyzing, and validating ONNX models.
    All methods are stateless except for the verbose flag.

    The following optimizations are always applied (hardcoded as True):
    - constant_to_initializer: Converts Constant nodes to initializers
    - simplify_gemm: Normalizes Gemm attributes (alpha=1, beta=1, trans=False)
    - reorder_by_strict_topological_order: Topological sorting of nodes
    """

    def __init__(self, verbose: bool = False):
        """Initialize SlimONNX.

        :param verbose: Print detailed progress messages
        """
        self.verbose = verbose

    def slim(
        self,
        onnx_path: str,
        target_path: str | None = None,
        config: OptimizationConfig | None = None,
        validation: ValidationConfig | None = None,
    ) -> dict | None:
        """Optimize ONNX model.

        The optimization pipeline:
        1. Load model
        2. Convert to Opset 21 for compatibility with shapeonnx
        3. Apply optimizations based on config
        4. Optionally downgrade to Opset 17 for ONNX Runtime compatibility
        5. Save optimized model
        6. Optionally validate outputs

        :param onnx_path: Path to input ONNX model
        :param target_path: Path to save optimized model (default: {input}_simplified.onnx)
        :param config: Optimization configuration (default: OptimizationConfig())
        :param validation: Validation configuration (default: ValidationConfig())
        :return: Optimization report if validation.validate_outputs=True, else None
        """
        config = config or OptimizationConfig()
        validation = validation or ValidationConfig()

        t = time.perf_counter()
        if self.verbose:
            print(f"Load and preprocess ONNX model from {onnx_path}")

        # Preprocess model (load, convert to opset 21, clear docs, mark SlimONNX)
        model = self.preprocess(
            onnx_path,
            target_opset=21,
            infer_shapes=False,
            clear_docstrings=True,
            mark_slimonnx=True,
        )

        if self.verbose:
            print("Optimize ONNX model")

        # Apply optimizations (constant_to_initializer, simplify_gemm, reorder always True)
        new_model = optimize_onnx(
            model,
            constant_to_initializer=True,  # Always True
            constant_folding=config.constant_folding,
            fuse_matmul_add=config.fuse_matmul_add,
            fuse_gemm_reshape_bn=config.fuse_gemm_reshape_bn,
            fuse_bn_reshape_gemm=config.fuse_bn_reshape_gemm,
            fuse_bn_gemm=config.fuse_bn_gemm,
            fuse_transpose_bn_transpose=config.fuse_transpose_bn_transpose,
            fuse_gemm_gemm=config.fuse_gemm_gemm,
            fuse_conv_bn=config.fuse_conv_bn,
            fuse_bn_conv=config.fuse_bn_conv,
            fuse_convtransposed_bn=config.fuse_convtransposed_bn,
            fuse_bn_convtransposed=config.fuse_bn_convtransposed,
            simplify_conv_to_flatten_gemm=config.simplify_conv_to_flatten_gemm,
            simplify_gemm=True,  # Always True
            remove_redundant_operations=config.remove_redundant_operations,
            reorder_by_strict_topological_order=True,  # Always True
            simplify_node_name=config.simplify_node_name,
            has_batch_dim=config.has_batch_dim,
            verbose=self.verbose,
        )

        t = time.perf_counter() - t
        if self.verbose:
            print(f"Optimization completed in {t:.4f}s")

        # Determine save path
        if target_path is None:
            target_path = onnx_path.replace(".onnx", "_simplified.onnx")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Downgrade for ONNX Runtime compatibility if validation requested
        if validation.validate_outputs:
            current_opset = (
                new_model.opset_import[0].version if new_model.opset_import else 0
            )
            if current_opset > 17:
                if self.verbose:
                    print(
                        f"Downgrade model from Opset {current_opset}/IR {new_model.ir_version} "
                        f"to Opset 17/IR 8 for ONNX Runtime compatibility"
                    )
                new_model = onnx.version_converter.convert_version(new_model, 17)
                new_model.ir_version = 8

        onnx.save(new_model, target_path)

        if self.verbose:
            print(f"Optimized model saved to {target_path}")

        # Validate outputs if requested
        validation_result = None
        if validation.validate_outputs:
            validation_result = self.validate_outputs(
                onnx_path, target_path, validation
            )

            if validation_result["all_match"]:
                if self.verbose:
                    print(f"Validation PASSED ({validation_result['num_tests']} tests)")
            else:
                print(
                    f"WARNING: Validation FAILED "
                    f"({validation_result['failed']}/{validation_result['num_tests']} tests failed, "
                    f"max_diff={validation_result['max_diff']:.2e})"
                )

        # Return report if validation was performed
        if validation.validate_outputs:
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

    def analyze(
        self,
        onnx_path: str,
        config: OptimizationConfig | None = None,
        analysis: AnalysisConfig | None = None,
    ) -> dict:
        """Analyze ONNX model structure, patterns, and validity.

        Performs comprehensive analysis by calling:
        - preprocess(): Load and shape inference
        - validate(): ONNX checker, runtime validation, graph checks
        - detect_patterns(): Optimization pattern detection
        - Structure analysis: Node counts, I/O info, op types

        :param onnx_path: Path to ONNX model
        :param config: Optimization configuration (for has_batch_dim)
        :param analysis: Analysis configuration (for exports)
        :return: Comprehensive analysis report with redesigned structure
        """
        config = config or OptimizationConfig()
        analysis = analysis or AnalysisConfig()

        if self.verbose:
            print(f"Analyze model: {onnx_path}")

        # Preprocess model (keep original version for accurate analysis)
        model = self.preprocess(
            onnx_path,
            target_opset=None,
            infer_shapes=True,
            clear_docstrings=False,
            mark_slimonnx=False,
        )

        original_opset = model.opset_import[0].version if model.opset_import else 0
        original_ir = model.ir_version

        # Extract nodes (converts Constant nodes to initializers)
        from . import utils

        input_nodes, output_nodes, nodes, initializers = utils.extract_nodes(
            model, has_batch_dim=config.has_batch_dim, verbose=self.verbose
        )

        # Infer shapes using shapeonnx
        try:
            from shapeonnx.shapeonnx.infer_shape import infer_onnx_shape

            data_shapes = infer_onnx_shape(
                input_nodes,
                output_nodes,
                nodes,
                initializers,
                has_batch_dim=config.has_batch_dim,
            )
            shape_inference_success = True
        except Exception as e:
            if self.verbose:
                print(f"Warning: Shape inference failed: {e}")
            data_shapes = None
            shape_inference_success = False

        # Validate model
        if self.verbose:
            print("Validate model")
        from .model_validate import validate_model

        validation = validate_model(model, data_shapes=data_shapes)

        # Detect patterns
        if self.verbose:
            print("Detect patterns")
        from .pattern_detect import detect_all_patterns

        patterns = detect_all_patterns(nodes, initializers, data_shapes)

        # Analyze structure
        if self.verbose:
            print("Analyze structure")
        from .structure_analysis import analyze_structure

        structure = analyze_structure(model, data_shapes)

        # Calculate optimization recommendations
        total_fusible = sum(
            p["count"] for p in patterns.values() if p["category"] == "fusion"
        )
        total_redundant = sum(
            p["count"] for p in patterns.values() if p["category"] == "redundant"
        )

        # Build redesigned report structure
        report = {
            "model": {
                "path": onnx_path,
                "opset": original_opset,
                "ir_version": original_ir,
                "shape_inference": "success" if shape_inference_success else "failed",
            },
            "structure": structure,
            "validation": validation,
            "patterns": patterns,
            "recommendations": {
                "fusible_patterns": total_fusible,
                "redundant_patterns": total_redundant,
                "estimated_node_reduction": total_fusible + total_redundant,
            },
        }

        # Export topology if requested
        if analysis.export_topology:
            from .structure_analysis import export_topology_json

            topo_path = analysis.topology_path or onnx_path.replace(
                ".onnx", "_topology.json"
            )
            export_topology_json(nodes, topo_path, data_shapes)
            if self.verbose:
                print(f"Topology exported to {topo_path}")

        # Export full analysis if requested
        if analysis.export_json:
            from .structure_analysis import generate_json_report

            json_path = analysis.json_path or onnx_path.replace(
                ".onnx", "_analysis.json"
            )
            generate_json_report(report, json_path)
            if self.verbose:
                print(f"Analysis exported to {json_path}")

        return report

    def compare(
        self,
        original_path: str,
        optimized_path: str,
    ) -> dict:
        """Compare two ONNX models.

        Analyzes both models and computes differences in structure and patterns.

        :param original_path: Path to original model
        :param optimized_path: Path to optimized model
        :return: Comparison report with redesigned structure
        """
        if self.verbose:
            print(
                f"Compare models:\n  Original: {original_path}\n  Optimized: {optimized_path}"
            )

        # Analyze both models (suppress verbose output)
        original_verbose = self.verbose
        self.verbose = False
        original_report = self.analyze(original_path)
        optimized_report = self.analyze(optimized_path)
        self.verbose = original_verbose

        # Calculate pattern differences
        patterns_fixed = {}
        for pattern_name in original_report["patterns"]:
            original_count = original_report["patterns"][pattern_name]["count"]
            optimized_count = optimized_report["patterns"][pattern_name]["count"]
            fixed = original_count - optimized_count
            if fixed != 0:
                patterns_fixed[pattern_name] = {
                    "before": original_count,
                    "after": optimized_count,
                    "fixed": fixed,
                }

        # Calculate node reduction
        original_nodes = original_report["structure"]["node_count"]
        optimized_nodes = optimized_report["structure"]["node_count"]
        node_reduction = original_nodes - optimized_nodes
        node_reduction_pct = (
            (node_reduction / original_nodes * 100) if original_nodes > 0 else 0
        )

        # Build redesigned comparison structure
        return {
            "original": original_report,
            "optimized": optimized_report,
            "diff": {
                "nodes": {
                    "before": original_nodes,
                    "after": optimized_nodes,
                    "reduction": node_reduction,
                    "reduction_pct": node_reduction_pct,
                },
                "patterns_fixed": patterns_fixed,
            },
        }

    def preprocess(
        self,
        onnx_path: str,
        target_opset: int | None = None,
        infer_shapes: bool = True,
        clear_docstrings: bool = True,
        mark_slimonnx: bool = True,
    ) -> onnx.ModelProto:
        """Load and preprocess ONNX model.

        Preprocessing steps:
        1. Load model from file
        2. Validate with ONNX checker
        3. Convert to target opset version (default: 21 for shapeonnx compatibility)
        4. Run shape inference (if enabled)
        5. Clear node docstrings (if enabled)
        6. Mark as processed by SlimONNX (if enabled)

        Recommended opset: 17-21 (tested with shapeonnx)
        Default: 21 (for shapeonnx compatibility)

        :param onnx_path: Path to ONNX model
        :param target_opset: Target opset version (None = keep original, default = 21)
        :param infer_shapes: Run ONNX shape inference (default: True)
        :param clear_docstrings: Clear node docstrings (default: True)
        :param mark_slimonnx: Mark model as processed by SlimONNX (default: True)
        :return: Preprocessed model
        """
        from .preprocess import load_and_preprocess

        return load_and_preprocess(
            onnx_path,
            target_opset=target_opset,
            infer_shapes=infer_shapes,
            check_model=True,
            clear_docstrings=clear_docstrings,
            mark_slimonnx=mark_slimonnx,
        )

    def validate(
        self,
        onnx_path: str,
        config: OptimizationConfig | None = None,
    ) -> dict:
        """Validate ONNX model correctness.

        Runs all validation checks:
        - ONNX checker
        - ONNX Runtime compatibility
        - Dead nodes detection
        - Broken connections detection
        - Orphan initializers detection
        - Type consistency
        - Shape consistency (if shapes available)

        :param onnx_path: Path to ONNX model
        :param config: Optimization configuration (for has_batch_dim)
        :return: Validation report
        """
        config = config or OptimizationConfig()

        model = self.preprocess(
            onnx_path,
            target_opset=21,
            infer_shapes=True,
            clear_docstrings=False,
            mark_slimonnx=False,
        )

        from . import utils

        input_nodes, output_nodes, nodes, initializers = utils.extract_nodes(
            model, has_batch_dim=config.has_batch_dim, verbose=self.verbose
        )

        # Infer shapes
        try:
            from shapeonnx.shapeonnx.infer_shape import infer_onnx_shape

            data_shapes = infer_onnx_shape(
                input_nodes,
                output_nodes,
                nodes,
                initializers,
                has_batch_dim=config.has_batch_dim,
            )
        except Exception:
            data_shapes = None

        from .model_validate import validate_model

        return validate_model(model, data_shapes=data_shapes)

    def detect_patterns(
        self,
        onnx_path: str,
        config: OptimizationConfig | None = None,
    ) -> dict:
        """Detect optimization patterns in ONNX model.

        Detects:
        - Fusion patterns (MatMul+Add, Conv+BN, etc.)
        - Redundant operations (Add zero, Mul one, identity Reshape, etc.)

        :param onnx_path: Path to ONNX model
        :param config: Optimization configuration (for has_batch_dim)
        :return: Pattern detection report
        """
        config = config or OptimizationConfig()

        model = self.preprocess(
            onnx_path,
            target_opset=21,
            infer_shapes=True,
            clear_docstrings=False,
            mark_slimonnx=False,
        )

        from . import utils

        input_nodes, output_nodes, nodes, initializers = utils.extract_nodes(
            model, has_batch_dim=config.has_batch_dim, verbose=self.verbose
        )

        # Infer shapes
        try:
            from shapeonnx.shapeonnx.infer_shape import infer_onnx_shape

            data_shapes = infer_onnx_shape(
                input_nodes,
                output_nodes,
                nodes,
                initializers,
                has_batch_dim=config.has_batch_dim,
            )
        except Exception:
            data_shapes = None

        from .pattern_detect import detect_all_patterns

        return detect_all_patterns(nodes, initializers, data_shapes)

    def validate_outputs(
        self,
        original_path: str,
        optimized_path: str,
        validation: ValidationConfig | None = None,
    ) -> dict:
        """Compare outputs of two ONNX models numerically.

        :param original_path: Path to original model
        :param optimized_path: Path to optimized model
        :param validation: Validation configuration
        :return: Validation report
        """
        validation = validation or ValidationConfig()

        from .model_validate import compare_model_outputs

        return compare_model_outputs(
            original_path,
            optimized_path,
            input_bounds=validation.input_bounds,
            test_data_path=validation.test_data_path,
            num_samples=validation.num_samples,
            rtol=validation.rtol,
            atol=validation.atol,
        )
