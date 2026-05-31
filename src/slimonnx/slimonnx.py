"""SlimONNX: ONNX model optimization and analysis toolkit."""

__docformat__ = "restructuredtext"
__all__ = ["SlimONNX"]

import logging
import time
import warnings
from pathlib import Path
from typing import Any, cast

import onnx
from onnx import NodeProto, TensorProto

# Module-level imports preserve mock.patch hot-swap of symbols in their owning
# subpackages: tests that ``mock.patch("slimonnx.model_validate.X")`` rely on
# the call site re-resolving ``X`` against ``slimonnx.model_validate`` rather
# than against a name bound at import time inside this module.
from slimonnx import (
    model_validate,
    pattern_detect,
    preprocess,
    structure_analysis,
    utils,
)
from slimonnx.configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from slimonnx.constants import OPSET_RUNTIME
from slimonnx.optimize_onnx._optimize import _optimize_with_config

_logger = logging.getLogger(__name__)

# Exceptions tolerated by shape inference fallback. shapeonnx is optional
# at runtime and may raise any of these on partial coverage / bad graphs.
_SHAPE_INFER_FALLBACK_ERRORS = (
    ImportError,
    ValueError,
    AttributeError,
    KeyError,
    RuntimeError,
)


def _enable_verbose() -> None:
    """Configure package-level logger for console output."""
    pkg_logger = logging.getLogger("slimonnx")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)


def _safe_infer_shapes(
    input_nodes: list,
    output_nodes: list,
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    has_batch_dim: bool,
) -> Any:
    """Run shapeonnx inference, returning None on any tolerated failure.

    Shape inference is best-effort: shapeonnx is optional and tolerates partial
    coverage. Tolerated exception types are listed in
    ``_SHAPE_INFER_FALLBACK_ERRORS``. A warning is emitted on failure so callers
    that swallow ``None`` still surface the cause.

    :param input_nodes: Graph input value infos.

    :param output_nodes: Graph output value infos.

    :param nodes: Graph nodes.

    :param initializers: Initializer map keyed by name.

    :param has_batch_dim: Whether the model carries a leading batch dimension.

    :return: Shape map keyed by tensor name, or ``None`` on failure.
    """
    try:
        from shapeonnx.infer_shape import infer_onnx_shape

        return infer_onnx_shape(
            input_nodes,
            output_nodes,
            nodes,
            initializers,
            has_batch_dim=has_batch_dim,
        )
    except _SHAPE_INFER_FALLBACK_ERRORS as error:
        # Include graph size in the warning so downstream callers (which
        # collapse the failure to ``None``) can correlate symptoms in
        # subsequent passes with the model that triggered them.
        warnings.warn(
            f"Shape inference failed on graph with {len(nodes)} nodes, "
            f"{len(initializers)} initializers ({type(error).__name__}): {error}",
            UserWarning,
            stacklevel=3,
        )
        return None


class SlimONNX:
    """ONNX model optimization and analysis toolkit.

    Provides methods for optimizing, analyzing, and validating ONNX models.

    The following optimizations are always applied (hardcoded as True):
    - constant_to_initializer: Converts Constant nodes to initializers
    - simplify_gemm: Normalizes Gemm attributes (alpha=1, beta=1, trans=False)
    - reorder_by_strict_topological_order: Topological sorting of nodes

    :param verbose: Print stage-by-stage progress.
    """

    def __init__(self, verbose: bool = False):
        """Wire stage-level logging when ``verbose`` is requested."""
        if verbose:
            _enable_verbose()

    def slim(
        self,
        onnx_path: str,
        target_path: str | None = None,
        config: OptimizationConfig | None = None,
        validation: ValidationConfig | None = None,
    ) -> dict:
        """Optimize ONNX model.

        The optimization pipeline:
        1. Load model
        2. Convert to OPSET_RUNTIME for ONNX Runtime compatibility
        3. Apply optimizations based on config
        4. Save optimized model
        5. Optionally validate outputs

        :param onnx_path: Path to input ONNX model.

        :param target_path: Path to save optimized model (default: {input}_simplified.onnx).

        :param config: Optimization configuration (default: OptimizationConfig()).

        :param validation: Validation configuration (default: ValidationConfig()).

        :return: Optimization report
        """
        config = config or OptimizationConfig()
        validation = validation or ValidationConfig()

        _logger.info(f"SlimONNX: optimizing {onnx_path}")
        t_total = time.perf_counter()
        t = time.perf_counter()

        model = self.preprocess(
            onnx_path,
            target_opset=OPSET_RUNTIME,
            infer_shapes=False,
            clear_docstrings=True,
            mark_slimonnx=True,
        )
        _logger.info(f"  Preprocess: loaded, opset conversion ({time.perf_counter() - t:.4f}s)")

        # simplify_gemm + reorder_by_strict_topological_order are pipeline
        # invariants -- always on at the SlimONNX entry point but exposed as
        # kwargs in optimize_onnx() for direct callers and unit tests.
        t = time.perf_counter()
        new_model = _optimize_with_config(
            model,
            config,
            simplify_gemm=True,
            reorder_by_strict_topological_order=True,
        )
        optimization_time = time.perf_counter() - t
        _logger.info(f"  Optimize: passes applied ({optimization_time:.4f}s)")

        t = time.perf_counter()
        if target_path is None:
            target_path = str(Path(onnx_path).with_name(Path(onnx_path).stem + "_simplified.onnx"))
        target_path_obj = Path(target_path)
        target_path_obj.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(new_model, str(target_path_obj))
        export_time = time.perf_counter() - t
        _logger.info(f"  Export: saved to {target_path} ({export_time:.4f}s)")

        validation_result = None
        if validation.validate_outputs:
            t = time.perf_counter()
            validation_result = self.validate_outputs(onnx_path, target_path, validation)
            _logger.info(f"  Validate: outputs compared ({time.perf_counter() - t:.4f}s)")
            if not validation_result["all_match"]:
                raise ValueError(
                    f"Validation failed: "
                    f"{validation_result['failed']}/{validation_result['num_tests']} tests failed, "
                    f"max_diff={validation_result['max_diff']:.2e}"
                )

        _logger.info(f"  Total: {time.perf_counter() - t_total:.4f}s")

        original_node_count = len(model.graph.node)
        optimized_node_count = len(new_model.graph.node)
        reduction = original_node_count - optimized_node_count
        reduction_pct = (reduction / original_node_count * 100) if original_node_count > 0 else 0

        return {
            "original_nodes": original_node_count,
            "optimized_nodes": optimized_node_count,
            "reduction": reduction,
            "reduction_pct": reduction_pct,
            "optimization_time": optimization_time,
            "validation": validation_result,
            "output_path": target_path,
        }

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

        :param onnx_path: Path to ONNX model.

        :param config: Optimization configuration (for has_batch_dim).

        :param analysis: Analysis configuration (for exports).

        :return: Comprehensive analysis report
        """
        config = config or OptimizationConfig()
        analysis = analysis or AnalysisConfig()

        model = self.preprocess(
            onnx_path,
            target_opset=None,
            infer_shapes=True,
            clear_docstrings=False,
            mark_slimonnx=False,
        )

        original_opset = model.opset_import[0].version if model.opset_import else 0
        original_ir = model.ir_version

        input_nodes, output_nodes, nodes, initializers = utils.extract_nodes(
            model, has_batch_dim=config.has_batch_dim
        )

        data_shapes = _safe_infer_shapes(
            input_nodes, output_nodes, nodes, initializers, config.has_batch_dim
        )
        shape_inference_success = data_shapes is not None

        validation = model_validate.validate_model(model, data_shapes=data_shapes)
        patterns = pattern_detect.detect_all_patterns(nodes, initializers, data_shapes)
        structure = structure_analysis.analyze_structure(model, data_shapes)

        total_fusible = sum(p["count"] for p in patterns.values() if p["category"] == "fusion")
        total_redundant = sum(p["count"] for p in patterns.values() if p["category"] == "redundant")

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

        onnx_path_obj = Path(onnx_path)
        if analysis.export_topology:
            topo_path = analysis.topology_path or str(
                onnx_path_obj.with_name(onnx_path_obj.stem + "_topology.json")
            )
            structure_analysis.export_topology_json(nodes, topo_path, data_shapes)

        if analysis.export_json:
            json_path = analysis.json_path or str(
                onnx_path_obj.with_name(onnx_path_obj.stem + "_analysis.json")
            )
            structure_analysis.generate_json_report(report, json_path)

        return report

    def compare(
        self,
        original_path: str,
        optimized_path: str,
    ) -> dict:
        """Compare two ONNX models.

        Analyzes both models and computes differences in structure and patterns.

        :param original_path: Path to original model.

        :param optimized_path: Path to optimized model.

        :return: Comparison report
        """
        original_report = self.analyze(original_path)
        optimized_report = self.analyze(optimized_path)

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

        original_nodes = original_report["structure"]["node_count"]
        optimized_nodes = optimized_report["structure"]["node_count"]
        node_reduction = original_nodes - optimized_nodes
        node_reduction_pct = (node_reduction / original_nodes * 100) if original_nodes > 0 else 0

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

        :param onnx_path: Path to ONNX model.

        :param target_opset: Target opset version (None = keep original, default = 21).

        :param infer_shapes: Run ONNX shape inference (default: True).

        :param clear_docstrings: Clear node docstrings (default: True).

        :param mark_slimonnx: Mark model as processed by SlimONNX (default: True).

        :return: Preprocessed model
        """
        return preprocess.load_and_preprocess(
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

        :param onnx_path: Path to ONNX model.

        :param config: Optimization configuration (for has_batch_dim).

        :return: Validation report
        """
        config = config or OptimizationConfig()

        model = self.preprocess(
            onnx_path,
            target_opset=OPSET_RUNTIME,
            infer_shapes=True,
            clear_docstrings=False,
            mark_slimonnx=False,
        )

        input_nodes, output_nodes, nodes, initializers = utils.extract_nodes(
            model, has_batch_dim=config.has_batch_dim
        )

        data_shapes = _safe_infer_shapes(
            input_nodes, output_nodes, nodes, initializers, config.has_batch_dim
        )

        return model_validate.validate_model(model, data_shapes=data_shapes)

    def detect_patterns(
        self,
        onnx_path: str,
        config: OptimizationConfig | None = None,
    ) -> dict:
        """Detect optimization patterns in ONNX model.

        Detects:
        - Fusion patterns (MatMul+Add, Conv+BN, etc.)
        - Redundant operations (Add zero, Mul one, identity Reshape, etc.)

        :param onnx_path: Path to ONNX model.

        :param config: Optimization configuration (for has_batch_dim).

        :return: Pattern detection report
        """
        config = config or OptimizationConfig()

        model = self.preprocess(
            onnx_path,
            target_opset=OPSET_RUNTIME,
            infer_shapes=True,
            clear_docstrings=False,
            mark_slimonnx=False,
        )

        input_nodes, output_nodes, nodes, initializers = utils.extract_nodes(
            model, has_batch_dim=config.has_batch_dim
        )

        data_shapes = _safe_infer_shapes(
            input_nodes, output_nodes, nodes, initializers, config.has_batch_dim
        )

        return cast(
            dict[Any, Any],
            pattern_detect.detect_all_patterns(nodes, initializers, data_shapes),
        )

    def validate_outputs(
        self,
        original_path: str,
        optimized_path: str,
        validation: ValidationConfig | None = None,
    ) -> dict:
        """Compare outputs of two ONNX models numerically.

        :param original_path: Path to original model.

        :param optimized_path: Path to optimized model.

        :param validation: Validation configuration.

        :return: Validation report
        """
        validation = validation or ValidationConfig()

        return model_validate.compare_model_outputs(
            original_path,
            optimized_path,
            input_bounds=validation.input_bounds,
            test_data_path=validation.test_data_path,
            num_samples=validation.num_samples,
            rtol=validation.rtol,
            atol=validation.atol,
        )
