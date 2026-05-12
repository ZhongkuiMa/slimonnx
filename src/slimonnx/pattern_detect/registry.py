"""Pattern detection registry."""

__docformat__ = "restructuredtext"
__all__ = ["PATTERNS", "detect_all_patterns"]

from collections.abc import Callable
from typing import Any

from onnx import NodeProto, TensorProto

from slimonnx.pattern_detect._enums import DetectorSig

PATTERNS = {
    # Fusion patterns
    "matmul_add": {
        "description": "MatMul + Add to Gemm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "conv_bn": {
        "description": "Conv + BatchNorm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "bn_conv": {
        "description": "BatchNorm + Conv fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "conv_transpose_bn": {
        "description": "ConvTranspose + BatchNorm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "bn_conv_transpose": {
        "description": "BatchNorm + ConvTranspose fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "depthwise_conv": {
        "description": "Depthwise convolution detection",
        "category": "fusion",
        "severity": "info",
    },
    "depthwise_conv_bn": {
        "description": "Depthwise Conv + BatchNorm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "bn_depthwise_conv": {
        "description": "BatchNorm + Depthwise Conv fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "gemm_reshape_bn": {
        "description": "Gemm + Reshape + BatchNorm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "bn_reshape_gemm": {
        "description": "BatchNorm + Reshape + Gemm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "bn_gemm": {
        "description": "BatchNorm + Gemm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "transpose_bn_transpose": {
        "description": "Transpose + BatchNorm + Transpose fusion",
        "category": "fusion",
        "severity": "optimization",
    },
    "gemm_gemm": {
        "description": "Consecutive Gemm chain fusion (linear merging)",
        "category": "fusion",
        "severity": "optimization",
    },
    # Redundant operation removal
    "consecutive_reshape": {
        "description": "Reshape to Reshape chain",
        "category": "redundant",
        "severity": "optimization",
    },
    "add_zero": {
        "description": "Add with zero constant",
        "category": "redundant",
        "severity": "redundant",
    },
    "sub_zero": {
        "description": "Sub with zero constant",
        "category": "redundant",
        "severity": "redundant",
    },
    "mul_one": {
        "description": "Mul with one constant",
        "category": "redundant",
        "severity": "redundant",
    },
    "div_one": {
        "description": "Div with one constant",
        "category": "redundant",
        "severity": "redundant",
    },
    "pad_zero": {
        "description": "Pad with zero padding",
        "category": "redundant",
        "severity": "redundant",
    },
    "identity_reshape": {
        "description": "Reshape with same input/output shape",
        "category": "redundant",
        "severity": "redundant",
    },
    # Inference optimizations
    "dropout": {
        "description": "Dropout nodes (training-only, should be removed for inference)",
        "category": "inference",
        "severity": "optimization",
    },
    # Constant folding
    "constant_foldable": {
        "description": "Operations with all-constant inputs (can be pre-computed)",
        "category": "constant_folding",
        "severity": "optimization",
    },
    # Shape optimization
    "reshape_negative_one": {
        "description": "Reshape with -1 shape that can be resolved to concrete values",
        "category": "shape_optimization",
        "severity": "optimization",
    },
}


_DetectorEntry = tuple[str, DetectorSig, Callable[..., list[Any]]]


def _build_detector_registry() -> list[_DetectorEntry]:
    """Build the detector registry with lazy imports."""
    from slimonnx.pattern_detect.constant_ops import detect_constant_foldable
    from slimonnx.pattern_detect.conv_bn import (
        detect_bn_conv,
        detect_bn_conv_transpose,
        detect_conv_bn,
        detect_conv_transpose_bn,
    )
    from slimonnx.pattern_detect.depthwise_conv import (
        detect_bn_depthwise_conv,
        detect_depthwise_conv,
        detect_depthwise_conv_bn,
    )
    from slimonnx.pattern_detect.dropout import detect_dropout
    from slimonnx.pattern_detect.gemm_bn import (
        detect_bn_gemm,
        detect_bn_reshape_gemm,
        detect_gemm_reshape_bn,
    )
    from slimonnx.pattern_detect.gemm_chains import detect_gemm_gemm
    from slimonnx.pattern_detect.matmul_add import detect_matmul_add
    from slimonnx.pattern_detect.redundant_ops import (
        detect_add_zero,
        detect_div_one,
        detect_identity_reshape,
        detect_mul_one,
        detect_pad_zero,
        detect_sub_zero,
    )
    from slimonnx.pattern_detect.reshape_chains import detect_consecutive_reshape
    from slimonnx.pattern_detect.reshape_negative_one import (
        detect_reshape_with_negative_one,
    )
    from slimonnx.pattern_detect.transpose_bn import detect_transpose_bn_transpose

    # (pattern_name, signature_type, detector_function)
    return [
        # Fusion patterns
        ("matmul_add", DetectorSig.NI, detect_matmul_add),
        ("conv_bn", DetectorSig.NIS, detect_conv_bn),
        ("bn_conv", DetectorSig.NIS, detect_bn_conv),
        ("conv_transpose_bn", DetectorSig.NIS, detect_conv_transpose_bn),
        ("bn_conv_transpose", DetectorSig.NIS, detect_bn_conv_transpose),
        ("depthwise_conv", DetectorSig.NIS, detect_depthwise_conv),
        ("depthwise_conv_bn", DetectorSig.NIS, detect_depthwise_conv_bn),
        ("bn_depthwise_conv", DetectorSig.NIS, detect_bn_depthwise_conv),
        ("gemm_reshape_bn", DetectorSig.NIS, detect_gemm_reshape_bn),
        ("bn_reshape_gemm", DetectorSig.NIS, detect_bn_reshape_gemm),
        ("bn_gemm", DetectorSig.NIS, detect_bn_gemm),
        ("transpose_bn_transpose", DetectorSig.NIS, detect_transpose_bn_transpose),
        ("gemm_gemm", DetectorSig.NIS, detect_gemm_gemm),
        # Redundant operations
        ("consecutive_reshape", DetectorSig.N, detect_consecutive_reshape),
        ("add_zero", DetectorSig.NI, detect_add_zero),
        ("sub_zero", DetectorSig.NI, detect_sub_zero),
        ("mul_one", DetectorSig.NI, detect_mul_one),
        ("div_one", DetectorSig.NI, detect_div_one),
        ("pad_zero", DetectorSig.NI, detect_pad_zero),
        ("identity_reshape", DetectorSig.NS, detect_identity_reshape),
        # Inference
        ("dropout", DetectorSig.NIS, detect_dropout),
        # Constant folding
        ("constant_foldable", DetectorSig.NIS, detect_constant_foldable),
        # Shape optimization
        ("reshape_negative_one", DetectorSig.NS_INIT, detect_reshape_with_negative_one),
    ]


def detect_all_patterns(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
) -> dict[str, dict]:
    """Detect all registered patterns.

    :param nodes: Model nodes.
    :param initializers: Model initializers.
    :param data_shapes: Inferred shapes.
    :return: Detection results per pattern.
    """
    detectors = _build_detector_registry()
    results = {}

    for name, sig, detector in detectors:
        # Call detector with appropriate arguments based on signature type
        if sig == DetectorSig.N:
            instances = detector(nodes)
        elif sig == DetectorSig.NI:
            instances = detector(nodes, initializers)
        elif sig == DetectorSig.NIS:
            instances = detector(nodes, initializers, data_shapes)
        elif sig == DetectorSig.NS:
            # Requires data_shapes — skip if not available
            if data_shapes is not None:
                instances = detector(nodes, data_shapes)
            else:
                instances = []
        elif sig == DetectorSig.NS_INIT:
            # Requires both initializers and data_shapes
            if data_shapes is not None:
                instances = detector(nodes, initializers, data_shapes)
            else:
                instances = []
        else:
            instances = []

        results[name] = {
            **PATTERNS[name],
            "count": len(instances),
            "instances": instances,
        }

    return results
