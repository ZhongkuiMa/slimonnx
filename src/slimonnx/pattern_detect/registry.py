"""Pattern detection registry."""

__docformat__ = "restructuredtext"
__all__ = ["PATTERNS", "detect_all_patterns"]

from onnx import NodeProto, TensorProto

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


def detect_all_patterns(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> dict[str, dict]:
    """Detect all registered patterns.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :param data_shapes: Inferred shapes
    :return: Detection results per pattern
    """
    # Import all detectors
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

    results = {}

    # Detect fusion patterns
    matmul_add_instances = detect_matmul_add(nodes, initializers)
    results["matmul_add"] = {
        **PATTERNS["matmul_add"],
        "count": len(matmul_add_instances),
        "instances": matmul_add_instances,
    }

    conv_bn_instances = detect_conv_bn(nodes, initializers, data_shapes)
    results["conv_bn"] = {
        **PATTERNS["conv_bn"],
        "count": len(conv_bn_instances),
        "instances": conv_bn_instances,
    }

    bn_conv_instances = detect_bn_conv(nodes, initializers, data_shapes)
    results["bn_conv"] = {
        **PATTERNS["bn_conv"],
        "count": len(bn_conv_instances),
        "instances": bn_conv_instances,
    }

    conv_transpose_bn_instances = detect_conv_transpose_bn(nodes, initializers, data_shapes)
    results["conv_transpose_bn"] = {
        **PATTERNS["conv_transpose_bn"],
        "count": len(conv_transpose_bn_instances),
        "instances": conv_transpose_bn_instances,
    }

    bn_conv_transpose_instances = detect_bn_conv_transpose(nodes, initializers, data_shapes)
    results["bn_conv_transpose"] = {
        **PATTERNS["bn_conv_transpose"],
        "count": len(bn_conv_transpose_instances),
        "instances": bn_conv_transpose_instances,
    }

    depthwise_conv_instances = detect_depthwise_conv(nodes, initializers, data_shapes)
    results["depthwise_conv"] = {
        **PATTERNS["depthwise_conv"],
        "count": len(depthwise_conv_instances),
        "instances": depthwise_conv_instances,
    }

    depthwise_conv_bn_instances = detect_depthwise_conv_bn(nodes, initializers, data_shapes)
    results["depthwise_conv_bn"] = {
        **PATTERNS["depthwise_conv_bn"],
        "count": len(depthwise_conv_bn_instances),
        "instances": depthwise_conv_bn_instances,
    }

    bn_depthwise_conv_instances = detect_bn_depthwise_conv(nodes, initializers, data_shapes)
    results["bn_depthwise_conv"] = {
        **PATTERNS["bn_depthwise_conv"],
        "count": len(bn_depthwise_conv_instances),
        "instances": bn_depthwise_conv_instances,
    }

    gemm_reshape_bn_instances = detect_gemm_reshape_bn(nodes, initializers, data_shapes)
    results["gemm_reshape_bn"] = {
        **PATTERNS["gemm_reshape_bn"],
        "count": len(gemm_reshape_bn_instances),
        "instances": gemm_reshape_bn_instances,
    }

    bn_reshape_gemm_instances = detect_bn_reshape_gemm(nodes, initializers, data_shapes)
    results["bn_reshape_gemm"] = {
        **PATTERNS["bn_reshape_gemm"],
        "count": len(bn_reshape_gemm_instances),
        "instances": bn_reshape_gemm_instances,
    }

    bn_gemm_instances = detect_bn_gemm(nodes, initializers, data_shapes)
    results["bn_gemm"] = {
        **PATTERNS["bn_gemm"],
        "count": len(bn_gemm_instances),
        "instances": bn_gemm_instances,
    }

    transpose_bn_transpose_instances = detect_transpose_bn_transpose(
        nodes, initializers, data_shapes
    )
    results["transpose_bn_transpose"] = {
        **PATTERNS["transpose_bn_transpose"],
        "count": len(transpose_bn_transpose_instances),
        "instances": transpose_bn_transpose_instances,
    }

    gemm_gemm_instances = detect_gemm_gemm(nodes, initializers, data_shapes)
    results["gemm_gemm"] = {
        **PATTERNS["gemm_gemm"],
        "count": len(gemm_gemm_instances),
        "instances": gemm_gemm_instances,
    }

    # Detect redundant operations
    consecutive_reshape_instances = detect_consecutive_reshape(nodes)
    results["consecutive_reshape"] = {
        **PATTERNS["consecutive_reshape"],
        "count": len(consecutive_reshape_instances),
        "instances": consecutive_reshape_instances,
    }

    add_zero_instances = detect_add_zero(nodes, initializers)
    results["add_zero"] = {
        **PATTERNS["add_zero"],
        "count": len(add_zero_instances),
        "instances": add_zero_instances,
    }

    sub_zero_instances = detect_sub_zero(nodes, initializers)
    results["sub_zero"] = {
        **PATTERNS["sub_zero"],
        "count": len(sub_zero_instances),
        "instances": sub_zero_instances,
    }

    mul_one_instances = detect_mul_one(nodes, initializers)
    results["mul_one"] = {
        **PATTERNS["mul_one"],
        "count": len(mul_one_instances),
        "instances": mul_one_instances,
    }

    div_one_instances = detect_div_one(nodes, initializers)
    results["div_one"] = {
        **PATTERNS["div_one"],
        "count": len(div_one_instances),
        "instances": div_one_instances,
    }

    pad_zero_instances = detect_pad_zero(nodes, initializers)
    results["pad_zero"] = {
        **PATTERNS["pad_zero"],
        "count": len(pad_zero_instances),
        "instances": pad_zero_instances,
    }

    # Detect identity reshape (needs shapes)
    if data_shapes is not None:
        identity_reshape_instances = detect_identity_reshape(nodes, data_shapes)
        results["identity_reshape"] = {
            **PATTERNS["identity_reshape"],
            "count": len(identity_reshape_instances),
            "instances": identity_reshape_instances,
        }
    else:
        results["identity_reshape"] = {
            **PATTERNS["identity_reshape"],
            "count": 0,
            "instances": [],
        }

    # Detect inference optimizations
    dropout_instances = detect_dropout(nodes, initializers, data_shapes)
    results["dropout"] = {
        **PATTERNS["dropout"],
        "count": len(dropout_instances),
        "instances": dropout_instances,
    }

    # Detect constant folding opportunities
    constant_foldable_instances = detect_constant_foldable(nodes, initializers, data_shapes)
    results["constant_foldable"] = {
        **PATTERNS["constant_foldable"],
        "count": len(constant_foldable_instances),
        "instances": constant_foldable_instances,
    }

    # Detect reshape with resolvable -1
    if data_shapes is not None:
        reshape_negative_one_instances = detect_reshape_with_negative_one(
            nodes, initializers, data_shapes
        )
        results["reshape_negative_one"] = {
            **PATTERNS["reshape_negative_one"],
            "count": len(reshape_negative_one_instances),
            "instances": reshape_negative_one_instances,
        }
    else:
        results["reshape_negative_one"] = {
            **PATTERNS["reshape_negative_one"],
            "count": 0,
            "instances": [],
        }

    return results
