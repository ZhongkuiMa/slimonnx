"""Pattern detection registry."""

__docformat__ = "restructuredtext"
__all__ = ["PATTERNS", "detect_all_patterns"]

from onnx import NodeProto, TensorProto

PATTERNS = {
    "matmul_add": {
        "description": "MatMul + Add to Gemm fusion",
        "category": "fusion",
        "severity": "optimization",
    },
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
    "mul_one": {
        "description": "Mul with one constant",
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
    from .matmul_add import detect_matmul_add
    from .redundant_ops import (
        detect_add_zero,
        detect_mul_one,
        detect_pad_zero,
        detect_identity_reshape,
    )
    from .reshape_chains import detect_consecutive_reshape

    results = {}

    # Detect MatMul+Add
    matmul_add_instances = detect_matmul_add(nodes, initializers)
    results["matmul_add"] = {
        **PATTERNS["matmul_add"],
        "count": len(matmul_add_instances),
        "instances": matmul_add_instances,
    }

    # Detect consecutive reshape
    reshape_instances = detect_consecutive_reshape(nodes)
    results["consecutive_reshape"] = {
        **PATTERNS["consecutive_reshape"],
        "count": len(reshape_instances),
        "instances": reshape_instances,
    }

    # Detect redundant operations
    add_zero_instances = detect_add_zero(nodes, initializers)
    results["add_zero"] = {
        **PATTERNS["add_zero"],
        "count": len(add_zero_instances),
        "instances": add_zero_instances,
    }

    mul_one_instances = detect_mul_one(nodes, initializers)
    results["mul_one"] = {
        **PATTERNS["mul_one"],
        "count": len(mul_one_instances),
        "instances": mul_one_instances,
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

    return results
