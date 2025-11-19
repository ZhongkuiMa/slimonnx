"""Pattern detection for ONNX model optimization."""

__docformat__ = "restructuredtext"
__all__ = [
    "PATTERNS",
    "detect_all_patterns",
    "detect_matmul_add",
    "detect_add_zero",
    "detect_mul_one",
    "detect_pad_zero",
    "detect_identity_reshape",
    "detect_consecutive_reshape",
]

from .registry import PATTERNS, detect_all_patterns
from .matmul_add import detect_matmul_add
from .redundant_ops import (
    detect_add_zero,
    detect_mul_one,
    detect_pad_zero,
    detect_identity_reshape,
)
from .reshape_chains import detect_consecutive_reshape
