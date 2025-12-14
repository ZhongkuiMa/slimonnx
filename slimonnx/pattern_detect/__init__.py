"""Pattern detection for ONNX model optimization."""

__docformat__ = "restructuredtext"
__all__ = [
    "PATTERNS",
    "detect_all_patterns",
    "detect_matmul_add",
    "detect_conv_bn",
    "detect_bn_conv",
    "detect_convtranspose_bn",
    "detect_bn_convtranspose",
    "detect_depthwise_conv",
    "detect_depthwise_conv_bn",
    "detect_bn_depthwise_conv",
    "detect_gemm_reshape_bn",
    "detect_bn_reshape_gemm",
    "detect_bn_gemm",
    "detect_transpose_bn_transpose",
    "detect_gemm_gemm",
    "detect_add_zero",
    "detect_sub_zero",
    "detect_mul_one",
    "detect_div_one",
    "detect_pad_zero",
    "detect_identity_reshape",
    "detect_consecutive_reshape",
    "detect_dropout",
    "detect_constant_foldable",
    "detect_reshape_with_negative_one",
]

from .registry import PATTERNS, detect_all_patterns
from .matmul_add import detect_matmul_add
from .conv_bn import (
    detect_conv_bn,
    detect_bn_conv,
    detect_convtranspose_bn,
    detect_bn_convtranspose,
)
from .depthwise_conv import (
    detect_depthwise_conv,
    detect_depthwise_conv_bn,
    detect_bn_depthwise_conv,
)
from .gemm_bn import (
    detect_gemm_reshape_bn,
    detect_bn_reshape_gemm,
    detect_bn_gemm,
)
from .transpose_bn import detect_transpose_bn_transpose
from .gemm_chains import detect_gemm_gemm
from .redundant_ops import (
    detect_add_zero,
    detect_sub_zero,
    detect_mul_one,
    detect_div_one,
    detect_pad_zero,
    detect_identity_reshape,
)
from .reshape_chains import detect_consecutive_reshape
from .dropout import detect_dropout
from .constant_ops import detect_constant_foldable
from .reshape_negative_one import detect_reshape_with_negative_one
