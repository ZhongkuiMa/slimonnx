"""Pattern detection for ONNX model optimization."""

__docformat__ = "restructuredtext"
__all__ = [
    "PATTERNS",
    "detect_add_zero",
    "detect_all_patterns",
    "detect_bn_conv",
    "detect_bn_conv_transpose",
    "detect_bn_depthwise_conv",
    "detect_bn_gemm",
    "detect_bn_reshape_gemm",
    "detect_consecutive_reshape",
    "detect_constant_foldable",
    "detect_conv_bn",
    "detect_conv_transpose_bn",
    "detect_depthwise_conv",
    "detect_depthwise_conv_bn",
    "detect_div_one",
    "detect_dropout",
    "detect_gemm_gemm",
    "detect_gemm_reshape_bn",
    "detect_identity_reshape",
    "detect_matmul_add",
    "detect_mul_one",
    "detect_pad_zero",
    "detect_reshape_with_negative_one",
    "detect_sub_zero",
    "detect_transpose_bn_transpose",
]

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
from slimonnx.pattern_detect.registry import PATTERNS, detect_all_patterns
from slimonnx.pattern_detect.reshape_chains import detect_consecutive_reshape
from slimonnx.pattern_detect.reshape_negative_one import detect_reshape_with_negative_one
from slimonnx.pattern_detect.transpose_bn import detect_transpose_bn_transpose
