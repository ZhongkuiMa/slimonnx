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

from slimonnx.constant_ops import detect_constant_foldable
from slimonnx.conv_bn import (
    detect_bn_conv,
    detect_bn_convtranspose,
    detect_conv_bn,
    detect_convtranspose_bn,
)
from slimonnx.depthwise_conv import (
    detect_bn_depthwise_conv,
    detect_depthwise_conv,
    detect_depthwise_conv_bn,
)
from slimonnx.dropout import detect_dropout
from slimonnx.gemm_bn import (
    detect_bn_gemm,
    detect_bn_reshape_gemm,
    detect_gemm_reshape_bn,
)
from slimonnx.gemm_chains import detect_gemm_gemm
from slimonnx.matmul_add import detect_matmul_add
from slimonnx.redundant_ops import (
    detect_add_zero,
    detect_div_one,
    detect_identity_reshape,
    detect_mul_one,
    detect_pad_zero,
    detect_sub_zero,
)
from slimonnx.registry import PATTERNS, detect_all_patterns
from slimonnx.reshape_chains import detect_consecutive_reshape
from slimonnx.reshape_negative_one import detect_reshape_with_negative_one
from slimonnx.transpose_bn import detect_transpose_bn_transpose
