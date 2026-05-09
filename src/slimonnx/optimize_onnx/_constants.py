"""Constants and type mappings internal to optimize_onnx.

Constants shared with other slimonnx subpackages live in
``slimonnx.constants``.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "AUTO_PAD_NOTSET",
    "CONV_2D_KERNEL_DIMS",
    "CONV_2D_WEIGHT_DIMS",
    "GEMM_REQUIRED_RANK",
    "ONNX_DTYPE_TO_NUMPY",
    "TRANSPOSE_CHW_TO_CWH",
]

import numpy as np

ONNX_DTYPE_TO_NUMPY: dict[int, type] = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    8: np.str_,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
    14: np.complex64,
    15: np.complex128,
    16: np.float16,  # bfloat16 -> float16 as approximation
}

AUTO_PAD_NOTSET = "NOTSET"

CONV_2D_KERNEL_DIMS = 2
CONV_2D_WEIGHT_DIMS = 4
GEMM_REQUIRED_RANK = 2

TRANSPOSE_CHW_TO_CWH = (0, 2, 1)
