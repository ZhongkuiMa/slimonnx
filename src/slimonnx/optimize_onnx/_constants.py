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

from types import MappingProxyType

import numpy as np

# Read-only view to prevent accidental mutation of this lookup table by
# any caller in the subpackage. ``MappingProxyType`` raises ``TypeError``
# on assignment / deletion. Note: bfloat16 (ONNX dtype 16) is approximated
# as float16 -- numpy has no native bfloat16, and downstream fusion math
# runs at fp32 anyway.
ONNX_DTYPE_TO_NUMPY: MappingProxyType[int, type] = MappingProxyType(
    {
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
        16: np.float16,
    }
)

AUTO_PAD_NOTSET = "NOTSET"

CONV_2D_KERNEL_DIMS = 2
CONV_2D_WEIGHT_DIMS = 4
GEMM_REQUIRED_RANK = 2

TRANSPOSE_CHW_TO_CWH = (0, 2, 1)
