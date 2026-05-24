"""Package-level constants shared across slimonnx subpackages.

Holds ONNX-spec defaults that are referenced from more than one subpackage.
Subpackage-internal constants stay in their own ``_constants.py``.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "DEFAULT_GEMM_ALPHA",
    "DEFAULT_GEMM_ATTRS",
    "DEFAULT_GEMM_BETA",
    "DEFAULT_GEMM_TRANS_A",
    "DEFAULT_GEMM_TRANS_B",
    "OPSET_RUNTIME",
    "OPSET_SHAPEONNX",
]

#: Opset version for shapeonnx shape inference compatibility.
OPSET_SHAPEONNX = 21

#: Opset version for ONNX Runtime compatibility.
OPSET_RUNTIME = 17

DEFAULT_GEMM_ALPHA = 1.0
DEFAULT_GEMM_BETA = 1.0
DEFAULT_GEMM_TRANS_A = 0
DEFAULT_GEMM_TRANS_B = 0

DEFAULT_GEMM_ATTRS: dict[str, int | float] = {
    "alpha": DEFAULT_GEMM_ALPHA,
    "beta": DEFAULT_GEMM_BETA,
    "transA": DEFAULT_GEMM_TRANS_A,
    "transB": DEFAULT_GEMM_TRANS_B,
}
