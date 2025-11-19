"""SlimONNX: ONNX model optimization and simplification."""

__docformat__ = "restructuredtext"
__all__ = [
    "SlimONNX",
    "OptimizationConfig",
    "ValidationConfig",
    "AnalysisConfig",
    "get_preset",
    "all_optimizations",
    "RECOMMENDED_OPSET",
    "MAX_TESTED_OPSET",
    "SLIMONNX_VERSION",
]

from .configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from .preprocess import MAX_TESTED_OPSET, RECOMMENDED_OPSET, SLIMONNX_VERSION
from .presets import all_optimizations, get_preset
from .slimonnx import SlimONNX
