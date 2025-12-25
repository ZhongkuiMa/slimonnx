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

from slimonnx.configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from slimonnx.preprocess import MAX_TESTED_OPSET, RECOMMENDED_OPSET, SLIMONNX_VERSION
from slimonnx.presets import all_optimizations, get_preset
from slimonnx.slimonnx.slimonnx import SlimONNX
