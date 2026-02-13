"""SlimONNX: ONNX model optimization and simplification."""

__docformat__ = "restructuredtext"
__version__ = "2026.2.0"

__all__ = [
    "MAX_TESTED_OPSET",
    "RECOMMENDED_OPSET",
    "SLIMONNX_VERSION",
    "AnalysisConfig",
    "OptimizationConfig",
    "SlimONNX",
    "ValidationConfig",
    "__version__",
    "all_optimizations",
    "get_preset",
]

from slimonnx.configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from slimonnx.preprocess import MAX_TESTED_OPSET, RECOMMENDED_OPSET, SLIMONNX_VERSION
from slimonnx.presets import all_optimizations, get_preset
from slimonnx.slimonnx import SlimONNX
