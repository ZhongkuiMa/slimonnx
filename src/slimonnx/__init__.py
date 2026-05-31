"""SlimONNX: ONNX model optimization and simplification."""

__docformat__ = "restructuredtext"

__all__ = [
    "MAX_TESTED_OPSET",
    "OPSET_RUNTIME",
    "OPSET_SHAPEONNX",
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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from slimonnx.configs import AnalysisConfig, OptimizationConfig, ValidationConfig
from slimonnx.constants import OPSET_RUNTIME, OPSET_SHAPEONNX
from slimonnx.preprocess import MAX_TESTED_OPSET, RECOMMENDED_OPSET, SLIMONNX_VERSION
from slimonnx.presets import all_optimizations, get_preset
from slimonnx.slimonnx import SlimONNX

# Single source of truth: package metadata from pyproject.toml. Fall back to
# the producer-tag version embedded by version_converter when the package is
# imported from a source tree without an installed distribution (e.g. editable
# install via sys.path manipulation without setuptools metadata).
try:
    __version__ = _pkg_version("slimonnx")
except PackageNotFoundError:  # pragma: no cover - source-tree fallback
    __version__ = SLIMONNX_VERSION
