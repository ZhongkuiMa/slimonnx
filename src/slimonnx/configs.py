"""Configuration dataclasses for SlimONNX."""

__docformat__ = "restructuredtext"
__all__ = ["AnalysisConfig", "OptimizationConfig", "ValidationConfig"]

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    """Immutable optimization configuration.

    Defines which optimizations to apply during ONNX model slimming.
    All flags default to False except simplify_node_name and has_batch_dim.

    Note: The following optimizations are always applied (hardcoded):
    - constant_to_initializer (converts Constant nodes to initializers)
    - simplify_gemm (normalizes Gemm attributes)
    - reorder_by_strict_topological_order (topological sorting)
    """

    # Fusion optimizations
    fuse_matmul_add: bool = False
    fuse_conv_bn: bool = False
    fuse_bn_conv: bool = False
    fuse_bn_conv_with_padding: bool = False
    fuse_conv_transpose_bn: bool = False
    fuse_bn_conv_transpose: bool = False
    fuse_depthwise_conv_bn: bool = False
    fuse_bn_depthwise_conv: bool = False
    fuse_gemm_reshape_bn: bool = False
    fuse_bn_reshape_gemm: bool = False
    fuse_bn_gemm: bool = False
    fuse_transpose_bn_transpose: bool = False
    fuse_gemm_gemm: bool = False

    # Simplification optimizations
    simplify_conv_to_flatten_gemm: bool = False
    remove_redundant_operations: bool = False

    # Inference optimizations
    remove_dropout: bool = True

    # Preprocessing
    constant_folding: bool = False

    # Postprocessing
    simplify_node_name: bool = False

    # Model properties
    has_batch_dim: bool = True


@dataclass(frozen=True)
class ValidationConfig:
    """Immutable validation configuration.

    Controls numerical output validation between original and optimized models.
    """

    validate_outputs: bool = False
    input_bounds: tuple[list[float], list[float]] | None = None
    test_data_path: str | None = None
    num_samples: int = 5
    rtol: float = 1e-5
    atol: float = 1e-6


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable analysis configuration.

    Controls model analysis and export options.
    """

    export_json: bool = False
    json_path: str | None = None
    export_topology: bool = False
    topology_path: str | None = None
