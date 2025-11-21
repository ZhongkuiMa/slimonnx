"""Preset optimization configurations for common benchmarks."""

__docformat__ = "restructuredtext"
__all__ = ["get_preset", "all_optimizations", "PRESET_NAMES"]

from functools import lru_cache

from .configs import OptimizationConfig

# Tuple of available preset names
PRESET_NAMES = (
    "acasxu_2023",
    "cctsdb_yolo_2023",
    "cersyve",
    "cgan_2023",
    "cifar100_2024",
    "collins_aerospace_benchmark",
    "collins_rul_cnn_2022",
    "cora_2024",
    "dist_shift_2023",
    "linearizenn",
    "lsnc",
    "lsnc_relu",
    "malbeware",
    "metaroom_2023",
    "ml4acopf_2024",
    "nn4sys",
    "nn4sys_2023",
    "relusplitter",
    "safenlp_2024",
    "sat_relu",
    "soundnessbench",
    "tinyimagenet_2024",
    "tllverifybench_2023",
    "traffic_signs_recognition_2023",
    "vggnet16_2022",
    "vit_2023",
    "yolo_2023",
    "test",
)


@lru_cache(maxsize=128)
def get_preset(
    benchmark_name: str, model_name: str | None = None
) -> OptimizationConfig:
    """Get preset optimization configuration for a benchmark.

    :param benchmark_name: Benchmark identifier (e.g., 'acasxu_2023', 'cgan_2023')
    :param model_name: Optional model filename for per-model exceptions
    :return: Optimization configuration optimized for the benchmark
    """
    # Model-specific exceptions for nn4sys_2023 (some have batch dim, some don't)
    if benchmark_name == "nn4sys_2023" and model_name:
        if "pensieve" in model_name and "parallel" in model_name:
            return OptimizationConfig(
                fuse_matmul_add=True,
                has_batch_dim=False,
            )

    presets = {
        "acasxu_2023": OptimizationConfig(
            fuse_matmul_add=True,
            remove_redundant_operations=True,
        ),
        "cctsdb_yolo_2023": OptimizationConfig(
            constant_folding=True,
            has_batch_dim=False,
        ),
        "cifar100_2024": OptimizationConfig(
            fuse_conv_bn=True,
            fuse_bn_conv=True,
        ),
        "cgan_2023": OptimizationConfig(
            fuse_conv_bn=True,
            fuse_bn_conv=True,
            fuse_convtransposed_bn=True,
            constant_folding=True,
            remove_redundant_operations=True,
            has_batch_dim=False,
        ),
        "collins_aerospace_benchmark": OptimizationConfig(),
        "collins_rul_cnn_2022": OptimizationConfig(
            simplify_conv_to_flatten_gemm=True,
            remove_redundant_operations=True,
        ),
        "cora_2024": OptimizationConfig(
            fuse_matmul_add=True,
        ),
        "dist_shift_2023": OptimizationConfig(
            remove_redundant_operations=True,
        ),
        "linearizenn": OptimizationConfig(),
        "lsnc": OptimizationConfig(
            constant_folding=True,
        ),
        "metaroom_2023": OptimizationConfig(),
        "ml4acopf_2024": OptimizationConfig(
            constant_folding=True,
        ),
        "nn4sys": OptimizationConfig(
            fuse_matmul_add=True,
        ),
        "nn4sys_2023": OptimizationConfig(
            fuse_matmul_add=True,
        ),
        "safenlp_2024": OptimizationConfig(
            fuse_matmul_add=True,
        ),
        "tinyimagenet_2024": OptimizationConfig(
            fuse_conv_bn=True,
        ),
        "tllverifybench_2023": OptimizationConfig(
            fuse_matmul_add=True,
        ),
        "traffic_signs_recognition_2023": OptimizationConfig(),
        "vggnet16_2022": OptimizationConfig(),
        "vit_2023": OptimizationConfig(
            constant_folding=True,
            fuse_matmul_add=True,
            fuse_transpose_bn_transpose=True,
            fuse_gemm_gemm=True,
            fuse_bn_gemm=True,
            remove_redundant_operations=True,
        ),
        "yolo_2023": OptimizationConfig(),
        "test": all_optimizations(has_batch_dim=False),
        "cersyve": OptimizationConfig(
            fuse_gemm_gemm=True,
        ),
        "lsnc_relu": OptimizationConfig(
            constant_folding=True,
        ),
        "malbeware": OptimizationConfig(),
        "relusplitter": OptimizationConfig(),
        "sat_relu": OptimizationConfig(),
        "soundnessbench": OptimizationConfig(),
    }

    # Return preset or default config
    return presets.get(benchmark_name, OptimizationConfig())


def all_optimizations(has_batch_dim: bool = True) -> OptimizationConfig:
    """Get configuration with all optimizations enabled.

    :param has_batch_dim: Whether model has batch dimension
    :return: Optimization configuration with all flags True
    """
    return OptimizationConfig(
        fuse_matmul_add=True,
        fuse_conv_bn=True,
        fuse_bn_conv=True,
        fuse_convtransposed_bn=True,
        fuse_bn_convtransposed=True,
        fuse_gemm_reshape_bn=True,
        fuse_bn_reshape_gemm=True,
        fuse_bn_gemm=True,
        fuse_transpose_bn_transpose=True,
        fuse_gemm_gemm=True,
        simplify_conv_to_flatten_gemm=True,
        remove_redundant_operations=True,
        constant_folding=True,
        simplify_node_name=True,
        has_batch_dim=has_batch_dim,
    )
