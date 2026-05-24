"""Preset optimization configurations for common benchmarks."""

__docformat__ = "restructuredtext"
__all__ = ["PRESET_NAMES", "all_optimizations", "get_preset"]

from functools import lru_cache

from slimonnx.configs import OptimizationConfig

#: Alias map for vnncomp2024 benchmark name changes and shorthand names.
_ALIASES: dict[str, str] = {
    "cifar100": "cifar100_2024",
    "collins_rul_cnn_2023": "collins_rul_cnn_2022",
    "cora": "cora_2024",
    "nn4sys": "nn4sys_2023",
    "safenlp": "safenlp_2024",
    "tinyimagenet": "tinyimagenet_2024",
    "vggnet16_2023": "vggnet16_2022",
}

#: Default config for benchmarks that only need constant folding.
_CONSTANT_FOLDING_PRESET = OptimizationConfig(constant_folding=True)

#: Benchmarks that only require constant_folding=True (no fusion flags).
_CF_ONLY_PRESETS = frozenset(
    {
        "collins_aerospace_benchmark",
        "linearizenn",
        "lsnc",
        "lsnc_relu",
        "malbeware",
        "metaroom_2023",
        "ml4acopf_2023",
        "ml4acopf_2024",
        "relusplitter",
        "sat_relu",
        "soundnessbench",
        "traffic_signs_recognition_2023",
        "vggnet16_2022",
        "yolo_2023",
    }
)

# Tuple of available preset names
PRESET_NAMES = (
    "acasxu_2023",
    "cctsdb_yolo_2023",
    "cersyve",
    "cgan_2023",
    "cifar100",  # alias of cifar100_2024
    "cifar100_2024",
    "collins_aerospace_benchmark",
    "collins_rul_cnn_2022",
    "collins_rul_cnn_2023",  # alias of collins_rul_cnn_2022
    "cora",  # alias of cora_2024
    "cora_2024",
    "dist_shift_2023",
    "linearizenn",
    "lsnc",
    "lsnc_relu",
    "malbeware",
    "metaroom_2023",
    "ml4acopf_2023",
    "ml4acopf_2024",
    "nn4sys",  # alias of nn4sys_2023
    "nn4sys_2023",
    "relusplitter",
    "safenlp",  # alias of safenlp_2024
    "safenlp_2024",
    "sat_relu",
    "soundnessbench",
    "tinyimagenet",  # alias of tinyimagenet_2024
    "tinyimagenet_2024",
    "tllverifybench_2023",
    "traffic_signs_recognition_2023",
    "vggnet16_2022",
    "vggnet16_2023",  # alias of vggnet16_2022
    "vit_2023",
    "yolo_2023",
    "test",
)


@lru_cache(maxsize=128)
def get_preset(benchmark_name: str, model_name: str | None = None) -> OptimizationConfig:
    """Get preset optimization configuration for a benchmark.

    :param benchmark_name: Benchmark identifier (e.g., 'acasxu_2023', 'cgan_2023').

    :param model_name: Optional model filename for per-model exceptions.

    :return: Optimization configuration optimized for the benchmark
    """
    # Model-specific exceptions for nn4sys_2023 (some have batch dim, some don't)
    if (
        benchmark_name == "nn4sys_2023"
        and model_name
        and "pensieve" in model_name
        and "parallel" in model_name
    ):
        return OptimizationConfig(
            fuse_matmul_add=True,
            has_batch_dim=False,
        )

    # Resolve aliases (e.g. "cifar100" -> "cifar100_2024")
    benchmark_name = _ALIASES.get(benchmark_name, benchmark_name)

    # Benchmarks that only need constant folding
    if benchmark_name in _CF_ONLY_PRESETS:
        return _CONSTANT_FOLDING_PRESET

    # Presets with non-default optimization flags only.
    # Benchmarks that only need constant_folding fall through to the default.
    presets = {
        "acasxu_2023": OptimizationConfig(
            fuse_matmul_add=True,
            remove_redundant_operations=True,
            constant_folding=True,
        ),
        "cctsdb_yolo_2023": OptimizationConfig(
            constant_folding=True,
            has_batch_dim=False,
        ),
        "cifar100_2024": OptimizationConfig(
            fuse_conv_bn=True,
            fuse_bn_conv=True,
            constant_folding=True,
        ),
        "cgan_2023": OptimizationConfig(
            fuse_conv_bn=True,
            fuse_bn_conv=True,
            fuse_conv_transpose_bn=True,
            fuse_gemm_reshape_bn=False,  # Disabled: padding issues with cGAN structure
            fuse_bn_reshape_gemm=False,  # Disabled: padding issues with cGAN structure
            constant_folding=True,
            remove_redundant_operations=True,
            has_batch_dim=True,
        ),
        "cersyve": OptimizationConfig(
            fuse_gemm_gemm=True,
            constant_folding=True,
        ),
        "collins_rul_cnn_2022": OptimizationConfig(
            simplify_conv_to_flatten_gemm=True,
            remove_redundant_operations=True,
            constant_folding=True,
        ),
        "cora_2024": OptimizationConfig(
            fuse_matmul_add=True,
            constant_folding=True,
        ),
        "dist_shift_2023": OptimizationConfig(
            remove_redundant_operations=True,
            constant_folding=True,
        ),
        "nn4sys_2023": OptimizationConfig(
            fuse_matmul_add=True,
            constant_folding=True,
        ),
        "safenlp_2024": OptimizationConfig(
            fuse_matmul_add=True,
            constant_folding=True,
        ),
        "tinyimagenet_2024": OptimizationConfig(
            fuse_conv_bn=True,
            constant_folding=True,
        ),
        "tllverifybench_2023": OptimizationConfig(
            fuse_matmul_add=True,
            constant_folding=True,
        ),
        "vit_2023": OptimizationConfig(
            constant_folding=True,
            fuse_matmul_add=True,
            fuse_transpose_bn_transpose=True,
            fuse_gemm_gemm=True,
            fuse_bn_gemm=True,
            remove_redundant_operations=True,
        ),
        "test": all_optimizations(has_batch_dim=False),
    }

    return presets.get(benchmark_name, OptimizationConfig())


def all_optimizations(has_batch_dim: bool = True) -> OptimizationConfig:
    """Get configuration with all optimizations enabled.

    :param has_batch_dim: Whether model has batch dimension.

    :return: Optimization configuration with all flags True
    """
    return OptimizationConfig(
        fuse_matmul_add=True,
        fuse_conv_bn=True,
        fuse_bn_conv=True,
        fuse_conv_transpose_bn=True,
        fuse_bn_conv_transpose=True,
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
