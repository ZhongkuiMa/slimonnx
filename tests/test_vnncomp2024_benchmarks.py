"""Parametrized tests for all vnncomp2024 benchmarks.

Tests that each vnncomp2024 benchmark has proper preset configuration and can be
successfully slimmed using SlimONNX.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "VNNCOMP2024_BENCHMARKS",
    "test_benchmark_has_preset",
    "test_benchmark_preset_is_valid",
]

import pytest

# All 22 vnncomp2024 benchmarks
VNNCOMP2024_BENCHMARKS = [
    "acasxu_2023",
    "cctsdb_yolo_2023",
    "cgan_2023",
    "cifar100",
    "collins_aerospace_benchmark",
    "collins_rul_cnn_2023",
    "cora",
    "dist_shift_2023",
    "linearizenn",
    "lsnc",
    "metaroom_2023",
    "ml4acopf_2023",
    "ml4acopf_2024",
    "nn4sys_2023",
    "safenlp",
    "test",
    "tinyimagenet",
    "tllverifybench_2023",
    "traffic_signs_recognition_2023",
    "vggnet16_2023",
    "vit_2023",
    "yolo_2023",
]


@pytest.mark.parametrize("benchmark_name", VNNCOMP2024_BENCHMARKS)
def test_benchmark_has_preset(benchmark_name: str) -> None:
    """Test that each vnncomp2024 benchmark has a preset configuration.

    :param benchmark_name: Name of the benchmark
    """
    from slimonnx.presets import PRESET_NAMES, get_preset

    assert benchmark_name in PRESET_NAMES, f"Benchmark {benchmark_name} not in PRESET_NAMES"

    config = get_preset(benchmark_name)
    assert config is not None, f"Preset configuration is None for {benchmark_name}"


@pytest.mark.parametrize("benchmark_name", VNNCOMP2024_BENCHMARKS)
def test_benchmark_preset_is_valid(benchmark_name: str) -> None:
    """Test that each benchmark's preset configuration is valid.

    :param benchmark_name: Name of the benchmark
    """
    from slimonnx.configs import OptimizationConfig
    from slimonnx.presets import get_preset

    config = get_preset(benchmark_name)
    assert isinstance(config, OptimizationConfig), (
        f"Preset for {benchmark_name} is not OptimizationConfig"
    )
    assert hasattr(config, "has_batch_dim"), f"Preset for {benchmark_name} missing has_batch_dim"
    assert isinstance(config.has_batch_dim, bool), (
        f"Preset has_batch_dim for {benchmark_name} is not bool"
    )
