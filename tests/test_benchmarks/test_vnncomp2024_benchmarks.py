"""Parametrized tests for all vnncomp2024 benchmarks.

Tests that each vnncomp2024 benchmark has proper preset configuration and can be
successfully slimmed using SlimONNX.
"""

__docformat__ = "restructuredtext"

import pytest

from slimonnx.configs import OptimizationConfig
from slimonnx.presets import PRESET_NAMES, get_preset

# All 22 vnncomp2024 benchmarks
VNNCOMP2024_BENCHMARKS = [
    "acasxu_2023",
]


@pytest.mark.parametrize("benchmark_name", VNNCOMP2024_BENCHMARKS)
def test_benchmark_has_preset(benchmark_name: str) -> None:
    """Test that each vnncomp2024 benchmark has a preset configuration.

    :param benchmark_name: Name of the benchmark.

    """
    assert benchmark_name in PRESET_NAMES, f"Benchmark {benchmark_name} not in PRESET_NAMES"

    config = get_preset(benchmark_name)
    assert config, f"Preset configuration is None for {benchmark_name}"


@pytest.mark.parametrize("benchmark_name", VNNCOMP2024_BENCHMARKS)
def test_benchmark_preset_is_valid(benchmark_name: str) -> None:
    """Test that each benchmark's preset configuration is valid.

    :param benchmark_name: Name of the benchmark.

    """
    config = get_preset(benchmark_name)
    assert isinstance(config, OptimizationConfig), (
        f"Preset for {benchmark_name} is not OptimizationConfig"
    )
    assert hasattr(config, "has_batch_dim"), f"Preset for {benchmark_name} missing has_batch_dim"
    assert isinstance(config.has_batch_dim, bool), (
        f"Preset has_batch_dim for {benchmark_name} is not bool"
    )
