"""Tests for optimization preset configurations."""

__docformat__ = "restructuredtext"

import pytest

from slimonnx.configs import OptimizationConfig
from slimonnx.presets import PRESET_NAMES, all_optimizations, get_preset


class TestPresetNames:
    """Test PRESET_NAMES constant."""

    def test_is_tuple(self):
        """Test that PRESET_NAMES is a tuple."""
        assert isinstance(PRESET_NAMES, tuple)

    def test_not_empty(self):
        """Test that PRESET_NAMES is not empty."""
        assert len(PRESET_NAMES) > 0

    def test_contains_known_presets(self):
        """Test that PRESET_NAMES contains expected presets."""
        assert "acasxu_2023" in PRESET_NAMES
        assert "cifar100" in PRESET_NAMES
        assert "cgan_2023" in PRESET_NAMES
        assert "vit_2023" in PRESET_NAMES
        assert "test" in PRESET_NAMES


class TestGetPreset:
    """Test get_preset function."""

    @pytest.mark.parametrize(
        "preset_name",
        [
            pytest.param("acasxu_2023", id="acasxu_2023"),
            pytest.param("cifar100", id="cifar100"),
            pytest.param("cifar100_2024", id="cifar100_2024"),
            pytest.param("tinyimagenet", id="tinyimagenet"),
            pytest.param("tinyimagenet_2024", id="tinyimagenet_2024"),
        ],
    )
    def test_returns_config_for_preset(self, preset_name):
        """Test that get_preset returns OptimizationConfig for various presets."""
        config = get_preset(preset_name)
        assert isinstance(config, OptimizationConfig)

    def test_acasxu_2023_flags(self):
        """Test acasxu_2023 preset enables matmul_add, redundant ops, and folding."""
        config = get_preset("acasxu_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True

    def test_cgan_2023_flags(self):
        """Test cgan_2023 preset enables conv/bn fusion and folding."""
        config = get_preset("cgan_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.fuse_conv_transpose_bn is True
        assert config.constant_folding is True
        assert config.has_batch_dim is True

    def test_vit_2023_flags(self):
        """Test vit_2023 preset enables matmul_add and various fusion patterns."""
        config = get_preset("vit_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.fuse_transpose_bn_transpose is True
        assert config.fuse_gemm_gemm is True
        assert config.fuse_bn_gemm is True
        assert config.remove_redundant_operations is True

    def test_cifar100_flags(self):
        """Test cifar100 preset enables conv/bn fusion and folding."""
        config = get_preset("cifar100")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.constant_folding is True

    def test_cifar100_2024_flags(self):
        """Test cifar100_2024 preset enables conv/bn fusion and folding."""
        config = get_preset("cifar100_2024")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.constant_folding is True

    def test_tinyimagenet_flags(self):
        """Test tinyimagenet preset enables conv/bn fusion and folding."""
        config = get_preset("tinyimagenet")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.constant_folding is True

    def test_tinyimagenet_2024_flags(self):
        """Test tinyimagenet_2024 preset enables conv/bn fusion and folding."""
        config = get_preset("tinyimagenet_2024")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.constant_folding is True

    def test_collins_rul_cnn_2022_flags(self):
        """Test collins_rul_cnn_2022 preset enables simplify and folding."""
        config = get_preset("collins_rul_cnn_2022")

        assert isinstance(config, OptimizationConfig)
        assert config.simplify_conv_to_flatten_gemm is True
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True

    @pytest.mark.parametrize(
        "preset_name",
        [
            pytest.param("nn4sys", id="nn4sys"),
            pytest.param("nn4sys_2023", id="nn4sys_2023"),
            pytest.param("safenlp_2024", id="safenlp_2024"),
        ],
    )
    def test_matmul_add_and_constant_folding(self, preset_name):
        """Test presets that enable fuse_matmul_add and constant_folding."""
        config = get_preset(preset_name)

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_nn4sys_2023_pensieve_parallel(self):
        """Test nn4sys_2023 model-specific exception for pensieve_parallel."""
        config = get_preset("nn4sys_2023", "pensieve_parallel_model.onnx")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.has_batch_dim is False

    def test_nn4sys_2023_other_model(self):
        """Test nn4sys_2023 with non-matching model name."""
        config = get_preset("nn4sys_2023", "other_model.onnx")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.has_batch_dim is not False

    @pytest.mark.parametrize(
        "preset_name",
        [
            pytest.param("lsnc", id="lsnc"),
            pytest.param("yolo_2023", id="yolo_2023"),
            pytest.param("lsnc_relu", id="lsnc_relu"),
        ],
    )
    def test_constant_folding_only(self, preset_name):
        """Test presets that only require constant_folding to be enabled."""
        config = get_preset(preset_name)

        assert isinstance(config, OptimizationConfig)
        assert config.constant_folding is True

    def test_preset_named_test(self):
        """Test getting test preset."""
        config = get_preset("test")

        assert isinstance(config, OptimizationConfig)
        # test preset uses all_optimizations with has_batch_dim=False
        assert config.has_batch_dim is False
        assert config.constant_folding is True

    def test_unknown_preset_returns_default_config(self):
        """Test getting unknown preset returns default config."""
        config = get_preset("unknown_preset")

        assert isinstance(config, OptimizationConfig)
        # Default config should have all False/None
        assert config.fuse_matmul_add is not True
        assert config.constant_folding is not True

    def test_returns_optimization_config(self):
        """Test that all presets return OptimizationConfig instances."""
        for preset_name in ["acasxu_2023", "cgan_2023", "cifar100_2024"]:
            config = get_preset(preset_name)
            assert isinstance(config, OptimizationConfig)

    def test_returns_same_instance_on_repeated_call(self):
        """Test that get_preset uses caching."""
        # Call twice and should get same object (due to lru_cache)
        config1 = get_preset("acasxu_2023")
        config2 = get_preset("acasxu_2023")

        assert config1 is config2

    def test_dist_shift_2023(self):
        """Test getting dist_shift_2023 preset."""
        config = get_preset("dist_shift_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True

    def test_cersyve_preset_enables_gemm_gemm_fusion(self):
        """Test getting cersyve preset."""
        config = get_preset("cersyve")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_gemm_gemm is True
        assert config.constant_folding is True

    def test_cctsdb_yolo_2023(self):
        """Test getting cctsdb_yolo_2023 preset."""
        config = get_preset("cctsdb_yolo_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.constant_folding is True
        assert config.has_batch_dim is False


class TestAllOptimizations:
    """Test all_optimizations function."""

    def test_has_batch_dim_true_by_default(self):
        """Test all_optimizations with default has_batch_dim."""
        config = all_optimizations()

        assert isinstance(config, OptimizationConfig)
        assert config.has_batch_dim is True
        assert config.fuse_matmul_add is True
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.constant_folding is True
        assert config.simplify_conv_to_flatten_gemm is True
        assert config.remove_redundant_operations is True

    @pytest.mark.parametrize(
        "has_batch_dim",
        [
            pytest.param(True, id="batch_dim_true"),
            pytest.param(False, id="batch_dim_false"),
        ],
    )
    def test_with_batch_dim(self, has_batch_dim):
        """Test all_optimizations with explicit has_batch_dim values."""
        config = all_optimizations(has_batch_dim=has_batch_dim)

        assert isinstance(config, OptimizationConfig)
        assert config.has_batch_dim is has_batch_dim
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_all_flags_true(self):
        """Test that all_optimizations enables all optimization flags."""
        config = all_optimizations()

        # Check all optimization flags are True
        assert config.fuse_matmul_add is True
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.fuse_conv_transpose_bn is True
        assert config.fuse_bn_conv_transpose is True
        assert config.fuse_gemm_reshape_bn is True
        assert config.fuse_bn_reshape_gemm is True
        assert config.fuse_bn_gemm is True
        assert config.fuse_transpose_bn_transpose is True
        assert config.fuse_gemm_gemm is True
        assert config.simplify_conv_to_flatten_gemm is True
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True
        assert config.simplify_node_name is True

    def test_returns_optimization_config(self):
        """Test that all_optimizations returns OptimizationConfig."""
        config = all_optimizations()
        assert isinstance(config, OptimizationConfig)

    def test_different_batch_dims_differ_only_in_has_batch_dim(self):
        """Test all_optimizations with both batch_dim values."""
        config_with_batch = all_optimizations(has_batch_dim=True)
        config_without_batch = all_optimizations(has_batch_dim=False)

        # Should differ only in has_batch_dim
        assert config_with_batch.has_batch_dim is True
        assert config_without_batch.has_batch_dim is False
        assert config_with_batch.fuse_matmul_add == config_without_batch.fuse_matmul_add
