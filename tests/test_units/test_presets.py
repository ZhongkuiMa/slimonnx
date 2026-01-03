"""Tests for optimization preset configurations."""

from slimonnx.configs import OptimizationConfig
from slimonnx.presets import PRESET_NAMES, all_optimizations, get_preset


class TestPresetNames:
    """Test PRESET_NAMES constant."""

    def test_preset_names_is_tuple(self):
        """Test that PRESET_NAMES is a tuple."""
        assert isinstance(PRESET_NAMES, tuple)

    def test_preset_names_not_empty(self):
        """Test that PRESET_NAMES is not empty."""
        assert len(PRESET_NAMES) > 0

    def test_preset_names_contains_known_presets(self):
        """Test that PRESET_NAMES contains expected presets."""
        assert "acasxu_2023" in PRESET_NAMES
        assert "cifar100" in PRESET_NAMES
        assert "cgan_2023" in PRESET_NAMES
        assert "vit_2023" in PRESET_NAMES
        assert "test" in PRESET_NAMES


class TestGetPreset:
    """Test get_preset function."""

    def test_get_preset_acasxu_2023(self):
        """Test getting acasxu_2023 preset."""
        config = get_preset("acasxu_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True

    def test_get_preset_cgan_2023(self):
        """Test getting cgan_2023 preset."""
        config = get_preset("cgan_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.fuse_conv_transpose_bn is True
        assert config.constant_folding is True
        assert config.has_batch_dim is True

    def test_get_preset_vit_2023(self):
        """Test getting vit_2023 preset."""
        config = get_preset("vit_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.fuse_transpose_bn_transpose is True
        assert config.fuse_gemm_gemm is True
        assert config.fuse_bn_gemm is True
        assert config.remove_redundant_operations is True

    def test_get_preset_cifar100(self):
        """Test getting cifar100 preset (alias)."""
        config = get_preset("cifar100")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.constant_folding is True

    def test_get_preset_cifar100_2024(self):
        """Test getting cifar100_2024 preset."""
        config = get_preset("cifar100_2024")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.fuse_bn_conv is True
        assert config.constant_folding is True

    def test_get_preset_tinyimagenet(self):
        """Test getting tinyimagenet preset (alias)."""
        config = get_preset("tinyimagenet")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.constant_folding is True

    def test_get_preset_tinyimagenet_2024(self):
        """Test getting tinyimagenet_2024 preset."""
        config = get_preset("tinyimagenet_2024")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_conv_bn is True
        assert config.constant_folding is True

    def test_get_preset_collins_rul_cnn_2022(self):
        """Test getting collins_rul_cnn_2022 preset."""
        config = get_preset("collins_rul_cnn_2022")

        assert isinstance(config, OptimizationConfig)
        assert config.simplify_conv_to_flatten_gemm is True
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True

    def test_get_preset_nn4sys(self):
        """Test getting nn4sys preset."""
        config = get_preset("nn4sys")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_get_preset_nn4sys_2023_general(self):
        """Test getting nn4sys_2023 preset without model_name."""
        config = get_preset("nn4sys_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_get_preset_nn4sys_2023_pensieve_parallel(self):
        """Test nn4sys_2023 model-specific exception for pensieve_parallel."""
        config = get_preset("nn4sys_2023", "pensieve_parallel_model.onnx")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.has_batch_dim is False

    def test_get_preset_nn4sys_2023_other_model(self):
        """Test nn4sys_2023 with non-matching model name."""
        config = get_preset("nn4sys_2023", "other_model.onnx")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.has_batch_dim is not False

    def test_get_preset_lsnc(self):
        """Test getting lsnc preset."""
        config = get_preset("lsnc")

        assert isinstance(config, OptimizationConfig)
        assert config.constant_folding is True

    def test_get_preset_yolo_2023(self):
        """Test getting yolo_2023 preset."""
        config = get_preset("yolo_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.constant_folding is True

    def test_get_preset_test(self):
        """Test getting test preset."""
        config = get_preset("test")

        assert isinstance(config, OptimizationConfig)
        # test preset uses all_optimizations with has_batch_dim=False
        assert config.has_batch_dim is False
        assert config.constant_folding is True

    def test_get_preset_unknown(self):
        """Test getting unknown preset returns default config."""
        config = get_preset("unknown_preset")

        assert isinstance(config, OptimizationConfig)
        # Default config should have all False/None
        assert config.fuse_matmul_add is not True
        assert config.constant_folding is not True

    def test_get_preset_returns_optimization_config(self):
        """Test that all presets return OptimizationConfig instances."""
        for preset_name in ["acasxu_2023", "cgan_2023", "cifar100_2024"]:
            config = get_preset(preset_name)
            assert isinstance(config, OptimizationConfig)

    def test_get_preset_caching(self):
        """Test that get_preset uses caching."""
        # Call twice and should get same object (due to lru_cache)
        config1 = get_preset("acasxu_2023")
        config2 = get_preset("acasxu_2023")

        assert config1 is config2

    def test_get_preset_dist_shift_2023(self):
        """Test getting dist_shift_2023 preset."""
        config = get_preset("dist_shift_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.remove_redundant_operations is True
        assert config.constant_folding is True

    def test_get_preset_safenlp_2024(self):
        """Test getting safenlp_2024 preset."""
        config = get_preset("safenlp_2024")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_get_preset_cersyve(self):
        """Test getting cersyve preset."""
        config = get_preset("cersyve")

        assert isinstance(config, OptimizationConfig)
        assert config.fuse_gemm_gemm is True
        assert config.constant_folding is True

    def test_get_preset_lsnc_relu(self):
        """Test getting lsnc_relu preset."""
        config = get_preset("lsnc_relu")

        assert isinstance(config, OptimizationConfig)
        assert config.constant_folding is True

    def test_get_preset_cctsdb_yolo_2023(self):
        """Test getting cctsdb_yolo_2023 preset."""
        config = get_preset("cctsdb_yolo_2023")

        assert isinstance(config, OptimizationConfig)
        assert config.constant_folding is True
        assert config.has_batch_dim is False


class TestAllOptimizations:
    """Test all_optimizations function."""

    def test_all_optimizations_default(self):
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

    def test_all_optimizations_with_batch_dim_true(self):
        """Test all_optimizations with has_batch_dim=True."""
        config = all_optimizations(has_batch_dim=True)

        assert isinstance(config, OptimizationConfig)
        assert config.has_batch_dim is True
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_all_optimizations_with_batch_dim_false(self):
        """Test all_optimizations with has_batch_dim=False."""
        config = all_optimizations(has_batch_dim=False)

        assert isinstance(config, OptimizationConfig)
        assert config.has_batch_dim is False
        assert config.fuse_matmul_add is True
        assert config.constant_folding is True

    def test_all_optimizations_all_flags_true(self):
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

    def test_all_optimizations_returns_optimization_config(self):
        """Test that all_optimizations returns OptimizationConfig."""
        config = all_optimizations()
        assert isinstance(config, OptimizationConfig)

    def test_all_optimizations_different_batch_dims(self):
        """Test all_optimizations with both batch_dim values."""
        config_with_batch = all_optimizations(has_batch_dim=True)
        config_without_batch = all_optimizations(has_batch_dim=False)

        # Should differ only in has_batch_dim
        assert config_with_batch.has_batch_dim is True
        assert config_without_batch.has_batch_dim is False
        assert config_with_batch.fuse_matmul_add == config_without_batch.fuse_matmul_add
