import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX(verbose=True)
    onnx_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cgan_2023/onnx/"
        # "cGAN_imgSz32_nCh_1.onnx"
        # "cGAN_imgSz32_nCh_1_transposedConvPadding_1.onnx"
        # "cGAN_imgSz32_nCh_3.onnx"
        # "cGAN_imgSz32_nCh_3_nonlinear_activations.onnx"
        # "cGAN_imgSz32_nCh_3_upsample.onnx"
        # "cGAN_imgSz64_nCh_1.onnx"
        # "cGAN_imgSz64_nCh_3.onnx"
        # "cGAN_imgSz32_nCh_3_small_transformer.onnx"
        "cGAN_imgSz32_nCh_3_small_transformer.onnx"
    )

    # Convert the model to version 22 to avoid many inconsistencies
    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, target_version=22)
    onnx_path = onnx_path.replace(".onnx", "_v22.onnx")
    onnx.save(model, onnx_path)

    target_path = onnx_path.replace(".onnx", "_simplified.onnx")
    # NOTE: This model has no batch dim.
    slimonnx.slim(
        onnx_path,
        target_path,
        fuse_constant_nodes=True,
        remove_redundant_operations=True,
        fuse_bn_conv=True,
        fuse_conv_bn=True,
        fuse_convtransposed_bn=True,
        fuse_bn_convtransposed=True,
    )
