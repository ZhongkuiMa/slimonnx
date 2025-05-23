import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX()
    onnx_path = (
        "../../vnncomp2024_benchmarks/benchmarks/vit_2023/onnx/" "ibp_3_3_8.onnx"
    )

    # Convert the model to version 22 to avoid many inconsistencies
    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, target_version=22)
    onnx_path = onnx_path.replace(".onnx", "_v22.onnx")
    onnx.save(model, onnx_path)

    target_path = onnx_path.replace(".onnx", "_simplified.onnx")

    slimonnx.slim(
        onnx_path,
        target_path,
        fuse_constant_nodes=True,
        fuse_matmul_add=True,
        # remove_redundant_operations=True,
        fuse_transpose_bn_transpose=True,
        fuse_bn_gemm=True,
        fuse_gemm_gemm=True,
    )
