import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX()
    onnx_path = (
        "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/onnx/"
        # "lindex.onnx"
        # "lindex_deep.onnx"
        # "mscn_128d.onnx"
        # "mscn_128d_dual.onnx"
        # "mscn_2048d.onnx"
        # "mscn_2048d_dual.onnx"
        # "pensieve_big_parallel.onnx"
        # "pensieve_big_simple.onnx"
        # "pensieve_mid_parallel.onnx"
        # "pensieve_mid_simple.onnx"
        # "pensieve_small_parallel.onnx"
        "pensieve_small_simple.onnx"
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
        fuse_matmul_add=True,
        simplify_gemm=True,
    )
