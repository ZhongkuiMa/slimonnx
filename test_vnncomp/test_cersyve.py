import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX(verbose=True)
    onnx_path = (
        "../../../vnncomp2025_benchmarks/benchmarks/cersyve/onnx/"
        "pendulum_pretrain_con.onnx"
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
        fuse_gemm_gemm=True,
    )
