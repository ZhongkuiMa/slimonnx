import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX(verbose=True)
    onnx_path = (
        "../../../vnncomp2024_benchmarks/benchmarks/cctsdb_yolo_2023/onnx/patch-1.onnx"
    )
    # onnx_path = (
    #     "../../vnncomp2024_benchmarks/benchmarks/cctsdb_yolo_2023/onnx/patch-3.onnx"
    # )

    # Convert the model to version 22 to avoid many inconsistencies
    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, target_version=21)
    onnx_path = onnx_path.replace(".onnx", "_v21.onnx")
    onnx.save(model, onnx_path)

    target_path = onnx_path.replace(".onnx", "_simplified.onnx")
    # NOTE: This model has no batch dim.
    slimonnx.verbose = True
    slimonnx.slim(
        onnx_path,
        target_path,
        has_batch_dim=False,
        fuse_constant_nodes=True,
    )
