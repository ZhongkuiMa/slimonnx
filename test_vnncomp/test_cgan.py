import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX()
    onnx_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cgan_2023/onnx"
        "/cGAN_imgSz32_nCh_3_small_transformer.onnx"
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
        constant_to_initializer=True,
        fuse_constant_nodes=True,
        simplify_node_name=True,
        reorder_by_strict_topological_order=True,
        verbose=True,
    )
