import onnx

from slimonnx import SlimONNX

if __name__ == "__main__":
    slimonnx = SlimONNX()
    onnx_path = (
        "../../vnncomp2024_benchmarks/benchmarks/collins_rul_cnn_2023/onnx"
        "/NN_rul_full_window_40.onnx"
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
        simplify_conv_to_flatten_gemm=True,
        simplify_node_name=True,
        remove_redundant_reshape=True,
        reorder_by_strict_topological_order=True,
        verbose=True,
    )
