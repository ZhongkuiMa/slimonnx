from slimonnx import SlimONNX
import onnx

if __name__ == "__main__":
    slimonnx = SlimONNX()
    onnx_path = "../nets/TinyImageNet_resnet_medium.onnx"

    # Convert the model to version 22 to avoid many inconsistencies
    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, target_version=22)
    onnx_path = onnx_path.replace(".onnx", "_v22.onnx")
    onnx.save(model, onnx_path)

    target_path = onnx_path.replace(".onnx", "_simplified.onnx")

    slimonnx.slim(
        onnx_path,
        target_path,
        constant_to_initializer=True,
        fuse_conv_bn=True,
        reorder_by_strict_topological_order=True,
        simplify_node_name=True,
        verbose=True,
    )
