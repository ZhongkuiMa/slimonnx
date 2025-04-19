# ShapeONNX: Shape Inference for ONNX Models

ShapeONNX is a tool to infer the shapes of ONNX models.

*I know [onnxruntime](https://onnxruntime.ai/) and it is a GREAT project. It has a function to infer the shape of a ONNX model and it works in almost all cases. But here our ShapeONNX aims something different.*

## Why you need it?

Sometimes, an ONNX model has some node operations involving extracting a shape of a tensor and operating on these shapes. This is a common operation in some PyTorch moddules including gathering, slicing, and so on. When onnxruntime encounters these operations, it wannot be able to infer the shape of the model and output a wrong result. Here, ShapeONNX appears to help you.