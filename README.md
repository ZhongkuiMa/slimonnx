# slimonnx

`slimonnx` is a tool to simplify or optimize an ONNX model (.onnx file).

## Installation

The following packages are required to be installed:

- `onnx`
- `numpy`

```bash


## Usage

Use `SlimONNX` class in `slimonnx/slimonnx.py` to simplify or optimize an ONNX model.

The following is an example:

```python
from slimonnx.slimonnx import SlimONNX

slim = SlimONNX()
slim.slim("model.onnx", "slim_model.onnx")
```

## Supported Features

Refer to the following docstring of `slim` method in `SlimONNX` class for supported features.

```python
def slim(
        self,
        onnx_path: str,
        target_parth: str | None = None,
        fuse_matmul_add: bool = False,
        fuse_gemm_reshape_bn: bool = False,
        fuse_bn_reshape_gemm: bool = False,
        fuse_bn_gemm: bool = False,
        fuse_transpose_bn_transpose: bool = False,
        fuse_gemm_gemm: bool = False,
        fuse_conv_bn: bool = False,
        fuse_bn_conv: bool = False,
        fuse_transposedconv_bn: bool = False,
        shape_to_initializer: bool = False,
        simplify_node_name: bool = True,
        reorder_by_strict_topological_order: bool = True,
):
    """
    Simplify the ONNX model by fusing some nodes.

    By default, all the node docstring will be removed from the ONNX model.

    :param model: The ONNX model to simplify.
    :param target_parth: The path to save the simplified ONNX model.
    :param fuse_matmul_add: Fuse a MatMul and an Add node into a single Gemm node.
    :param fuse_gemm_reshape_bn: Fuse a Gemm, a Reshape, and a BatchNormalization
        node into a Gemm and a Reshape node.
    :param fuse_bn_reshape_gemm: Fuse a BatchNormalization, a Reshape, and a Gemm
        node into a Reshape and a Gemm node.
    :param fuse_bn_gemm: Fuse a BatchNormalization and a Gemm node into a Gemm node.
    :param fuse_transpose_bn_transpose: Fuse a Transpose, a BatchNormalization,
        and a Transpose node into a Gemm node.
    :param fuse_gemm_gemm: Fuse two Gemm nodes into a single Gemm node.
    :param fuse_conv_bn: Fuse a Conv and BatchNormalization node into a Conv node.
    :param fuse_bn_conv: Fuse a BatchNormalization and a Conv node into a Conv node.
    :param fuse_transposedconv_bn: Fuse a ConvTranspose and a BatchNormalization
        node into a ConvTranspose node.
    :param shape_to_initializer: Convert the shape nodes to initializers.
    :param simplify_node_name: Simplify the node name by topological order.
    :param reorder_by_strict_topological_order: Reorder the nodes by topological
        order and simiplify their names.

    :return: The simplified ONNX model.
    """
```

ATTENSION: The resulting model maybe not supported by the shape infering of `onnxruntime`.
