"""BatchNorm + Conv fusion with padding support.

This module provides advanced BN-Conv fusion that handles Conv layers with padding.

__docformat__ = "restructuredtext"
__all__ = ["_fuse_bn_conv_with_padding"]

Mathematical Background:
-----------------------
The standard BN-Conv fusion fails when Conv has padding because:

Original flow:
    x -> BN -> pad(0) -> Conv
    BN(x) = bn_weight * x + bn_bias
    After padding: regions outside original have value 0
    Conv sees: pad(bn_weight * x + bn_bias)

Naive fused flow (INCORRECT):
    x -> pad(0) -> Fused_Conv
    Fused_Conv sees: pad(x)
    But this misses bn_bias contribution!

Correct Solution Approach 1: Explicit Pad Node
---------------------------------------------
Transform the graph to make padding explicit:
    x -> BN -> Pad -> Conv(no_pad)

Then fuse becomes:
    x -> Fused_BN_Pad -> Conv(no_pad)

Where Fused_BN_Pad computes:
    output = pad(bn_weight * x + bn_bias)

This is correct but requires:
1. Splitting Conv's padding into separate Pad operation
2. Adjusting Conv's attributes (pads=[0,0,0,0])
3. Maintaining correct output shapes

Correct Solution Approach 2: Adjusted Bias
------------------------------------------
Keep the fusion but adjust the bias term to account for padding.

For each output pixel, we need to know which input pixels contribute
and whether they came from padding vs original image.

This is complex because it requires:
1. Analyzing Conv's receptive field for each output pixel
2. Computing different bias corrections for edge vs center pixels
3. Creating position-dependent bias (not supported in standard Conv)

Recommended Solution: Approach 1 (Explicit Pad)
==============================================
"""

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from ._utils import (
    _is_only_next_node,
    _get_batchnorm_params,
    _get_conv_params,
    compute_batchnorm_fusion_params,
)


def _fuse_bn_conv_with_padding(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Transform BN+Conv pattern by making Conv's padding explicit.

    Transformation:
        BN -> Conv(pad>0)
        =>
        BN -> Pad -> Conv(pad=0)

    This transformation makes padding explicit, which allows the BN-Conv pattern
    to be potentially fused by other optimizations (e.g., standard BN-Conv fusion
    can now work since Conv has no padding).

    Note: This function does NOT fuse BN into Conv. It only separates Conv's
    implicit padding into an explicit Pad node. The actual BN-Conv fusion
    should be done by calling _fuse_conv_bn_or_bn_conv afterwards.

    :param nodes: List of nodes in the graph
    :param initializers: Dictionary of initializers
    :return: Optimized list of nodes

    Implementation:
    1. Find BN -> Conv(pad>0) patterns
    2. For each pattern:
       a. Create explicit Pad node with Conv's padding
       b. Create new Conv node with pad=0
       c. Replace Conv with Pad + Conv (keep BN unchanged)

    Example:
        Original:
            BN(scale, bias, mean, var) -> Conv(weight, bias, pads=[1,1,1,1])

        After transformation:
            BN(scale, bias, mean, var) -> Pad(pads=[1,1,1,1]) -> Conv(weight, bias, pads=[0,0,0,0])

        Then standard BN-Conv fusion can apply:
            Identity -> Pad -> Conv(fused_weight, fused_bias, pads=[0,0,0,0])
    """
    new_nodes = []
    pre_node = None

    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        # Check for BN -> Conv pattern
        if not (pre_node.op_type == "BatchNormalization" and node.op_type == "Conv"):
            pre_node = node
            continue

        bn_node, conv_node = pre_node, node

        # Get Conv parameters
        _, _, attrs = _get_conv_params(conv_node, initializers, False)

        # Check if Conv has padding
        if not any(p != 0 for p in attrs["pads"]):
            # No padding - can use standard fusion
            pre_node = node
            continue

        # Get Conv parameters to extract padding
        _, _, attrs = _get_conv_params(conv_node, initializers, False)

        # Create Pad node between BN and Conv
        # Pad takes BN's output and applies Conv's padding
        pad_output_name = bn_node.output[0] + "_padded"
        pad_node = _create_pad_node_from_conv_pads(
            attrs["pads"], bn_node.output[0], pad_output_name, initializers
        )

        # Create Conv node with pads=[0,0,0,0]
        zero_pads = [0] * len(attrs["pads"])
        new_conv_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=[pad_output_name]
            + list(conv_node.input[1:]),  # Pad output + original weights/bias
            outputs=conv_node.output,
            name=conv_node.name,
            kernel_shape=attrs["kernel_shape"],
            pads=zero_pads,
            strides=attrs["strides"],
            dilations=attrs["dilations"],
            group=attrs["group"],
            auto_pad=attrs["auto_pad"],
        )

        # Replace Conv with Pad + new Conv (keep BN)
        # Remove only the last node (Conv)
        new_nodes.pop()

        # Add Pad and new Conv
        new_nodes.append(pad_node)
        new_nodes.append(new_conv_node)

        pre_node = new_conv_node

    return new_nodes


def _create_pad_node_from_conv_pads(
    pads: list[int],
    input_name: str,
    output_name: str,
    initializers: dict[str, TensorProto],
) -> NodeProto:
    """Create a Pad node from Conv's pads attribute.

    Conv pads format: [x1_begin, x2_begin, x1_end, x2_end] for 2D
    ONNX Pad format: [0, 0, x1_begin, x2_begin, 0, 0, x1_end, x2_end] for NCHW

    :param pads: Conv's pads attribute
    :param input_name: Input tensor name
    :param output_name: Output tensor name
    :param initializers: Dictionary of initializers (to add pads tensor)
    :return: Pad node
    """
    # Convert Conv pads to ONNX Pad format
    # Conv: [top, left, bottom, right] for 2D with NCHW
    # Pad: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
    pad_value = np.array(
        [
            0,
            0,
            pads[0],
            pads[1],  # Begin: [0, 0, top, left]
            0,
            0,
            pads[2],
            pads[3],  # End: [0, 0, bottom, right]
        ],
        dtype=np.int64,
    )

    # For ONNX opset 11+, pads must be an input tensor, not an attribute
    pads_name = output_name + "_pads"
    pads_tensor = onnx.numpy_helper.from_array(pad_value, pads_name)
    initializers[pads_name] = pads_tensor

    # Create constant_value initializer
    constant_value_name = output_name + "_value"
    constant_value = onnx.numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), constant_value_name
    )
    initializers[constant_value_name] = constant_value

    pad_node = onnx.helper.make_node(
        "Pad",
        inputs=[input_name, pads_name, constant_value_name],
        outputs=[output_name],
        mode="constant",
    )

    return pad_node
