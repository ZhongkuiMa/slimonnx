__docformat__ = "restructuredtext"
__all__ = ["_set_always_first_var"]

from onnx import NodeProto, TensorProto


def _set_always_first_var(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[NodeProto]:
    """
    Set the first variable of the model to be always the first variable.
    """
    count = 0
    for node in nodes:
        if node.op_type in {"Add", "Sub", "Mul", "Div", "MatMul"}:
            num_var = 0
            for input_name in node.input:
                if input_name in initializers:
                    # If the input is a constant, skip it.
                    continue
                num_var += 1
            if num_var != 1:
                continue

            input1 = node.input[0]
            input2 = node.input[1]
            if input1 in initializers and input2 not in initializers:
                # Swap the inputs.
                node.input[0] = input2
                node.input[1] = input1
                count += 1

    return nodes
