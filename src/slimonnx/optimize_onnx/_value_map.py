"""Track explicitly requested surviving node-output values through renaming."""

__docformat__ = "restructuredtext"
__all__ = ["get_value_map"]

import json
from collections import Counter

from onnx import ModelProto

_METADATA_KEY = "slimonnx.value_map.v1"


def _make_value_map(model: ModelProto, trace_values: tuple[str, ...]) -> dict[str, str | None]:
    """Validate requested source outputs before any graph mutation."""
    if not isinstance(trace_values, tuple) or any(
        not isinstance(name, str) or not name for name in trace_values
    ):
        raise TypeError("trace_values must be a tuple of non-empty output names")
    if len(set(trace_values)) != len(trace_values):
        raise ValueError("trace_values contains duplicate output names")
    if not trace_values:
        return {}
    counts = Counter(name for node in model.graph.node for name in node.output if name)
    for name in trace_values:
        if counts[name] != 1:
            raise ValueError(f"trace_values requires one source producer for {name!r}")
    return dict.fromkeys(trace_values)


def _store_value_map(model: ModelProto, value_map: dict[str, str | None]) -> None:
    """Store only this call's source-to-canonical mapping on the final model."""
    if value_map:
        entry = model.metadata_props.add()
        entry.key = _METADATA_KEY
        entry.value = json.dumps(value_map, sort_keys=True)


def _unique_map(pairs: list[tuple[str, str | None]]) -> dict[str, str | None]:
    """Reject duplicate JSON keys instead of silently taking the last value."""
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate value-map source")
    return dict(pairs)


def get_value_map(model: ModelProto) -> dict[str, str | None]:
    """Read source-output names mapped to canonical node-output names.

    Only explicitly requested values are represented. A missing source value
    after optimization maps to ``None``; so does a surviving internal name
    whose value was repurposed by fusion. Constants folded into initializers
    are not surviving node outputs. An untraced model returns an empty mapping.
    The map describes this one normalization call, not transitive history or
    the internal structure of a fused operator. It carries no numeric evidence.

    :param model: Slimmed model, including one reloaded from serialized bytes.
    :return: Independent source-to-canonical mapping.
    :raises ValueError: When metadata is malformed, ambiguous or disconnected.
    """
    entries = [entry.value for entry in model.metadata_props if entry.key == _METADATA_KEY]
    if not entries:
        return {}
    if len(entries) != 1:
        raise ValueError("ambiguous value-map metadata")
    result = json.loads(entries[0], object_pairs_hook=_unique_map)
    if not isinstance(result, dict) or any(
        not isinstance(source, str)
        or not source
        or (target is not None and (not isinstance(target, str) or not target))
        for source, target in result.items()
    ):
        raise ValueError("malformed value-map metadata")
    counts = Counter(name for node in model.graph.node for name in node.output if name)
    targets = [value for value in result.values() if value is not None]
    if len(set(targets)) != len(targets) or any(counts[value] != 1 for value in targets):
        raise ValueError("ambiguous or disconnected value-map target")
    return result
