"""Enum types for the pattern_detect subpackage."""

__docformat__ = "restructuredtext"
__all__ = ["DetectorSig"]

from enum import StrEnum


class DetectorSig(StrEnum):
    """Detector function call signature, determining which arguments are passed."""

    N = "n"
    """Signature ``(nodes,)``."""
    NI = "ni"
    """Signature ``(nodes, initializers)``."""
    NIS = "nis"
    """Signature ``(nodes, initializers, data_shapes)``."""
    NS = "ns"
    """Signature ``(nodes, data_shapes)`` — skipped when ``data_shapes`` is ``None``."""
    NS_INIT = "ns_init"
    """Signature ``(nodes, initializers, data_shapes)`` — skipped when ``data_shapes`` is ``None``."""
