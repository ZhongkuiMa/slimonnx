"""Enum types for the pattern_detect subpackage."""

__docformat__ = "restructuredtext"
__all__ = ["DetectorSig"]

from enum import StrEnum


class DetectorSig(StrEnum):
    """Detector function call signature, determining which arguments are passed.

    Detectors that need ``data_shapes`` should tolerate ``data_shapes=None``
    internally (return an empty match list) so the dispatcher can stay
    simple. The ``NS`` variant is the one exception: those detectors hard-
    require shapes and are guarded by the dispatcher.
    """

    N = "n"
    """Signature ``(nodes,)``."""
    NI = "ni"
    """Signature ``(nodes, initializers)``."""
    NIS = "nis"
    """Signature ``(nodes, initializers, data_shapes)``."""
    NS = "ns"
    """Signature ``(nodes, data_shapes)`` -- skipped when ``data_shapes`` is ``None``."""
