"""Shared fixtures for optimize_onnx tests."""

__docformat__ = "restructuredtext"

import numpy as np
import pytest
from onnx import numpy_helper


def createmake_initializer(name, array):
    """Create a TensorProto initializer from numpy array."""
    return numpy_helper.from_array(array.astype(np.float32), name)


@pytest.fixture
def make_initializer():
    """Fixture providing initializer creation helper."""
    return createmake_initializer
