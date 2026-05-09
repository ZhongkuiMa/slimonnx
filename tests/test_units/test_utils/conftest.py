"""Fixtures for utility function tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_npy_path():
    """Fixture providing a temporary .npy file path."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_npz_path():
    """Fixture providing a temporary .npz file path."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)
