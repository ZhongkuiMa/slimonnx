"""Fixtures for structure analysis tests."""

__docformat__ = "restructuredtext"

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_onnx_path():
    """Fixture providing a temporary ONNX file path."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_json_path():
    """Fixture providing a temporary JSON file path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name
    yield json_path
    Path(json_path).unlink(missing_ok=True)


@pytest.fixture
def temp_topo_path():
    """Fixture providing a temporary topology file path."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        topo_path = f.name
    yield topo_path
    Path(topo_path).unlink(missing_ok=True)
