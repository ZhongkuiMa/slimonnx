"""Fixtures for structure analysis tests."""

from pathlib import Path

import pytest


@pytest.fixture
def report_path(tmp_path: Path) -> Path:
    """Provide a temporary path for a JSON report file."""
    return tmp_path / "report.json"
