"""Pytest configuration and fixtures for slimonnx tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def baselines_dir():
    """Baselines directory for storing regression test data.

    :return: Path to baselines directory (created if needed)
    """
    path = Path(__file__).parent / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def results_dir():
    """Results directory for generated models.

    :return: Path to results directory (created if needed)
    """
    path = Path(__file__).parent / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path
