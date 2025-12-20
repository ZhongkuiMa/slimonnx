#!/usr/bin/env python3
"""Build script to set up benchmarks symlink for slimonnx tests.

This script creates a symlink from slimonnx/test/vnncomp2024_benchmarks
to the actual benchmarks directory location.
"""

import sys
from pathlib import Path


def setup_benchmarks_symlink() -> int:
    """
    Set up symlink to benchmarks directory.

    :return: Exit code (0 for success, 1 for error)
    """
    # Get the tests directory (where this script is located)
    tests_dir = Path(__file__).parent

    # Target symlink path
    symlink_path = tests_dir / "vnncomp2024_benchmarks"

    # Find the benchmarks directory
    # Try ../../../vnncomp2024_benchmarks/benchmarks relative to tests dir
    benchmarks_dir = (
        tests_dir / ".." / ".." / ".." / "vnncomp2024_benchmarks" / "benchmarks"
    ).resolve()

    if not benchmarks_dir.exists():
        print(f"Error: Benchmarks directory not found at {benchmarks_dir}")
        print("Please ensure vnncomp2024_benchmarks is cloned in the parent directory")
        return 1

    # Check if symlink/file already exists
    if symlink_path.exists() or symlink_path.is_symlink():
        if symlink_path.is_symlink():
            current_target = symlink_path.resolve()
            if current_target == benchmarks_dir:
                print(f"Symlink already exists and points to correct location: {benchmarks_dir}")
                return 0
            print(f"Removing existing symlink pointing to {current_target}")
            symlink_path.unlink()
        else:
            print(f"Removing existing file/directory: {symlink_path}")
            if symlink_path.is_dir():
                symlink_path.rmdir()
            else:
                symlink_path.unlink()

    # Create the symlink
    try:
        symlink_path.symlink_to(benchmarks_dir)
        print(f"Created symlink: {symlink_path} -> {benchmarks_dir}")
        return 0
    except OSError as e:
        print(f"Error creating symlink: {e}")
        print(f"Attempted to create: {symlink_path} -> {benchmarks_dir}")
        return 1


if __name__ == "__main__":
    sys.exit(setup_benchmarks_symlink())
