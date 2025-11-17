"""Script to collect ONNX models from VNNComp 2024 benchmark directories.

This script copies ONNX models from the VNNComp benchmarks directory into
a local benchmarks/ folder, organizing them into subdirectories by benchmark name.
"""

import os

from utils import (
    find_benchmarks_folders,
    find_onnx_folders,
    find_all_onnx_files,
)

if __name__ == "__main__":
    dir_name = "../../../vnncomp2024_benchmarks/benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collect {len(benchmark_dirs)} benchmark directories.")
    onnx_dirs = find_onnx_folders(benchmark_dirs)
    print(f"Collect {len(onnx_dirs)} ONNX directories.")

    # Create a folder in the current directory to store the collected ONNX models
    dir_name = "benchmarks/"
    os.makedirs(dir_name, exist_ok=True)
    # Create subfolders for each benchmark
    for bdir in benchmark_dirs:
        benchmark_name = os.path.basename(bdir)
        os.makedirs(os.path.join(dir_name, benchmark_name), exist_ok=True)

    max_onnx_per_benchmark = 20
    i = 0
    # Copy the collected ONNX models to the corresponding benchmark subfolders
    for onnx_dir in onnx_dirs:
        benchmark_name = os.path.basename(os.path.dirname(onnx_dir))
        target_dir = os.path.join(dir_name, benchmark_name)

        onnx_files = find_all_onnx_files([onnx_dir], num_limit=max_onnx_per_benchmark)
        for onnx_path in onnx_files:
            target_path = os.path.join(target_dir, os.path.basename(onnx_path))
            with open(onnx_path, "rb") as src_file:
                with open(target_path, "wb") as dst_file:
                    dst_file.write(src_file.read())
            i += 1

    print(f"Copied {i} ONNX models to {dir_name}")
