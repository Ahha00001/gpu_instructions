#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Path to the `.py` file you want to run
PYTHON_SCRIPT_PATH="/home/hlcv_team023/gpu_instructions/"
# Path to the Python binary of the conda environment
CONDA_PYTHON_BINARY_PATH="/home/hlcv_team023/miniconda3/envs/hlcv/bin/python"
cd $PYTHON_SCRIPT_PATH
$CONDA_PYTHON_BINARY_PATH "$@"