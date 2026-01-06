#!/bin/bash
set -e
echo "=== Creating Conda Environment (gptq-svd) ==="

conda create -n gptq-svd python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gptq-svd

echo "=== Installing Magma ==="
conda install -c conda-forge magma==2.7.2 -y

echo "=== Installing Pytorch 2.6.0 (CUDA 12.4) ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -e .

echo "=== Patching CuDNN for JAX ==="
pip install --upgrade nvidia-cudnn-cu12

CUDNN_PATH=$(python -c "import os, nvidia.cudnn; print(os.path.join(list(nvidia.cudnn.__path__)[0], 'lib'))")
echo "export LD_LIBRARY_PATH=$CUDNN_PATH:\$LD_LIBRARY_PATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

echo "=== Setting Safety Flags ==="
echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export TOKENIZERS_PARALLELISM=false" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "SUCCESS"
