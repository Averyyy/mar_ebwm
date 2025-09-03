#!/bin/bash

# PyTorch 2.6.0 with CUDA 12.6 Installation Script
# This script will create conda environment and install PyTorch 2.6.0 with CUDA 12.6 support

set -e  # Exit on error

echo "========================================="
echo "PyTorch 2.6.0 with CUDA 12.6 Setup"
echo "========================================="

# Set environment name
ENV_NAME="ebm_gh200"

# Check if conda is installed
echo "Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "Conda version: $(conda --version)"

# Remove existing environment if it exists
echo "Checking if environment $ENV_NAME exists..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists, removing it..."
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

# Activate environment
echo "Activating environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Current environment: $(conda info --envs | grep '*')"

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch 2.6.0 with CUDA 12.6
echo "Installing PyTorch 2.6.0 with CUDA 12.6..."
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
echo "Installing other dependencies..."
python -m pip install einops==0.8.1
python -m pip install timm==1.0.15
python -m pip install wandb==0.19.9
python -m pip install tensorboard==2.19.0
python -m pip install safetensors==0.5.3
python -m pip install huggingface-hub==0.30.2
python -m pip install scipy==1.15.2
python -m pip install pillow==11.0.0
python -m pip install opencv-python==4.11.0.86
python -m pip install pyyaml==6.0.2
python -m pip install tqdm==4.67.1
python -m pip install numpy==2.1.2

# Install torch-fidelity from GitHub
echo "Installing torch-fidelity..."
python -m pip install git+https://github.com/LTH14/torch-fidelity.git@master

# Verify PyTorch CUDA installation
echo "========================================="
echo "Verifying PyTorch CUDA installation..."
echo "========================================="

python -c "
import torch
import torchvision

print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'Current GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Warning: CUDA is not available!')
"

echo "========================================="
echo "Installation completed!"
echo "========================================="

echo "Environment $ENV_NAME has been created and configured!"
echo ""
echo "To use this environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation, run after activating the environment:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""

# Display installed package versions
echo "Installed main package versions:"
python -m pip list | grep -E "(torch|torchvision|torchaudio|numpy|pillow|opencv|wandb)"

echo ""
echo "========================================="
echo "Environment setup completed! Use 'conda activate $ENV_NAME' to activate the environment"
echo "========================================"