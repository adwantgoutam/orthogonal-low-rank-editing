#!/bin/bash
# Setup script for M1/16GB with Python 3.11 and proper environment

set -e

echo "=========================================="
echo "Setting up Python 3.11 Environment"
echo "=========================================="
echo ""

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    brew install pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    source ~/.zshrc
fi

# Install Python 3.11.7
echo "Installing Python 3.11.7..."
pyenv install 3.11.7 --skip-existing

# Set local Python version
echo "Setting local Python to 3.11.7..."
cd "$(dirname "$0")"
pyenv local 3.11.7

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python version
echo ""
echo "Python version:"
python -V

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install -U pip

# Install PyTorch and dependencies
echo ""
echo "Installing PyTorch and dependencies..."
pip install torch torchvision torchaudio
pip install transformers accelerate datasets sentencepiece safetensors
pip install numpy scipy pandas tqdm matplotlib psutil seaborn requests

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'CPU available: {torch.cuda.is_available() == False}')
"

echo ""
echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To set BLAS thread limits (run before experiments):"
echo "  export VECLIB_MAXIMUM_THREADS=1"
echo "  export OMP_NUM_THREADS=1"
echo "  export MKL_NUM_THREADS=1"
echo "  export OPENBLAS_NUM_THREADS=1"
echo "  export TOKENIZERS_PARALLELISM=false"
echo ""

