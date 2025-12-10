# Setup and installation script
# Neural analysis dependencies

set -e

echo "=========================================================================="
echo "  Setup"
echo "=========================================================================="
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: $MACHINE"
echo ""

# Find suitable Python version
echo "Checking for compatible Python versions..."

# Check available Python versions
PYTHON_CMD=""

for version in python3.11 python3.10 python3.9 python3; do
    if command -v $version &> /dev/null; then
        PYTHON_VERSION=$($version --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$($version -c 'import sys; print(sys.version_info.major)')
        PYTHON_MINOR=$($version -c 'import sys; print(sys.version_info.minor)')
        
        # Check if version is compatible 3.9-3.13 (3.11 preferred)
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 13 ]; then
            PYTHON_CMD=$version
            echo "Found compatible Python: $version ($PYTHON_VERSION)"
            break
        else
            echo "Skipping $version ($PYTHON_VERSION) - not compatible"
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "ERROR: No compatible Python version found (need 3.9-3.13)"
    echo ""
    echo "Your Python 3.14 is too new for current PyTorch releases."
    echo ""
    echo "Solutions:"
    echo "1. Install Python 3.11 via Homebrew:"
    echo "   brew install python@3.11"
    echo ""
    echo "2. Use pyenv to manage multiple Python versions:"
    echo "   brew install pyenv"
    echo "   pyenv install 3.11.7"
    echo "   pyenv local 3.11.7"
    echo ""
    exit 1
fi

echo ""
echo "Using: $PYTHON_CMD (version $PYTHON_VERSION)"
echo ""

# Create virtual environment
echo "Creating virtual environment 'venv'..."
$PYTHON_CMD -m venv venv

echo "Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

echo "pip upgraded"
echo ""

# Check Python version in venv
VENV_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Virtual environment Python: $VENV_PYTHON_VERSION"
echo ""

# Determine PyTorch installation strategy based on Python version
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')

echo "Installing PyTorch..."
if [ "$MACHINE" = "Mac" ]; then
    echo "  Installing PyTorch for Mac..."
    
    # For Mac, install latest compatible version
    if [ "$PYTHON_MINOR" -le 11 ]; then
        # Python 3.11 or older - use specific version
        pip install torch==2.1.0 torchvision==0.16.0 -q
    else
        # Python 3.12+ - use latest
        pip install torch torchvision -q
    fi
else
    # Linux - check for GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  Installing PyTorch with CUDA support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    else
        echo "  Installing PyTorch (CPU only)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    fi
fi

echo "PyTorch installed"
echo ""

# Install other dependencies
echo "Installing scientific packages..."
pip install -q \
    allensdk \
    'numpy<2.0' \
    scipy \
    scikit-learn \
    pandas

echo "Scientific packages installed"
echo ""

echo "Installing visualization packages..."
pip install -q \
    matplotlib \
    seaborn \
    pillow \
    tqdm

echo "Visualization packages installed"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "
import sys
import torch
import torchvision
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy
import scipy
import sklearn
import pandas
import matplotlib
import seaborn
import PIL
import tqdm

print('All packages imported successfully')
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy: {numpy.__version__}')
print(f'  AllenSDK: OK')
"

echo ""
echo "=========================================================================="
echo "  INSTALLATION COMPLETE"
echo "=========================================================================="
echo ""
echo "Virtual environment created with: $PYTHON_CMD ($PYTHON_VERSION)"
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python base_neural_analysis.py"
echo "  3. Results in ./analysis_results/"
echo ""
echo "=========================================================================="
