#!/usr/bin/env bash
# Creates ./venv with everything neural_alignment.py and ood_test.py need.
# allensdk only supports Python 3.9 to 3.11, so one of those must be installed.
set -e

python=""
for candidate in python3.11 python3.10 python3.9; do
    if command -v "$candidate" >/dev/null 2>&1; then
        python="$candidate"
        break
    fi
done
if [ -z "$python" ]; then
    echo "Need Python 3.9, 3.10 or 3.11 for allensdk. On a Mac: brew install python@3.11"
    exit 1
fi
echo "Using $python ($($python --version))"

$python -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

if command -v nvidia-smi >/dev/null 2>&1; then
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    pip install -q torch torchvision
fi
pip install -q allensdk timm 'numpy<2.0' scipy scikit-learn scikit-image pandas matplotlib pillow tqdm

python -c "import torch, timm, allensdk, sklearn; print('ok: torch', torch.__version__, 'timm', timm.__version__)"
echo "Next: source venv/bin/activate && python neural_alignment.py"
