#!/bin/bash
# Twin Prime DGX Spark Setup
# Run this INSIDE the NGC PyTorch container

set -e

echo "=== Twin Prime Selection Bias - DGX Spark Setup ==="
echo ""

# Check if we're in a container
if [ ! -f /.dockerenv ]; then
    echo "ERROR: Run this inside the NGC PyTorch container."
    echo ""
    echo "First: ./run-container.sh"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Install dependencies (numpy already in NGC container)
echo "Installing dependencies..."
pip install scipy pandas matplotlib pyyaml numba
echo ""

# Quick GPU test with numba
echo "Testing Numba CUDA..."
python -c "
from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        out[idx] = x[idx] + y[idx]

n = 1000
x = np.ones(n)
y = np.ones(n)
out = np.zeros(n)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array(n)

add_kernel[10, 100](d_x, d_y, d_out)
result = d_out.copy_to_host()
print(f'Numba CUDA test: {result[0]} (expected 2.0)')
assert result[0] == 2.0, 'CUDA test failed!'
print('Numba CUDA: OK')
"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To run experiments:"
echo "  cd /workspace/project"
echo "  python run_gpu.py              # K=10^8"
echo "  python run_gpu.py --K 1e9      # K=10^9"
echo ""
