#!/bin/bash
# setup_runpod.sh — One-time setup on RunPod (2x GPU pod)
set -eo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Pre-MoE × SGLang PoC — RunPod Setup"
echo "============================================"

# 1. GPU check
echo "[1/6] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: Need at least 2 GPUs for TP=2. Found $GPU_COUNT."
    exit 1
fi

# 2. HF cache (avoid filling root disk)
export HF_HOME=${HF_HOME:-/workspace/huggingface_cache}
mkdir -p "$HF_HOME"
echo 'export HF_HOME=/workspace/huggingface_cache' >> ~/.bashrc 2>/dev/null || true
echo "[2/6] HF_HOME=$HF_HOME"

# 3. Install SGLang first (it pins its own torch + transformers versions)
echo "[3/6] Installing SGLang + dependencies..."
pip install "sglang[all]" -q
pip install numpy matplotlib ninja accelerate -q

# 4. Fix nvJitLink version mismatch BEFORE any torch import
#    pip-installed nvidia packages (12.8) conflict with system CUDA (12.4)
#    Solution: put pip's nvidia libs first in LD_LIBRARY_PATH
echo "[4/6] Fixing LD_LIBRARY_PATH..."
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
NVJITLINK_DIR="${SITE_PKGS}/nvidia/nvjitlink/lib"
if [ ! -d "$NVJITLINK_DIR" ]; then
    NVJITLINK_DIR=""
fi
TORCH_LIB="${SITE_PKGS}/torch/lib"
export LD_LIBRARY_PATH=$NVJITLINK_DIR:$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$NVJITLINK_DIR:$TORCH_LIB:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc 2>/dev/null || true
echo "  nvjitlink=$NVJITLINK_DIR"

# Verify torch loads
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# 5. Build C++ extension + install premoe package
echo "[5/6] Building C++ extension & installing premoe..."
rm -rf build/ *.egg-info pre_moe_cpp*.so
python setup.py build_ext --inplace
BUILD_EXT=0 pip install -e . --no-build-isolation

# 6. Verify
echo "[6/6] Verifying..."
python -c "
import pre_moe_cpp
funcs = [x for x in dir(pre_moe_cpp) if not x.startswith('_')]
print(f'  C++ OK: {funcs}')
assert len(funcs) >= 5, f'Expected >=5 functions, got {len(funcs)}: {funcs}'
"

python -c "
import torch, sglang
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs {torch.cuda.device_count()}')
print(f'  SGLang {sglang.__version__}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next steps:"
echo "    1. bash scripts/run_benchmark.sh extract train"
echo "    2. bash scripts/run_ttft_benchmark.sh"
echo "============================================"
