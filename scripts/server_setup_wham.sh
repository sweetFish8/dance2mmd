#!/usr/bin/env bash
# WHAM environment setup on a Linux GPU host. Everything lives under
# $DANCE2MMD_ROOT (default $HOME), so set it to a roomy disk. Idempotent: re-running skips
# finished steps. Runs in stages so we can poll progress.
#
#   stage 1: miniconda + conda env 'wham' (py3.9) + torch cu118  (this file)
#   stage 2: clone WHAM + python deps + pytorch3d wheel          (server_setup_wham2.sh)
#   stage 3: download WHAM checkpoints                           (server_setup_wham3.sh)
# SMPL body models are a manual, license-gated download (see docs/GPU_SETUP.md).
set -euo pipefail

ROOT="${DANCE2MMD_ROOT:-$HOME}"
CONDA="$ROOT/miniconda3"
LOG="$ROOT/dance2mmd/setup_stage1.log"
mkdir -p "$ROOT/dance2mmd"

echo "[stage1] start $(date)"

# --- miniconda ---
if [ ! -x "$CONDA/bin/conda" ]; then
  echo "[stage1] installing miniconda -> $CONDA"
  cd /tmp
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
  bash mc.sh -b -p "$CONDA"
  rm -f mc.sh
else
  echo "[stage1] miniconda already present"
fi

# keep the conda pkgs cache under the workspace (avoid filling host /)
export CONDA_PKGS_DIRS="$ROOT/.conda_pkgs"
source "$CONDA/etc/profile.d/conda.sh"

# accept Anaconda channel ToS (newer conda refuses to use defaults otherwise)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# --- env ---
if ! conda env list | grep -q "/envs/wham"; then
  echo "[stage1] creating env 'wham' (python 3.9)"
  conda create -y -n wham python=3.9
else
  echo "[stage1] env 'wham' already exists"
fi
conda activate wham

# --- torch (cu118 works on modern NVIDIA GPUs incl. Ada/Ampere; nvcc not needed) ---
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[stage1] installing torch 2.0.0 cu118"
  pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 \
    --index-url https://download.pytorch.org/whl/cu118
else
  echo "[stage1] torch+cuda already working"
fi

echo "[stage1] torch check:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
echo "[stage1] DONE $(date)"
