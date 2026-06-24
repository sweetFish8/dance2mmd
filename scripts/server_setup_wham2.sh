#!/usr/bin/env bash
# stage 2: clone WHAM + python deps + pytorch3d prebuilt wheel (no nvcc needed).
set -euo pipefail
ROOT="${DANCE2MMD_ROOT:-$HOME}"
CONDA="$ROOT/miniconda3"
WHAM="$ROOT/dance2mmd/WHAM"
export CONDA_PKGS_DIRS="$ROOT/.conda_pkgs"
source "$CONDA/etc/profile.d/conda.sh"
conda activate wham

echo "[stage2] start $(date)"

# --- clone WHAM (+ submodules: DPVO etc.) ---
if [ ! -d "$WHAM/.git" ]; then
  echo "[stage2] cloning WHAM"
  git clone https://github.com/yohanshin/WHAM.git "$WHAM"
fi
cd "$WHAM"
git submodule update --init --recursive || echo "[stage2] (submodule step had warnings, continuing)"

# --- pytorch3d (prebuilt wheel matching py3.9 / cu118 / torch2.0.0) ---
if ! python -c "import pytorch3d" 2>/dev/null; then
  echo "[stage2] installing pytorch3d prebuilt wheel"
  pip install --no-cache-dir fvcore iopath
  pip install --no-cache-dir --no-index pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html \
    || echo "[stage2] WARN: prebuilt pytorch3d wheel failed; will need a fallback"
fi

# --- WHAM python requirements ---
if [ -f requirements.txt ]; then
  echo "[stage2] pip install -r requirements.txt"
  # let torch stay pinned; don't let requirements downgrade it
  pip install --no-cache-dir -r requirements.txt || echo "[stage2] WARN: some requirements failed (see log)"
fi

# common extra deps WHAM uses for the demo (detector/2d-pose). Harmless if already present.
pip install --no-cache-dir gdown ultralytics yacs joblib smplx loguru \
  || echo "[stage2] WARN: extra deps partial"

echo "[stage2] import smoke test:"
python - <<'PY'
mods = ["torch","pytorch3d","cv2","smplx","yacs","ultralytics"]
for m in mods:
    try:
        __import__(m); print("  OK", m)
    except Exception as e:
        print("  FAIL", m, "->", repr(e)[:120])
import torch; print("  cuda_avail", torch.cuda.is_available())
PY
echo "[stage2] DONE $(date)"
