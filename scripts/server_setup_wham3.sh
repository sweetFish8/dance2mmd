#!/usr/bin/env bash
# stage 3: download WHAM checkpoints + auxiliary demo data (detector/2D-pose
# weights). These are NOT license-gated (Google Drive). The SMPL *body models*
# ARE license-gated and must be placed by hand — see the printout at the end.
set -euo pipefail
ROOT="${DANCE2MMD_ROOT:-$HOME}"
CONDA="$ROOT/miniconda3"
WHAM="$ROOT/dance2mmd/WHAM"
export CONDA_PKGS_DIRS="$ROOT/.conda_pkgs"
source "$CONDA/etc/profile.d/conda.sh"
conda activate wham
cd "$WHAM"

echo "[stage3] start $(date)"

# WHAM ships a fetch script for demo weights/checkpoints.
if [ -f fetch_demo_data.sh ]; then
  echo "[stage3] running WHAM fetch_demo_data.sh"
  bash fetch_demo_data.sh || echo "[stage3] WARN: fetch_demo_data.sh had errors (see log)"
else
  echo "[stage3] no fetch_demo_data.sh in repo root; listing repo scripts:"
  ls *.sh 2>/dev/null || true
fi

echo "[stage3] checkpoints present:"
find . -maxdepth 3 \( -name '*.pth.tar' -o -name '*.pt' -o -name '*.ckpt' \) 2>/dev/null | head -20

echo "[stage3] SMPL body model status:"
ls -la dataset/body_models/smpl/ 2>/dev/null || echo "  (no SMPL files yet — MANUAL step required)"

cat <<'EOF'

============================================================
 MANUAL, ONE-TIME, LICENSE-GATED STEP (only you can do this)
============================================================
WHAM needs the SMPL neutral body model. Download requires a free account:
  1. Register/login: https://smpl.is.tue.mpg.de  (and https://smplify.is.tue.mpg.de)
  2. Download "SMPL_python_v.1.1.0.zip" (or the WHAM-specified SMPL_NEUTRAL.pkl).
  3. Place it as:  WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl
     (WHAM's README lists the exact filenames/paths it expects.)
Then we can run the demo on a real video.
============================================================
EOF
echo "[stage3] DONE $(date)"
