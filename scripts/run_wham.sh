#!/usr/bin/env bash
# Run WHAM on one video and convert to the dance2mmd intermediate JSON.
# Usage: bash scripts/run_wham.sh /path/to/video.mp4 [clip_name]
# DPVO (global SLAM) isn't compiled, so we use --estimate_local_only.
set -euo pipefail
R="${DANCE2MMD_ROOT:-$HOME}"
WHAM="$R/dance2mmd/WHAM"
export CONDA_PKGS_DIRS="$R/.conda_pkgs"
source "$R/miniconda3/etc/profile.d/conda.sh"
conda activate wham
cd "$WHAM"

VIDEO="$1"
NAME="${2:-$(basename "${VIDEO%.*}")}"
OUT="output/demo/$NAME"
mkdir -p "$OUT"

echo "[run_wham] video=$VIDEO name=$NAME"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1

echo "[run_wham] === WHAM inference (local-only) ==="
python demo.py --video "$VIDEO" --output_pth "$OUT" --save_pkl --estimate_local_only

echo "[run_wham] === convert -> canonical JSON ==="
FPS=$(python -c "import cv2; c=cv2.VideoCapture('$VIDEO'); print(round(c.get(cv2.CAP_PROP_FPS) or 30, 3))" 2>/dev/null || echo 30)
python "$R/dance2mmd/scripts/wham_to_canonical.py" \
  --pkl "$OUT/wham_output.pkl" --out "$OUT/$NAME.json" --fps "$FPS" --wham_root "$WHAM"

echo "[run_wham] DONE -> $WHAM/$OUT/$NAME.json"
ls -la "$OUT"
