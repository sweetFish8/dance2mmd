#!/usr/bin/env bash
# Run WHAM via the official Docker image (cuda11.3/py3.9/torch1.11, with ViTPose
# + DPVO + pytorch3d prebuilt). Use an Ampere-class GPU (e.g. RTX 3090): the
# image's cu11.3 runtime does NOT support Ada GPUs like the RTX 4090 (sm_89).
#
#   bash scripts/run_wham_docker.sh /path/to/video.mp4 [clip_name]
#
# The WHAM dir (checkpoints + SMPL already in place) is bind-mounted, so the
# image only supplies the Python env. Local-only (no DPVO/SLAM) for now.
set -euo pipefail
IMG="yusun9/wham-vitpose-dpvo-cuda11.3-python3.9"
ROOT="${DANCE2MMD_ROOT:-$HOME}/dance2mmd"
WHAM="$ROOT/WHAM"
# absolute path: cwd inside the container is $WHAM, so a relative path would double up
VIDEO="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
NAME="${2:-$(basename "${VIDEO%.*}")}"
STEM="$(basename "${VIDEO%.*}")"
# WHAM appends the video stem to --output_pth, so the real output dir is:
OUT="output/demo/$NAME/$STEM"

DOCKER="docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=$HOME \
  -v "$HOME":"$HOME" -w $WHAM $IMG"

echo "[docker] === WHAM inference (local-only) on $VIDEO ==="
$DOCKER python demo.py --video "$VIDEO" --output_pth "output/demo/$NAME" --save_pkl --estimate_local_only

echo "[docker] === convert -> canonical JSON ==="
FPS=$($DOCKER python -c "import cv2; c=cv2.VideoCapture('$VIDEO'); print(round(c.get(cv2.CAP_PROP_FPS) or 30,3))" 2>/dev/null | tr -d '\r' || echo 30)
$DOCKER python "$ROOT/scripts/wham_to_canonical.py" \
  --pkl "$WHAM/$OUT/wham_output.pkl" --out "$WHAM/$OUT/$NAME.json" --fps "$FPS" --wham_root "$WHAM"

echo "[docker] DONE -> $WHAM/$OUT/$NAME.json"
ls -la "$WHAM/$OUT"
