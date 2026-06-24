#!/usr/bin/env bash
# stage 4: resolve the chumpy/numpy clash, place our own SMPL models (no login),
# and download the login-free auxiliary data + checkpoints that
# fetch_demo_data.sh would have fetched. Idempotent.
set -uo pipefail
R="${DANCE2MMD_ROOT:-$HOME}"
WHAM="$R/dance2mmd/WHAM"
SMPLSRC="$R/dance2mmd/SMPL_unzip/SMPL_python_v.1.1.0/smpl/models"
export CONDA_PKGS_DIRS="$R/.conda_pkgs"
source "$R/miniconda3/etc/profile.d/conda.sh"
conda activate wham
cd "$WHAM"

echo "[stage4] start $(date)"

# --- 1. numpy<1.24 (chumpy needs np.bool/np.float aliases) + matching opencv + chumpy ---
NPV=$(python -c "import numpy;print(numpy.__version__)" 2>/dev/null || echo none)
if [ "$NPV" != "1.23.5" ]; then
  echo "[stage4] pinning numpy==1.23.5 (was $NPV) + opencv 4.9 + chumpy"
  pip install --no-cache-dir "numpy==1.23.5" "opencv-python==4.9.0.80" "chumpy==0.70"
else
  echo "[stage4] numpy already 1.23.5"
  python -c "import chumpy" 2>/dev/null || pip install --no-cache-dir "chumpy==0.70"
fi

# --- 2. auxiliary body_models (regressors, faces, mean params) — gdrive, no login ---
if [ ! -f dataset/body_models/J_regressor_wham.npy ]; then
  echo "[stage4] downloading aux body_models.tar.gz"
  gdown "https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t" -O dataset/body_models.tar.gz \
    && tar -xf dataset/body_models.tar.gz -C dataset/ && rm -f dataset/body_models.tar.gz \
    || echo "[stage4] WARN: aux body_models download/extract failed"
else
  echo "[stage4] aux body_models already present"
fi

# --- 3. place OUR SMPL models (after the tar extract, so not clobbered) ---
mkdir -p dataset/body_models/smpl
cp -f "$SMPLSRC/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" dataset/body_models/smpl/SMPL_NEUTRAL.pkl
cp -f "$SMPLSRC/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"        dataset/body_models/smpl/SMPL_FEMALE.pkl
cp -f "$SMPLSRC/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"        dataset/body_models/smpl/SMPL_MALE.pkl
echo "[stage4] SMPL models placed:"; ls -la dataset/body_models/smpl/

# --- 4. checkpoints (gdrive, no login) ---
mkdir -p checkpoints
declare -A CKPT=(
  ["wham_vit_w_3dpw.pth.tar"]="1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB"
  ["wham_vit_bedlam_w_3dpw.pth.tar"]="19qkI-a6xuwob9_RFNSPWf1yWErwVVlks"
  ["hmr2a.ckpt"]="1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y"
  ["dpvo.pth"]="1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT"
  ["yolov8x.pt"]="1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo"
  ["vitpose-h-multi-coco.pth"]="1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso"
)
for name in "${!CKPT[@]}"; do
  if [ -s "checkpoints/$name" ]; then echo "[stage4] have $name"; continue; fi
  echo "[stage4] gdown $name"
  gdown "https://drive.google.com/uc?id=${CKPT[$name]}&export=download&confirm=t" -O "checkpoints/$name" \
    || echo "[stage4] WARN: failed $name"
done

# --- 5. smoke test ---
echo "[stage4] smoke test:"
python - <<'PY'
import importlib, numpy as np, os
print("  numpy", np.__version__)
for m in ["torch","pytorch3d","cv2","smplx","chumpy","ultralytics","yacs"]:
    try: importlib.import_module(m); print("  OK", m)
    except Exception as e: print("  FAIL", m, "->", repr(e)[:140])
from configs import constants as _C
for f in [_C.BMODEL.JOINTS_REGRESSOR_WHAM, _C.BMODEL.FACES, _C.BMODEL.MEAN_PARAMS]:
    print(("  have " if os.path.exists(f) else "  MISSING "), f)
try:
    import smplx
    smplx.create(_C.BMODEL.FLDR, model_type="smpl", gender="neutral", num_betas=10)
    print("  SMPL load OK")
except Exception as e:
    print("  SMPL load FAIL ->", repr(e)[:200])
PY
echo "[stage4] DONE $(date)"
