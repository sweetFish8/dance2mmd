#!/usr/bin/env bash
# stage 5: fix the two stage-4 failures without re-downloading checkpoints.
#   - numpy/opencv pin done separately from chumpy (combined install rolled back)
#   - chumpy needs --no-build-isolation (its setup.py does `import pip`)
#   - aux body_models download failed only because dataset/ didn't exist yet
set -uo pipefail
R="${DANCE2MMD_ROOT:-$HOME}"
WHAM="$R/dance2mmd/WHAM"
SMPLSRC="$R/dance2mmd/SMPL_unzip/SMPL_python_v.1.1.0/smpl/models"
export CONDA_PKGS_DIRS="$R/.conda_pkgs"
source "$R/miniconda3/etc/profile.d/conda.sh"
conda activate wham
cd "$WHAM"
echo "[stage5] start $(date)"

# 1. numpy<1.24 + matching opencv (chumpy excluded so a chumpy build error can't roll these back)
if [ "$(python -c 'import numpy;print(numpy.__version__)' 2>/dev/null)" != "1.23.5" ]; then
  echo "[stage5] pin numpy==1.23.5 + opencv 4.9"
  pip install --no-cache-dir "numpy==1.23.5" "opencv-python==4.9.0.80"
fi

# 2. chumpy without build isolation (so its setup.py can `import pip` from this env)
if ! python -c "import chumpy" 2>/dev/null; then
  echo "[stage5] install chumpy (--no-build-isolation)"
  pip install --no-cache-dir wheel setuptools
  pip install --no-cache-dir --no-build-isolation "chumpy==0.70" \
    || pip install --no-cache-dir --no-build-isolation "git+https://github.com/mattloper/chumpy.git"
fi

# 3. aux body_models (regressors/faces/mean params) — dir must exist for gdown -O
mkdir -p dataset/body_models
if [ ! -f dataset/body_models/J_regressor_wham.npy ]; then
  echo "[stage5] download aux body_models.tar.gz"
  gdown "https://drive.google.com/uc?id=1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ&export=download&confirm=t" -O dataset/body_models.tar.gz \
    && tar -xf dataset/body_models.tar.gz -C dataset/ && rm -f dataset/body_models.tar.gz
fi

# 4. (re)place our SMPL models in case the tar shipped its own body_models/
mkdir -p dataset/body_models/smpl
cp -f "$SMPLSRC/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" dataset/body_models/smpl/SMPL_NEUTRAL.pkl
cp -f "$SMPLSRC/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"        dataset/body_models/smpl/SMPL_FEMALE.pkl
cp -f "$SMPLSRC/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"        dataset/body_models/smpl/SMPL_MALE.pkl

echo "[stage5] body_models contents:"; ls dataset/body_models/ ; ls dataset/body_models/smpl/

echo "[stage5] smoke test:"
python - <<'PY'
import importlib, numpy as np, os
print("  numpy", np.__version__)
for m in ["torch","pytorch3d","cv2","smplx","chumpy","ultralytics","yacs"]:
    try: importlib.import_module(m); print("  OK", m)
    except Exception as e: print("  FAIL", m, "->", repr(e)[:140])
from configs import constants as _C
for f in [_C.BMODEL.JOINTS_REGRESSOR_WHAM, _C.BMODEL.FACES, _C.BMODEL.MEAN_PARAMS, _C.BMODEL.JOINTS_REGRESSOR_H36M]:
    print(("  have " if os.path.exists(f) else "  MISSING "), f)
try:
    import smplx
    smplx.create(_C.BMODEL.FLDR, model_type="smpl", gender="neutral", num_betas=10)
    print("  SMPL load OK")
except Exception as e:
    print("  SMPL load FAIL ->", repr(e)[:200])
PY
echo "[stage5] DONE $(date)"
