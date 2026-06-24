# GPU backend (WHAM)

High-accuracy extraction runs on a Linux host with an NVIDIA GPU. The client
machine is unchanged — the GPU host just produces a `joints.npy` (or the
canonical JSON directly), which `dance2mmd.extract_wham` consumes.

Two ways to run it. The Docker route is by far the easiest (no dependency hell).

Set a workspace dir on a roomy disk and point the scripts at it:

```bash
export DANCE2MMD_ROOT=$HOME            # or any path with plenty of free space
mkdir -p "$DANCE2MMD_ROOT/dance2mmd" && cd "$DANCE2MMD_ROOT/dance2mmd"
git clone <your-fork-url> .
```

## Option A — official WHAM Docker image (recommended)

WHAM ships a ready image with ViTPose + DPVO + pytorch3d already built:
`yusun9/wham-vitpose-dpvo-cuda11.3-python3.9`.

> **GPU note:** that image's CUDA 11.3 runtime supports Ampere (e.g. RTX 3090)
> but **not** Ada GPUs like the RTX 4090 (sm_89). On a 4090 you must build a
> CUDA 11.8+ environment yourself (Option B), which conflicts with WHAM's pinned
> ViTPose/mmcv — so prefer an Ampere card for the Docker route.

```bash
docker pull yusun9/wham-vitpose-dpvo-cuda11.3-python3.9
# clone WHAM into the workspace and drop in checkpoints + SMPL (see below)
git clone https://github.com/yohanshin/WHAM.git "$DANCE2MMD_ROOT/dance2mmd/WHAM"
# then:
bash scripts/run_wham_docker.sh /path/to/video.mp4 myclip
```

`run_wham_docker.sh` runs the WHAM demo (local-only, no SLAM) and converts the
output to the canonical JSON via `scripts/wham_to_canonical.py`.

## Option B — build the env yourself (CUDA 11.8+, for Ada GPUs)

`scripts/server_setup_wham{,2,3,4,5}.sh` build a Miniconda env (`wham`, py3.9,
torch 2.0 cu118) + WHAM + pytorch3d wheel + checkpoints, and resolve the
chumpy/numpy clash. Run them in order. Note: WHAM's ViTPose (mmcv==1.3.9) is
pinned to old torch and is painful on this stack — Option A is much smoother.

## Checkpoints & SMPL body model

WHAM's `fetch_demo_data.sh` downloads checkpoints + auxiliary data from Google
Drive (no login) and the SMPL neutral model from a registration-gated site.
The scripts here fetch the login-free parts automatically; the **SMPL body
model is a manual, license-gated download you must do yourself**:

1. Register at https://smpl.is.tue.mpg.de (free).
2. Download the SMPL python package and place the neutral model at
   `WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl`
   (the v1.0.0 or v1.1.0 neutral `.pkl` both work).

## Bring it back / finish locally

```bash
# copy the canonical JSON to the client and preview / convert
dance2mmd view  myclip.json          # CALIBRATE: figure upright & facing you?
dance2mmd retarget myclip.json -o myclip.vmd --model "YourModelName"
```

**Calibration:** if the figure is upside-down / lying flat in the viewer, change
`up=` in `from_smpl_joints()` (`y`, `z`, or `-y`). Get the canonical JSON right
*before* generating the VMD. The three.js viewer is the instrument.
