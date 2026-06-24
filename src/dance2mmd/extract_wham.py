"""Backend B: high-accuracy GPU extraction (WHAM / GVHMR / any SMPL source).

The heavy model runs on the GPU box and dumps **SMPL 24-joint world positions**
as a .npy / .npz of shape (num_frames, 24, 3). This adapter maps those onto the
canonical skeleton and emits the identical intermediate Motion. That keeps the
GPU-specific, install-heavy code out of this package: only the small,
well-defined SMPL->canonical mapping lives here.

Run WHAM on the server first (see docs/GPU_SETUP.md), have it save joints as
`joints.npy`, then:

    dance2mmd extract dummy.mp4 --backend wham   # (cli wraps extract() below)
  or directly:
    from dance2mmd.extract_wham import from_smpl_joints
    m = from_smpl_joints("joints.npy", fps=30, up="y")

SMPL 24-joint order (standard):
  0 pelvis 1 L_hip 2 R_hip 3 spine1 4 L_knee 5 R_knee 6 spine2 7 L_ankle
  8 R_ankle 9 spine3 10 L_foot 11 R_foot 12 neck 13 L_collar 14 R_collar
  15 head 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow 20 L_wrist
  21 R_wrist 22 L_hand 23 R_hand
"""

from __future__ import annotations
import numpy as np
from .skeleton import JOINTS, JOINT_INDEX
from .motion import Motion

# canonical joint -> SMPL index
SMPL_MAP = {
    "CENTER": 0, "SPINE": 3, "CHEST": 9, "NECK": 12, "HEAD": 15,
    "SHOULDER_L": 16, "ELBOW_L": 18, "WRIST_L": 20,
    "SHOULDER_R": 17, "ELBOW_R": 19, "WRIST_R": 21,
    "HIP_L": 1, "KNEE_L": 4, "ANKLE_L": 7, "TOE_L": 10,
    "HIP_R": 2, "KNEE_R": 5, "ANKLE_R": 8, "TOE_R": 11,
}


def from_smpl_joints(npy_path: str, fps: float = 30.0, up: str = "y",
                     source: dict | None = None) -> Motion:
    """Build a Motion from SMPL 24-joint world positions.

    up: which axis is 'up' in the source data ('y' or 'z'). Many SMPL pipelines
        output Y-down or Z-up; if the figure looks upside-down / lying flat in
        the viewer, change this. The viewer is your calibration tool.
    """
    arr = np.load(npy_path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[arr.files[0]]
    arr = np.asarray(arr, np.float32)
    if arr.ndim != 3 or arr.shape[1] < 24 or arr.shape[2] != 3:
        raise ValueError(f"expected (frames, >=24, 3) SMPL joints; got {arr.shape}")

    # normalize to canonical Y-up, +Z forward.
    if up == "z":
        arr = arr[:, :, [0, 2, 1]]
        arr[:, :, 2] *= -1.0
    elif up == "-y":
        arr[:, :, 1] *= -1.0

    F = arr.shape[0]
    P = np.zeros((F, len(JOINTS), 3), np.float32)
    for name, smpl_i in SMPL_MAP.items():
        P[:, JOINT_INDEX[name]] = arr[:, smpl_i]

    src = {"backend": "wham", "smpl_joints": npy_path, "up": up}
    if source:
        src.update(source)
    return Motion(fps=fps, positions=P, source=src)


def extract(video_path: str) -> Motion:
    """CLI hook. We don't run WHAM in-process (it has its own heavy env); we
    expect a sibling `joints.npy` produced by the server-side run script."""
    import os
    cand = os.path.splitext(video_path)[0] + ".joints.npy"
    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"expected SMPL joints at {cand}. Run WHAM on the GPU box first "
            f"(docs/GPU_SETUP.md) and save joints there, or call "
            f"from_smpl_joints() directly.")
    return from_smpl_joints(cand, source={"video": video_path})
