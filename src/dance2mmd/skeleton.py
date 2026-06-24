"""Canonical skeleton definition shared by every extractor backend.

This is the *contract* of the whole pipeline. Any pose-estimation backend
(MediaPipe on the Mac, WHAM / GVHMR on the GPU box, ...) only has to emit joint
positions for these names. Everything downstream (the VMD retargeter, the
three.js viewer) consumes this and nothing else, so backends are swappable.

Coordinate convention for the intermediate format:
    - Right-handed, **Y up**, **meters**.
    - Z is "forward" (toward the camera by default).
    - The root (CENTER) is placed at the pelvis.
Extractors are responsible for converting their native output into this frame.
"""

from __future__ import annotations

# Canonical joint names. Kept deliberately small: just what MMD bones need.
JOINTS = [
    "CENTER",      # pelvis / 下半身の起点 (root)
    "SPINE",       # 上半身
    "CHEST",       # 上半身2
    "NECK",        # 首
    "HEAD",        # 頭
    "SHOULDER_L", "ELBOW_L", "WRIST_L",
    "SHOULDER_R", "ELBOW_R", "WRIST_R",
    "HIP_L", "KNEE_L", "ANKLE_L", "TOE_L",
    "HIP_R", "KNEE_R", "ANKLE_R", "TOE_R",
]

JOINT_INDEX = {name: i for i, name in enumerate(JOINTS)}

# Parent of each joint, for drawing bones and computing rotations.
# CENTER is the root (parent = None).
PARENT = {
    "CENTER": None,
    "SPINE": "CENTER",
    "CHEST": "SPINE",
    "NECK": "CHEST",
    "HEAD": "NECK",
    "SHOULDER_L": "CHEST", "ELBOW_L": "SHOULDER_L", "WRIST_L": "ELBOW_L",
    "SHOULDER_R": "CHEST", "ELBOW_R": "SHOULDER_R", "WRIST_R": "ELBOW_R",
    "HIP_L": "CENTER", "KNEE_L": "HIP_L", "ANKLE_L": "KNEE_L", "TOE_L": "ANKLE_L",
    "HIP_R": "CENTER", "KNEE_R": "HIP_R", "ANKLE_R": "KNEE_R", "TOE_R": "ANKLE_R",
}

# Bone segments (parent -> child) for visualization.
BONES = [(p, c) for c, p in PARENT.items() if p is not None]
