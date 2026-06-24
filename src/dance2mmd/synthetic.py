"""Generate a synthetic Motion (no video needed) for testing the JSON ->
viewer -> VMD half of the pipeline before any real extraction backend is set up.

Produces a standing figure that waves both arms and bobs — enough to confirm
the viewer animates and the VMD has sane per-frame rotations.
"""

from __future__ import annotations
import numpy as np
from .skeleton import JOINTS, JOINT_INDEX
from .motion import Motion


def make(num_frames: int = 120, fps: float = 30.0) -> Motion:
    F = num_frames
    P = np.zeros((F, len(JOINTS), 3), np.float32)

    def setj(name, xyz_fn):
        idx = JOINT_INDEX[name]
        for f in range(F):
            P[f, idx] = xyz_fn(f / fps, f)

    bob = lambda t: 0.03 * np.sin(2 * np.pi * 0.8 * t)  # vertical bob

    setj("CENTER", lambda t, f: (0, 0.9 + bob(t), 0))
    setj("SPINE",  lambda t, f: (0, 1.05 + bob(t), 0))
    setj("CHEST",  lambda t, f: (0, 1.25 + bob(t), 0))
    setj("NECK",   lambda t, f: (0, 1.45 + bob(t), 0))
    setj("HEAD",   lambda t, f: (0, 1.58 + bob(t), 0))
    # hips
    setj("HIP_L",  lambda t, f: (0.10, 0.88 + bob(t), 0))
    setj("HIP_R",  lambda t, f: (-0.10, 0.88 + bob(t), 0))
    # legs straight down
    setj("KNEE_L", lambda t, f: (0.10, 0.50 + bob(t), 0))
    setj("KNEE_R", lambda t, f: (-0.10, 0.50 + bob(t), 0))
    setj("ANKLE_L", lambda t, f: (0.10, 0.08, 0))
    setj("ANKLE_R", lambda t, f: (-0.10, 0.08, 0))
    setj("TOE_L",  lambda t, f: (0.10, 0.02, 0.12))
    setj("TOE_R",  lambda t, f: (-0.10, 0.02, 0.12))
    # shoulders
    setj("SHOULDER_L", lambda t, f: (0.18, 1.40 + bob(t), 0))
    setj("SHOULDER_R", lambda t, f: (-0.18, 1.40 + bob(t), 0))

    # arms waving: elbow/wrist swing up & down
    def arm(side):
        s = 1 if side == "L" else -1
        sx = 0.18 * s
        def elbow(t, f):
            a = 0.6 * np.sin(2 * np.pi * 0.7 * t + (0 if side == "L" else np.pi))
            return (sx + s * 0.22 * np.cos(a), 1.40 + 0.22 * np.sin(a), 0)
        def wrist(t, f):
            a = 0.6 * np.sin(2 * np.pi * 0.7 * t + (0 if side == "L" else np.pi))
            return (sx + s * 0.44 * np.cos(a), 1.40 + 0.44 * np.sin(a), 0)
        setj(f"ELBOW_{side}", elbow)
        setj(f"WRIST_{side}", wrist)
    arm("L"); arm("R")

    return Motion(fps=fps, positions=P,
                  source={"backend": "synthetic", "frames": F})
