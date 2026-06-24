"""Intermediate Motion  ->  MMD bone rotations  ->  VMD.

This is the hand-written conversion logic (the part we deliberately do NOT hand
to an AI). It is a swing-only FK retargeter:

  for each driven MMD bone, take the canonical segment (parent_joint -> child_joint),
  compute the shortest-arc rotation that maps the bone's *rest* direction to the
  *observed* direction, build global rotations down the MMD hierarchy, then
  express each as a local rotation relative to its MMD parent and convert the
  coordinate handedness to MMD's left-handed Y-up frame.

Assumptions / known limitations (v1):
  - Target model is near a T-pose at rest (arms straight out along X). A real
    model in an A-stance needs `arm_rest_deg` tuned, or a per-bone offset.
  - Twist (e.g. forearm roll) is not recovered — swing only.
  - Legs are driven by FK (足/ひざ rotations), not foot-IK. Good enough for a
    first pass; foot-IK + ground contact is a planned upgrade (see docs).

The single coordinate-handedness flip lives in `_to_mmd_quat` and is the first
thing to re-check if a real MMD playback looks mirrored or inverted.
"""

from __future__ import annotations

import numpy as np

from . import quat
from .skeleton import JOINT_INDEX
from .motion import Motion
from .vmd import BoneFrame, write_vmd

# Each driven bone: MMD name -> (parent_joint, child_joint, rest_dir, mmd_parent_bone)
# rest_dir is the bone's direction in the MODEL's bind pose, in canonical Y-up
# space (+X = model's left, +Y = up, +Z = forward). Standard MMD models rest in
# an A-stance: arms angled ~35° below horizontal (NOT a T-pose), so the arm/elbow
# rests are set accordingly. mmd_parent_bone is the bone this one's rotation is
# expressed relative to (None => root/world).
_A = 0.819  # cos(35°)
_B = 0.574  # sin(35°)
_BONES = {
    "下半身":   ("SPINE", "CENTER", (0, -1, 0), None),
    "上半身":   ("SPINE", "CHEST", (0, 1, 0), None),
    "上半身2":  ("CHEST", "NECK", (0, 1, 0), "上半身"),
    # SMPL's head joint sits forward of the neck, so NECK->HEAD points up-and-
    # forward (~45°) even at rest. Match that as the bind direction, else the
    # head is permanently tilted down ("俯き").
    "首":      ("NECK", "HEAD", (0, 0.69, 0.72), "上半身2"),
    "左腕":    ("SHOULDER_L", "ELBOW_L", (_A, -_B, 0), "上半身2"),
    "左ひじ":  ("ELBOW_L", "WRIST_L", (_A, -_B, 0), "左腕"),
    "右腕":    ("SHOULDER_R", "ELBOW_R", (-_A, -_B, 0), "上半身2"),
    "右ひじ":  ("ELBOW_R", "WRIST_R", (-_A, -_B, 0), "右腕"),
    "左足":    ("HIP_L", "KNEE_L", (0, -1, 0), "下半身"),
    "左ひざ":  ("KNEE_L", "ANKLE_L", (0, -1, 0), "左足"),
    "右足":    ("HIP_R", "KNEE_R", (0, -1, 0), "下半身"),
    "右ひざ":  ("KNEE_R", "ANKLE_R", (0, -1, 0), "右足"),
}

# Order matters: a bone must come after its mmd parent so the parent's global
# rotation is already computed.
_ORDER = ["下半身", "上半身", "上半身2", "首",
          "左腕", "左ひじ", "右腕", "右ひじ",
          "左足", "左ひざ", "右足", "右ひざ"]


def _to_mmd_quat(q_canonical):
    """Convert a canonical right-handed (Y-up, +Z forward) quaternion to MMD's
    left-handed (Y-up, +Z into screen) frame.

    Handedness flip on the Z axis: (x, y, z, w) -> (-x, -y, z, w).
    *** If real MMD playback looks mirrored/inverted, this is the line to flip. ***
    """
    x, y, z, w = q_canonical
    return (-x, -y, z, w)


def retarget(motion: Motion, scale: float = 1.0) -> list[BoneFrame]:
    """Produce VMD bone keyframes (one set per frame) from the intermediate Motion.

    scale only affects the センター position track; rotations are scale-free.
    """
    P = motion.positions  # (F, J, 3)
    frames: list[BoneFrame] = []

    # rest directions are constant; observed directions come from positions.
    def seg_dir(f, a, b):
        return P[f, JOINT_INDEX[b]] - P[f, JOINT_INDEX[a]]

    # center + foot-IK position tracks, all relative to their frame-0 position.
    center0 = P[0, JOINT_INDEX["CENTER"]].copy()
    ankleL0 = P[0, JOINT_INDEX["ANKLE_L"]].copy()
    ankleR0 = P[0, JOINT_INDEX["ANKLE_R"]].copy()

    ident = np.array([0, 0, 0, 1.0])
    for f in range(motion.num_frames):
        global_rot = {}  # mmd bone name -> global quaternion (canonical frame)

        for name in _ORDER:
            pj, cj, rest, parent = _BONES[name]
            obs = seg_dir(f, pj, cj)
            parent_g = global_rot[parent] if parent in global_rot else ident
            # express the observed bone direction in the parent's CURRENT frame,
            # then take the minimal swing from the bone's bind direction. Doing
            # this per-joint in the parent frame keeps the bend correct and avoids
            # the spurious twist that independent world-space swings accumulate.
            obs_local = quat.rotate(quat.conj(parent_g), obs)
            local = quat.from_two_vectors(np.array(rest, float), obs_local)
            global_rot[name] = quat.mul(parent_g, local)
            frames.append(BoneFrame(name=name, frame=f,
                                    quat=_to_mmd_quat(quat.normalize(local))))

        # センター translation (canonical meters -> MMD units via scale).
        d = (P[f, JOINT_INDEX["CENTER"]] - center0) * scale
        frames.append(BoneFrame(name="センター", frame=f,
                                pos=(float(d[0]), float(d[1]), float(-d[2]))))  # +Z flip

        # foot-IK targets follow the ankles, so the legs actually step instead of
        # being pinned to bind pose by the IK solver. Same world-delta convention.
        for ik_name, j, j0 in (("左足ＩＫ", "ANKLE_L", ankleL0),
                               ("右足ＩＫ", "ANKLE_R", ankleR0)):
            di = (P[f, JOINT_INDEX[j]] - j0) * scale
            frames.append(BoneFrame(name=ik_name, frame=f,
                                    pos=(float(di[0]), float(di[1]), float(-di[2]))))

    return frames


def motion_to_vmd(motion: Motion, out_path: str, model_name: str = "model",
                  scale: float = 12.5) -> int:
    """Full convenience path: Motion -> .vmd file. Returns keyframe count.

    Default scale 12.5 maps ~1 m to ~12.5 MMD units (typical model is ~20 units
    tall ≈ 1.6 m). Tune per model.
    """
    frames = retarget(motion, scale=scale)
    write_vmd(out_path, frames, model_name=model_name)
    return len(frames)
