"""Cleanup applied to the intermediate Motion before retargeting.

Monocular pose estimate is jittery; a light temporal smooth makes the VMD far
less twitchy. Kept separate from extraction so it applies to *any* backend.
"""

from __future__ import annotations
import numpy as np
from .motion import Motion


def smooth(motion: Motion, window: int = 5) -> Motion:
    """Centered moving-average over positions. window must be odd."""
    if window <= 1:
        return motion
    if window % 2 == 0:
        window += 1
    F = motion.num_frames
    pad = window // 2
    P = motion.positions
    padded = np.concatenate([P[:1].repeat(pad, 0), P, P[-1:].repeat(pad, 0)], axis=0)
    kernel = np.ones(window, np.float32) / window
    out = np.empty_like(P)
    for j in range(P.shape[1]):
        for c in range(3):
            out[:, j, c] = np.convolve(padded[:, j, c], kernel, mode="valid")
    return Motion(fps=motion.fps, positions=out,
                  confidence=motion.confidence, source={**motion.source, "smoothed": window})


def recenter_ground(motion: Motion) -> Motion:
    """Shift so the lowest foot over the whole clip sits at y=0 (feet on floor)."""
    from .skeleton import JOINT_INDEX
    P = motion.positions.copy()
    feet = np.concatenate([P[:, JOINT_INDEX["ANKLE_L"], 1],
                           P[:, JOINT_INDEX["ANKLE_R"], 1]])
    P[:, :, 1] -= float(np.min(feet))
    return Motion(fps=motion.fps, positions=P,
                  confidence=motion.confidence, source=motion.source)
