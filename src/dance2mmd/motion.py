"""The intermediate "bone" motion container + its JSON (de)serialization.

This is the editable middle layer. A backend produces a `Motion`; you can
inspect / smooth / trim the JSON by hand; then the retargeter or viewer reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .skeleton import JOINTS, JOINT_INDEX

FORMAT_VERSION = 1


@dataclass
class Motion:
    fps: float
    # positions: (num_frames, num_joints, 3), float meters, Y-up. Order == skeleton.JOINTS
    positions: np.ndarray
    # per-(frame,joint) confidence in [0,1], or None if the backend has none.
    confidence: Optional[np.ndarray] = None
    source: dict = field(default_factory=dict)  # free-form provenance (backend, video, etc.)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=np.float32)
        if self.positions.ndim != 3 or self.positions.shape[1] != len(JOINTS) \
                or self.positions.shape[2] != 3:
            raise ValueError(
                f"positions must be (frames, {len(JOINTS)}, 3); got {self.positions.shape}"
            )
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    def joint(self, name: str) -> np.ndarray:
        """Trajectory of one joint: (num_frames, 3)."""
        return self.positions[:, JOINT_INDEX[name], :]

    # --- IO -----------------------------------------------------------------
    def to_json(self, path: str) -> None:
        obj = {
            "format_version": FORMAT_VERSION,
            "fps": self.fps,
            "joints": JOINTS,
            "source": self.source,
            "positions": np.round(self.positions, 5).tolist(),
        }
        if self.confidence is not None:
            obj["confidence"] = np.round(self.confidence, 4).tolist()
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def from_json(cls, path: str) -> "Motion":
        with open(path) as f:
            obj = json.load(f)
        if obj.get("joints") != JOINTS:
            raise ValueError(
                "joint order in file does not match current skeleton.JOINTS; "
                "regenerate the intermediate file."
            )
        conf = obj.get("confidence")
        return cls(
            fps=obj["fps"],
            positions=np.asarray(obj["positions"], dtype=np.float32),
            confidence=np.asarray(conf, dtype=np.float32) if conf is not None else None,
            source=obj.get("source", {}),
        )
