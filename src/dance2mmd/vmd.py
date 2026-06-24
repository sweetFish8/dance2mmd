"""Minimal VMD (Vocaloid Motion Data) writer — bone frames only.

VMD layout (little-endian) we emit:
    char[30]  signature  "Vocaloid Motion Data 0002" + NUL padding
    char[20]  model name (shift_jis)
    uint32    bone keyframe count
    repeat:
        char[15]  bone name (shift_jis, NUL padded)
        uint32    frame index
        float[3]  position (x, y, z)        # MMD left-handed, Y-up
        float[4]  rotation quaternion (x, y, z, w)
        byte[64]  interpolation curve params
    uint32    morph count   (0)
    uint32    camera count  (0)
    uint32    light count   (0)
    uint32    self-shadow count (0)
    uint32    IK/show count  (0)

MMD bone names are Japanese and encoded shift_jis (see bones.py).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SIGNATURE = b"Vocaloid Motion Data 0002"

# Default linear-ish interpolation curve (the classic 20/20/107/107 bezier),
# repeated to fill the 64-byte block the way MMD lays it out.
_DEFAULT_INTERP = bytes(
    [20, 20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107]
) + bytes(
    [20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0]
) + bytes(
    [0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0]
) + bytes(
    [0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 0]
)
assert len(_DEFAULT_INTERP) == 64


@dataclass
class BoneFrame:
    name: str                 # MMD bone name (unicode; encoded on write)
    frame: int
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (0.0, 0.0, 0.0, 1.0)   # x, y, z, w


def _fixed(s: str, n: int) -> bytes:
    b = s.encode("shift_jis", errors="replace")[:n]
    return b + b"\x00" * (n - len(b))


def write_vmd(path: str, bone_frames: list[BoneFrame], model_name: str = "model") -> None:
    with open(path, "wb") as f:
        f.write(_fixed("Vocaloid Motion Data 0002", 30))
        f.write(_fixed(model_name, 20))
        f.write(struct.pack("<I", len(bone_frames)))
        for bf in bone_frames:
            f.write(_fixed(bf.name, 15))
            f.write(struct.pack("<I", int(bf.frame)))
            f.write(struct.pack("<3f", *map(float, bf.pos)))
            f.write(struct.pack("<4f", *map(float, bf.quat)))
            f.write(_DEFAULT_INTERP)
        # no morph / camera / light / shadow / IK keyframes
        f.write(struct.pack("<I", 0))  # morph
        f.write(struct.pack("<I", 0))  # camera
        f.write(struct.pack("<I", 0))  # light
        f.write(struct.pack("<I", 0))  # self shadow
        f.write(struct.pack("<I", 0))  # IK / model display


def read_vmd_header(path: str):
    """Tiny reader used by tests/sanity checks: returns (signature, model, count)."""
    with open(path, "rb") as f:
        sig = f.read(30).split(b"\x00")[0]
        model = f.read(20).split(b"\x00")[0].decode("shift_jis", errors="replace")
        (count,) = struct.unpack("<I", f.read(4))
    return sig, model, count
