"""Tiny quaternion helpers (x, y, z, w order). No scipy dependency."""

from __future__ import annotations
import numpy as np


def normalize(q):
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def from_two_vectors(a, b):
    """Shortest-arc rotation that maps unit-ish vector a onto b (swing only)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a = a / na
    b = b / nb
    d = float(np.dot(a, b))
    if d > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -1.0 + 1e-9:
        # 180°: pick any axis orthogonal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + d])
    return normalize(q)


def rotate(q, v):
    """Rotate 3-vector v by quaternion q (x,y,z,w)."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ])


def slerp(q0, q1, t):
    q0, q1 = normalize(q0), normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    th0 = np.arccos(d)
    th = th0 * t
    q2 = normalize(q1 - q0 * d)
    return q0 * np.cos(th) + q2 * np.sin(th)
