"""Backend A: MediaPipe Pose. Runs on the Mac (CPU), no GPU needed.

Use this for fast iteration / sanity-checking the choreography. It is lower
accuracy than the GPU backends but emits the exact same intermediate Motion,
so the rest of the pipeline doesn't care which backend produced the file.

MediaPipe world-landmark frame: origin at the hip center, x right, y DOWN,
z toward the camera. We convert to the canonical right-handed Y-up meters frame.
"""

from __future__ import annotations

import numpy as np

from .skeleton import JOINTS, JOINT_INDEX
from .motion import Motion

# MediaPipe Pose landmark indices we rely on.
MP = dict(
    NOSE=0,
    SHOULDER_L=11, SHOULDER_R=12,
    ELBOW_L=13, ELBOW_R=14,
    WRIST_L=15, WRIST_R=16,
    HIP_L=23, HIP_R=24,
    KNEE_L=25, KNEE_R=26,
    ANKLE_L=27, ANKLE_R=28,
    TOE_L=31, TOE_R=32,
)


def _canonical_frame(world, vis):
    """One frame: build the 19 canonical joints from 33 MediaPipe landmarks.

    `world` is (33,3) in MediaPipe's frame; `vis` is (33,) visibility.
    Returns (pos (19,3) Y-up meters, conf (19,)).
    """
    # Convert MediaPipe (x right, y down, z toward cam) -> canonical (x right, y up, z forward).
    w = world.copy()
    w[:, 1] *= -1.0   # y down -> y up
    w[:, 2] *= -1.0   # z toward cam -> z forward

    def g(name):
        return w[MP[name]]

    def cg(name):
        return vis[MP[name]]

    hip_mid = (g("HIP_L") + g("HIP_R")) * 0.5
    sh_mid = (g("SHOULDER_L") + g("SHOULDER_R")) * 0.5
    nose = g("NOSE")

    pos = np.zeros((len(JOINTS), 3), np.float32)
    conf = np.zeros((len(JOINTS),), np.float32)

    def put(name, p, c):
        pos[JOINT_INDEX[name]] = p
        conf[JOINT_INDEX[name]] = c

    # Spine chain (synthesized — MediaPipe has no torso joints).
    put("CENTER", hip_mid, min(cg("HIP_L"), cg("HIP_R")))
    put("CHEST", sh_mid, min(cg("SHOULDER_L"), cg("SHOULDER_R")))
    put("SPINE", hip_mid + (sh_mid - hip_mid) * 0.5, min(cg("HIP_L"), cg("SHOULDER_L")))
    put("NECK", sh_mid + (nose - sh_mid) * 0.3, cg("NOSE"))
    put("HEAD", nose, cg("NOSE"))

    # Direct mappings.
    for name in ("SHOULDER_L", "SHOULDER_R", "ELBOW_L", "ELBOW_R",
                 "WRIST_L", "WRIST_R", "HIP_L", "HIP_R",
                 "KNEE_L", "KNEE_R", "ANKLE_L", "ANKLE_R", "TOE_L", "TOE_R"):
        put(name, g(name), cg(name))

    return pos, conf


def extract(video_path: str, model_complexity: int = 2,
            min_detection_confidence: float = 0.5) -> Motion:
    """Run MediaPipe Pose over a video and return the intermediate Motion."""
    import cv2  # opencv-python
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )

    frames_pos, frames_conf = [], []
    last_pos, last_conf = None, None
    n_read = n_detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_world_landmarks:
            n_detected += 1
            lms = res.pose_world_landmarks.landmark
            world = np.array([[lm.x, lm.y, lm.z] for lm in lms], np.float32)
            vis = np.array([lm.visibility for lm in lms], np.float32)
            last_pos, last_conf = _canonical_frame(world, vis)
        elif last_pos is None:
            continue  # no detection yet; skip leading blank frames
        # hold last good pose on dropout (conf zeroed so retargeter can ignore)
        frames_pos.append(last_pos)
        frames_conf.append(last_conf if res.pose_world_landmarks else last_conf * 0.0)

    cap.release()
    pose.close()

    if not frames_pos:
        raise RuntimeError("no pose detected in any frame")

    return Motion(
        fps=fps,
        positions=np.stack(frames_pos),
        confidence=np.stack(frames_conf),
        source={"backend": "mediapipe",
                "video": video_path,
                "frames_read": n_read,
                "frames_detected": n_detected,
                "model_complexity": model_complexity},
    )
