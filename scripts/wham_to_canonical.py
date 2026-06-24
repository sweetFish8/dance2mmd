#!/usr/bin/env python
"""Convert WHAM demo output (wham_output.pkl) -> dance2mmd intermediate JSON.

Runs inside the 'wham' conda env on the GPU box (needs WHAM's SMPL + checkpoints
already set up). Computes world-frame SMPL 24-joint positions by running the
SMPL body model forward on WHAM's predicted world pose, then hands them to
dance2mmd.extract_wham.from_smpl_joints (same canonical contract as every backend).

    python scripts/wham_to_canonical.py \
        --pkl output/demo/<clip>/wham_output.pkl \
        --out output/demo/<clip>/<clip>.json --fps 30
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--wham_root", default=None, help="WHAM repo root (for imports)")
    ap.add_argument("--subject", type=int, default=None, help="track id; default=longest")
    args = ap.parse_args()

    wham_root = args.wham_root or os.getcwd()
    sys.path.insert(0, wham_root)

    import torch, joblib, smplx

    results = joblib.load(args.pkl)
    # pick the subject with the most frames (the main dancer)
    if args.subject is not None and args.subject in results:
        sid = args.subject
    else:
        sid = max(results.keys(), key=lambda k: len(results[k]["pose_world"]))
    r = results[sid]
    print(f"[convert] subject={sid}, frames={len(r['pose_world'])}, all_ids={list(results.keys())}")

    pose_world = np.asarray(r["pose_world"], np.float32)   # (F,72) axis-angle, Y-up world
    trans_world = np.asarray(r["trans_world"], np.float32)  # (F,3)
    betas = np.asarray(r["betas"], np.float32)
    F = pose_world.shape[0]
    if betas.ndim == 1:
        betas = np.tile(betas[None], (F, 1))
    betas = betas[:, :10]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use a *vanilla* smplx.SMPL (WHAM's SMPL subclass overrides forward() to take
    # rot6d, which doesn't match axis-angle input). Same SMPL_NEUTRAL.pkl.
    model_path = os.path.join(wham_root, "dataset", "body_models", "smpl")
    smpl = smplx.SMPL(model_path=model_path, gender="neutral",
                      num_betas=10, create_transl=False).to(device).eval()

    go = torch.tensor(pose_world[:, :3], device=device)
    bp = torch.tensor(pose_world[:, 3:72], device=device)
    bt = torch.tensor(betas, device=device)
    tr = torch.tensor(trans_world, device=device)

    joints = []
    step = 512
    with torch.no_grad():
        for s in range(0, F, step):
            e = min(s + step, F)
            out = smpl(global_orient=go[s:e], body_pose=bp[s:e],
                       betas=bt[s:e], transl=tr[s:e], pose2rot=True)
            joints.append(out.joints[:, :24, :].cpu().numpy())  # SMPL 24 body joints, world
    joints = np.concatenate(joints, 0).astype(np.float32)        # (F,24,3)
    print(f"[convert] joints {joints.shape}")

    # hand off to the shared adapter (Y-up world -> canonical JSON)
    try:
        from dance2mmd.extract_wham import from_smpl_joints
    except ImportError:
        # fallback: add the package src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from dance2mmd.extract_wham import from_smpl_joints

    tmp_npy = args.out + ".joints.npy"
    np.save(tmp_npy, joints)
    m = from_smpl_joints(tmp_npy, fps=args.fps, up="y",
                         source={"backend": "wham", "subject": int(sid), "pkl": args.pkl})
    m.to_json(args.out)
    os.remove(tmp_npy)
    print(f"[convert] wrote {args.out}: {m.num_frames} frames @ {args.fps} fps")

if __name__ == "__main__":
    main()
