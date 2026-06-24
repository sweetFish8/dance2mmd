"""Command-line entry point.

    dance2mmd extract  video.mp4 -o motion.json [--backend mediapipe]
    dance2mmd retarget motion.json -o dance.vmd [--scale 12.5] [--model NAME]
    dance2mmd view     motion.json            # open the three.js preview
    dance2mmd info     motion.json

Backends:
    mediapipe  (default) — runs locally on the Mac, no GPU.
    wham                 — high accuracy; runs on the GPU box. See docs/GPU_SETUP.md.
                           (the wham adapter just needs to emit the same JSON.)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import webbrowser

from .motion import Motion
from .postprocess import smooth, recenter_ground


def _extract(args):
    if args.backend == "mediapipe":
        from .extract_mediapipe import extract
        m = extract(args.video, model_complexity=args.complexity)
    elif args.backend == "wham":
        from .extract_wham import extract  # GPU box
        m = extract(args.video)
    else:
        sys.exit(f"unknown backend: {args.backend}")

    if args.smooth > 1:
        m = smooth(m, args.smooth)
    if args.ground:
        m = recenter_ground(m)
    m.to_json(args.out)
    print(f"wrote {args.out}: {m.num_frames} frames @ {m.fps:.2f} fps "
          f"(backend={m.source.get('backend')})")


def _retarget(args):
    from .retarget import motion_to_vmd
    m = Motion.from_json(args.motion)
    n = motion_to_vmd(m, args.out, model_name=args.model, scale=args.scale)
    print(f"wrote {args.out}: {n} bone keyframes from {m.num_frames} frames")


def _view(args):
    here = os.path.dirname(os.path.abspath(__file__))
    viewer = os.path.normpath(os.path.join(here, "..", "..", "viewer", "index.html"))
    if not os.path.exists(viewer):
        sys.exit(f"viewer not found at {viewer}")
    url = f"file://{viewer}?motion=" + os.path.abspath(args.motion)
    print(f"opening {url}\n(if the file:// load is blocked, run:  "
          f"python -m http.server  in the project root and open "
          f"http://localhost:8000/viewer/index.html?motion=../{os.path.relpath(args.motion)})")
    webbrowser.open(url)


def _info(args):
    with open(args.motion) as f:
        obj = json.load(f)
    print(json.dumps({
        "fps": obj.get("fps"),
        "frames": len(obj.get("positions", [])),
        "joints": len(obj.get("joints", [])),
        "source": obj.get("source", {}),
    }, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="dance2mmd")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="video -> intermediate bone JSON")
    e.add_argument("video")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--backend", default="mediapipe")
    e.add_argument("--complexity", type=int, default=2, help="mediapipe model_complexity 0/1/2")
    e.add_argument("--smooth", type=int, default=5, help="moving-avg window (1=off)")
    e.add_argument("--ground", action="store_true", help="drop feet to y=0")
    e.set_defaults(func=_extract)

    r = sub.add_parser("retarget", help="intermediate JSON -> .vmd")
    r.add_argument("motion")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--scale", type=float, default=12.5)
    r.add_argument("--model", default="model")
    r.set_defaults(func=_retarget)

    v = sub.add_parser("view", help="open the three.js preview of the bone JSON")
    v.add_argument("motion")
    v.set_defaults(func=_view)

    i = sub.add_parser("info", help="print summary of a bone JSON")
    i.add_argument("motion")
    i.set_defaults(func=_info)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
