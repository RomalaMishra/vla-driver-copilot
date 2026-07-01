"""
CLI entry point: run the full pipeline on a single video clip + command,
save an annotated output video.

Usage:
    python scripts/run_example.py --clip data/clips/example_01 --command "park behind the white van"

The --clip directory should contain frames named frame_0000.jpg, frame_0001.jpg, ...
(sequential, in playback order).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from pipeline.track import track_clip
from viz.overlay import render_clip


def load_frames(clip_dir: Path) -> list:
    paths = sorted(clip_dir.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.jpg files found in {clip_dir}")
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def main():
    p = argparse.ArgumentParser(description="Run the VLA driver-command pipeline on a clip")
    p.add_argument("--clip", required=True, help="Directory of frame_*.jpg files")
    p.add_argument("--command", required=True, help="Driver command, e.g. 'park behind the white van'")
    p.add_argument("--out", default=None, help="Output video path (default: outputs/<clip_name>.mp4)")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    clip_dir = Path(args.clip)
    frames = load_frames(clip_dir)
    print(f"Loaded {len(frames)} frames from {clip_dir}")

    results = track_clip(frames, args.command)

    out_path = args.out or f"outputs/{clip_dir.name}.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    render_clip(frames, args.command, results, out_path, fps=args.fps)

    print(f"\nCommand: \"{args.command}\"")
    print(f"Decision: {results[0].maneuver} -- {results[0].rationale}")
    print(f"Saved annotated video -> {out_path}")


if __name__ == "__main__":
    main()
