"""
Assembles the final demo reel: a title card, a few annotated example clips
back-to-back, and a closing card with the eval numbers.

Usage:
    python -m demo.make_demo_video

Edit CLIPS below to point at your real clip directories + commands once
you have labeled data in place.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.track import track_clip
from viz.overlay import render_clip

OUT_PATH = Path(__file__).parent.parent / "outputs" / "demo_reel.mp4"
EVAL_RESULTS_PATH = Path(__file__).parent.parent / "outputs" / "eval_results.json"
FPS = 10
CARD_SECONDS = 2

# Edit this list to point at your real clip directories + the command for each.
# Keep this to curated, working examples -- the demo reel is meant to show
# the system doing its job, not to double as the failure-analysis section
# (that lives in eval/run_eval.py output and report/writeup.md instead).
CLIPS = [
    {"clip_dir": "data/clips/example_01", "command": "park behind the white van"},
    {"clip_dir": "data/clips/example_02", "command": "stop, there's a kid near the road"},
]


def _title_card(text: str, size=(1280, 720)) -> np.ndarray:
    w, h = size
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    lines = text.split("\n")
    y = h // 2 - (len(lines) * 20)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, line, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += 40
    return frame


def _load_frames(clip_dir: Path) -> list:
    paths = sorted(clip_dir.glob("frame_*.jpg"))
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def _closing_card_text() -> str:
    if not EVAL_RESULTS_PATH.exists():
        return "VLA Driver Copilot\n\ngithub.com/RomalaMishra/vla-driver-copilot"
    results = json.loads(EVAL_RESULTS_PATH.read_text())
    acc = results.get("overall_maneuver_accuracy", "n/a")
    return (
        f"VLA Driver Copilot\n\n"
        f"Maneuver accuracy: {acc}\n\n"
        f"github.com/RomalaMishra/vla-driver-copilot"
    )


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    size = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    def write_still(img, seconds):
        nonlocal writer, size
        if size is None:
            size = (img.shape[1], img.shape[0])
        for _ in range(seconds * FPS):
            writer.write(cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_RGB2BGR))

    # Determine frame size from the first real clip so the title card matches.
    first_clip = Path(__file__).parent.parent / CLIPS[0]["clip_dir"]
    sample_frames = _load_frames(first_clip)
    if not sample_frames:
        raise FileNotFoundError(
            f"No frames found in {first_clip} -- populate data/clips/ with real "
            f"extracted frames before generating the demo reel."
        )
    h, w = sample_frames[0].shape[:2]
    size = (w, h)
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, size)

    write_still(_title_card("VLA Driver Copilot\n\nDriver commands -> grounded actions", (w, h)), CARD_SECONDS)

    for clip_cfg in CLIPS:
        clip_dir = Path(__file__).parent.parent / clip_cfg["clip_dir"]
        frames = _load_frames(clip_dir)
        if not frames:
            print(f"[skip] no frames in {clip_dir}")
            continue
        results = track_clip(frames, clip_cfg["command"])
        tmp_path = str(OUT_PATH.parent / f"_tmp_{clip_dir.name}.mp4")
        render_clip(frames, clip_cfg["command"], results, tmp_path, fps=FPS)

        cap = cv2.VideoCapture(tmp_path)
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            writer.write(bgr)
        cap.release()
        Path(tmp_path).unlink(missing_ok=True)

    write_still(_title_card(_closing_card_text(), (w, h)), CARD_SECONDS)
    writer.release()
    print(f"Demo reel saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
