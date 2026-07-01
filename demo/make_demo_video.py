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
from viz.overlay import _truncate_to_width, draw_frame

OUT_PATH = Path(__file__).parent.parent / "outputs" / "demo_reel.mp4"
EVAL_RESULTS_PATH = Path(__file__).parent.parent / "outputs" / "eval_results.json"
FPS = 10
CARD_SECONDS = 2

# Edit this list to point at your real clip directories + the command for each.
# Keep this to curated, working examples -- the demo reel is meant to show
# the system doing its job, not to double as the failure-analysis section
# (that lives in eval/run_eval.py output and report/writeup.md instead).
CLIPS = [
    {"clip_dir": "data/clips/example_02", "command": "stop behind the white car ahead"},
    {"clip_dir": "data/clips/example_01", "command": "turn left at the upcoming junction"},
]


def _title_card(lines: list, size: tuple) -> np.ndarray:
    w, h = size
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    y = h // 2 - (len(lines) * 20)
    for line in lines:
        line = _truncate_to_width(line, w - 40, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, line, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += 40
    return frame


def _load_frames(clip_dir: Path) -> list:
    paths = sorted(clip_dir.glob("frame_*.jpg"))
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def _closing_card_lines() -> list:
    if not EVAL_RESULTS_PATH.exists():
        return ["VLA Driver Copilot", "", "github.com/RomalaMishra/", "vla-driver-copilot"]
    results = json.loads(EVAL_RESULTS_PATH.read_text())
    acc = results.get("overall_maneuver_accuracy", "n/a")
    return [
        "VLA Driver Copilot", "",
        f"Maneuver accuracy: {acc}", "",
        "github.com/RomalaMishra/", "vla-driver-copilot",
    ]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    first_clip = Path(__file__).parent.parent / CLIPS[0]["clip_dir"]
    sample_frames = _load_frames(first_clip)
    if not sample_frames:
        raise FileNotFoundError(
            f"No frames found in {first_clip} -- populate data/clips/ with real "
            f"extracted frames before generating the demo reel."
        )
    h, w = sample_frames[0].shape[:2]
    size = (w, h)

    # Build the whole reel as one list of RGB frames in memory, then write
    # once -- writing per-clip temp files and re-reading them via
    # cv2.VideoCapture silently dropped frames on this system (mp4v codec
    # round-trip issue), so this avoids that entirely.
    all_frames = []

    title = _title_card(["VLA Driver Copilot", "", "Driver commands -> grounded actions"], size)
    all_frames += [title] * (CARD_SECONDS * FPS)

    for clip_cfg in CLIPS:
        clip_dir = Path(__file__).parent.parent / clip_cfg["clip_dir"]
        frames = _load_frames(clip_dir)
        if not frames:
            print(f"[skip] no frames in {clip_dir}")
            continue
        results = track_clip(frames, clip_cfg["command"])
        for frame, result in zip(frames, results):
            annotated = draw_frame(frame, clip_cfg["command"], result)
            all_frames.append(cv2.resize(annotated, size))

    closing = _title_card(_closing_card_lines(), size)
    all_frames += [closing] * (CARD_SECONDS * FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, size)
    for frame in all_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"Demo reel saved -> {OUT_PATH} ({len(all_frames)} frames, {len(all_frames) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
