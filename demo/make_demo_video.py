"""
Assembles the demo reel: straight into real nuScenes footage, a running
driver <-> copilot exchange overlaid as the clip plays, no title cards or
branding -- just the pipeline working on real data.

The clip is split into a few segments, each with its own driver command or
question, so the video shows both question-answering ("what's that truck?")
and command-grounded action ("follow the white van ahead") back to back.

Usage:
    python -m demo.make_demo_video
"""

from pathlib import Path

import cv2

from pipeline.track import track_clip
from viz.overlay import _truncate_to_width

OUT_PATH = Path(__file__).parent.parent / "outputs" / "demo_reel.mp4"
CLIP_DIR = Path(__file__).parent.parent / "data" / "clips" / "scene-0061"
FPS = 10
HOLD = 5  # nuScenes keyframes are ~2Hz; hold each one 5x at 10fps output
          # so the text is readable instead of flashing by

# (start_frame, end_frame, driver line) -- mix of questions and commands,
# matched to what's actually visible in this scene at each point. The
# parked truck is only clearly in view for the first few frames before it
# scrolls out of frame, so that segment is short.
SEGMENTS = [
    (0, 4, "stay clear of the truck on the left"),
    (4, 22, "follow the white van ahead"),
    (22, 39, "is it safe to keep going straight?"),
]


def _load_frames() -> list:
    paths = sorted(CLIP_DIR.glob("frame_*.jpg"))
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def _wrap_text(text: str, max_width_px: int, font, scale: float, thickness: int) -> list:
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw > max_width_px and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _draw_chat_frame(frame, driver_line: str, result):
    out = frame.copy()
    h, w = out.shape[:2]

    if result.box is not None:
        x1, y1, x2, y2 = result.box
        color = (0, 220, 0) if result.grounding_score and result.grounding_score > 0.3 else (0, 0, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

    copilot_lines = _wrap_text(f"Copilot: {result.rationale}", w - 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[:2]
    panel_h = 40 + 24 * len(copilot_lines) + 34
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    driver_text = _truncate_to_width(f'Driver: "{driver_line}"', w - 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(out, driver_text, (10, h - panel_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1)

    y = h - panel_h + 46
    for line in copilot_lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y += 22

    cv2.putText(out, f"[{result.maneuver.upper()}]", (10, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return out


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = _load_frames()
    if not frames:
        raise FileNotFoundError(f"No frames in {CLIP_DIR} -- run data/extract_nuscenes.py first.")

    h, w = frames[0].shape[:2]
    all_frames = []

    for start, end, command in SEGMENTS:
        segment_frames = frames[start:end]
        results = track_clip(segment_frames, command)
        for frame, result in zip(segment_frames, results):
            annotated = _draw_chat_frame(frame, command, result)
            all_frames += [annotated] * HOLD

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (w, h))
    for frame in all_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"Demo reel saved -> {OUT_PATH} ({len(all_frames)} frames, {len(all_frames) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
