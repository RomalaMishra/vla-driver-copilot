"""
Draws the HUD-style overlay (target box, distance readout, decision text)
on each frame and assembles the result into a real video file -- this is
what makes the demo look like a live perception/decision monitor running
on the camera feed, not a slideshow of annotated stills.
"""

import cv2
import numpy as np

_BOX_COLOR_OK = (0, 220, 0)
_BOX_COLOR_LOST = (0, 0, 220)
_TEXT_COLOR = (255, 255, 255)
_BG_COLOR = (30, 30, 30)


def draw_frame(frame: np.ndarray, command: str, result) -> np.ndarray:
    """
    Args:
        frame: (H, W, 3) uint8 RGB frame
        command: the driver command being acted on, shown as a caption
        result: pipeline.track.FrameResult for this frame

    Returns:
        (H, W, 3) uint8 RGB annotated frame
    """
    out = frame.copy()
    h, w = out.shape[:2]

    if result.box is not None:
        x1, y1, x2, y2 = result.box
        color = _BOX_COLOR_OK if result.grounding_score and result.grounding_score > 0.3 else _BOX_COLOR_LOST
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"{result.distance_bin} ({result.distance_trend})", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _draw_hud_panel(out, command, result)
    return out


def _draw_hud_panel(out: np.ndarray, command: str, result) -> None:
    h, w = out.shape[:2]
    panel_h = 70
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), _BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    cv2.putText(out, f'Command: "{command}"', (10, h - panel_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1)

    maneuver_text = f"Decision: {result.maneuver.upper()}"
    cv2.putText(out, maneuver_text, (10, h - panel_h + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.putText(out, result.rationale, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def render_clip(frames: list, command: str, results: list, out_path: str, fps: int = 10) -> str:
    """
    Args:
        frames: list of (H, W, 3) uint8 RGB frames
        command: driver command shown as caption
        results: list of pipeline.track.FrameResult, same length as frames
        out_path: output .mp4 path
        fps: output frame rate

    Returns:
        out_path
    """
    if not frames:
        raise ValueError("No frames to render")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame, result in zip(frames, results):
        annotated = draw_frame(frame, command, result)
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    writer.release()
    return out_path
