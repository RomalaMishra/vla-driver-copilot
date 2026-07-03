"""
Multi-frame pipeline: runs decide() once per clip (the expensive VLM call),
then re-grounds the same target description on every subsequent frame
(cheap, local model only) to keep the box locked onto the target as the
scene plays out -- this is what makes the demo video feel live instead of
a single annotated still.

Distance is tracked as a near/mid/far bin + frame-to-frame trend (see
perception/depth.py for why we don't claim precise metric distance).
"""

from dataclasses import dataclass

import numpy as np

from perception import depth, grounding
from pipeline.decide import Decision, decide


@dataclass
class FrameResult:
    frame_index: int
    box: list | None          # None if target not found / lost this frame
    grounding_score: float | None
    distance_bin: str          # "near" | "mid" | "far" | "unknown"
    distance_trend: str        # "closing" | "opening" | "steady"
    maneuver: str
    rationale: str
    confidence: float = 0.0


def track_clip(frames: list, command: str) -> list:
    """
    Args:
        frames: list of (H, W, 3) uint8 RGB frames, in playback order
        command: the driver command issued at the start of the clip

    Returns:
        list of FrameResult, one per input frame
    """
    if not frames:
        return []

    initial: Decision = decide(frames[0], command)
    results = []
    prev_bin = "unknown"

    for i, frame in enumerate(frames):
        box, score = None, None

        if initial.target_description is None:
            # Command wasn't about a specific object (e.g. "is it safe to
            # change lanes") -- nothing to track, maneuver decision stands
            # for the whole clip.
            pass
        elif i == 0:
            box, score = initial.box, initial.grounding_score
        else:
            detections = grounding.ground(frame, initial.target_description)
            if detections:
                box, score = detections[0]["box"], detections[0]["score"]

        if box is not None:
            depth_map = depth.estimate(frame)
            curr_bin = depth.depth_bin(depth_map, box)
        else:
            curr_bin = "unknown"

        results.append(FrameResult(
            frame_index=i,
            box=box,
            grounding_score=score,
            distance_bin=curr_bin,
            distance_trend=depth.trend(prev_bin, curr_bin),
            maneuver=initial.maneuver,
            rationale=initial.rationale,
            confidence=initial.confidence,
        ))
        prev_bin = curr_bin

    return results
