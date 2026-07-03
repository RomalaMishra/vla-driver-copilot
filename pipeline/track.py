"""
Multi-frame pipeline: runs decide() once per clip (the expensive VLM call),
then re-grounds the same target description on every subsequent frame
(cheap, local model only) to keep the box locked onto the target as the
scene plays out -- this is what makes the demo video feel live instead of
a single annotated still.

Re-grounding each frame independently means a naive top-1-by-score pick
will happily latch onto an unrelated object the moment the real target is
briefly occluded or leaves frame (GroundingDINO-tiny at a loose threshold
has no shortage of false positives to offer). _closest_detection below
adds the tracking-by-detection part that's usually implicit in "tracking":
prefer whichever candidate is actually near where the target last was,
and report "lost" rather than jump to a far-away high scorer.

Distance is tracked as a near/mid/far bin + frame-to-frame trend (see
perception/depth.py for why we don't claim precise metric distance).
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from perception import depth, grounding
from pipeline.decide import Decision, decide

# How far (as a fraction of the frame diagonal) a detection may be from the
# previous frame's box and still count as "the same object". Wide enough to
# tolerate real motion between sampled frames, tight enough to reject a
# same-class object elsewhere in the scene.
MAX_TRACK_JUMP_FRAC = 0.35


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


def _center(box: list) -> tuple:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _closest_detection(detections: list, prev_box: list | None, frame_shape: tuple):
    """
    Pick the detection that's actually the same object as last frame,
    instead of whichever scored highest this frame.

    With no prior box (first sighting), the top-scoring detection is the
    best guess. With a prior box, prefer the nearest candidate -- and if
    the nearest one is still farther than MAX_TRACK_JUMP_FRAC of the frame
    diagonal, treat the target as lost this frame rather than snapping the
    box onto a different object.
    """
    if not detections:
        return None, None
    if prev_box is None:
        best = detections[0]
        return best["box"], best["score"]

    h, w = frame_shape[:2]
    max_dist = MAX_TRACK_JUMP_FRAC * (h ** 2 + w ** 2) ** 0.5
    prev_c = _center(prev_box)

    best, best_dist = None, None
    for d in detections:
        c = _center(d["box"])
        dist = ((c[0] - prev_c[0]) ** 2 + (c[1] - prev_c[1]) ** 2) ** 0.5
        if dist <= max_dist and (best_dist is None or dist < best_dist):
            best, best_dist = d, dist

    if best is None:
        return None, None
    return best["box"], best["score"]


def track_clip_iter(frames: list, command: str, start_index: int = 0) -> Iterator[FrameResult]:
    """
    Same contract as track_clip, but yields each FrameResult as soon as
    it's computed instead of blocking until the whole clip is tracked.
    Frame start_index -- the only one that needs the hosted VLM call -- is
    ready almost immediately; callers that only care about the
    driver-facing answer (maneuver/rationale) don't need to wait for the
    rest.

    start_index lets the caller ground the decision in whatever frame was
    actually on screen when the driver spoke, rather than always frame 0
    -- a live clip is playing continuously, so "what should I do" has to
    be answered against *now*, not the start of the recording. Frames are
    then walked forward from there, wrapping around to cover the whole
    clip (matching the looping video_feed playback).
    """
    if not frames:
        return

    n = len(frames)
    start_index %= n
    order = [(start_index + k) % n for k in range(n)]

    initial: Decision = decide(frames[start_index], command)
    prev_bin = "unknown"
    prev_box = None

    for pos, i in enumerate(order):
        frame = frames[i]
        box, score = None, None

        if initial.target_description is None:
            # Command wasn't about a specific object (e.g. "is it safe to
            # change lanes") -- nothing to track, maneuver decision stands
            # for the whole clip.
            pass
        elif pos == 0:
            box, score = initial.box, initial.grounding_score
        else:
            detections = grounding.ground(frame, initial.target_description)
            box, score = _closest_detection(detections, prev_box, frame.shape)

        if box is not None:
            depth_map = depth.estimate(frame)
            curr_bin = depth.depth_bin(depth_map, box)
            prev_box = box
        else:
            curr_bin = "unknown"

        yield FrameResult(
            frame_index=i,
            box=box,
            grounding_score=score,
            distance_bin=curr_bin,
            distance_trend=depth.trend(prev_bin, curr_bin),
            maneuver=initial.maneuver,
            rationale=initial.rationale,
            confidence=initial.confidence,
        )
        prev_bin = curr_bin


def track_clip(frames: list, command: str) -> list:
    """
    Args:
        frames: list of (H, W, 3) uint8 RGB frames, in playback order
        command: the driver command issued at the start of the clip

    Returns:
        list of FrameResult, one per input frame, in playback order
        (frame_index order, not computation order).
    """
    return sorted(track_clip_iter(frames, command), key=lambda r: r.frame_index)
