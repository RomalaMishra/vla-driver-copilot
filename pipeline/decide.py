"""
Single-frame decision pipeline: (frame, driver command) -> structured action.

This is the two-stage architecture in one place:
    1. reasoning.vlm_client.interpret()  -- VLM decides intent + maneuver
    2. perception.grounding.ground()     -- localizes the VLM's target description

See eval/ablation.py for the single-stage comparison (VLM asked to output
the bounding box itself) that this design choice is evaluated against.
"""

from dataclasses import dataclass, field

import numpy as np

from perception import grounding
from reasoning import vlm_client


@dataclass
class Decision:
    maneuver: str
    rationale: str
    confidence: float
    target_description: str | None
    box: list | None          # [x1, y1, x2, y2] or None if not grounded
    grounding_score: float | None
    raw_vlm_output: dict = field(default_factory=dict)


def decide(frame: np.ndarray, command: str) -> Decision:
    """
    Args:
        frame: (H, W, 3) uint8 RGB image
        command: natural-language driver command

    Returns:
        Decision -- maneuver is "unknown" if the VLM was not confident,
        box is None if there was no target to ground or grounding failed
        to find it (both are legitimate outcomes, not errors -- a system
        that never says "I don't know" is worse than one that does).
    """
    intent = vlm_client.interpret(frame, command)

    box, score = None, None
    target = intent.get("target_description")
    if target:
        detections = grounding.ground(frame, target)
        if detections:
            box, score = detections[0]["box"], detections[0]["score"]

    return Decision(
        maneuver=intent.get("maneuver", "unknown"),
        rationale=intent.get("rationale", ""),
        confidence=float(intent.get("confidence", 0.0)),
        target_description=target,
        box=box,
        grounding_score=score,
        raw_vlm_output=intent,
    )
