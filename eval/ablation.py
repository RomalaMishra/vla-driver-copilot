"""
Ablation: two-stage pipeline (VLM decides intent -> GroundingDINO localizes)
vs. single-stage (VLM asked to output the bounding box itself directly).

This exists to answer, with evidence rather than assertion, the natural
question "isn't the grounding model doing the real work?" -- it quantifies
how much a dedicated open-vocabulary detector actually buys you over just
asking a general VLM for coordinates.

Usage:
    python -m eval.ablation
"""

import json
from pathlib import Path

import cv2
import numpy as np

from eval.run_eval import iou
from pipeline.decide import decide
from reasoning.vlm_client import interpret_single_stage

EXAMPLES_PATH = Path(__file__).parent / "labeled_examples.json"


def run() -> dict:
    data = json.loads(EXAMPLES_PATH.read_text())
    two_stage_ious, single_stage_ious = [], []
    two_stage_acc, single_stage_acc = [], []

    for ex in data["examples"]:
        frame_path = Path(__file__).parent.parent / ex["frame_path"]
        if not frame_path.exists() or not ex.get("gold_box"):
            continue

        frame = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)

        two_stage = decide(frame, ex["command"])
        two_stage_ious.append(iou(two_stage.box, ex["gold_box"]))
        two_stage_acc.append(two_stage.maneuver == ex["gold_maneuver"])

        single = interpret_single_stage(frame, ex["command"])
        single_stage_ious.append(iou(single.get("box"), ex["gold_box"]))
        single_stage_acc.append(single.get("maneuver") == ex["gold_maneuver"])

    return {
        "n": len(two_stage_ious),
        "two_stage_mean_iou": float(np.mean(two_stage_ious)) if two_stage_ious else None,
        "single_stage_mean_iou": float(np.mean(single_stage_ious)) if single_stage_ious else None,
        "two_stage_maneuver_acc": float(np.mean(two_stage_acc)) if two_stage_acc else None,
        "single_stage_maneuver_acc": float(np.mean(single_stage_acc)) if single_stage_acc else None,
    }


if __name__ == "__main__":
    result = run()
    print(f"\n{'=' * 50}")
    print("  Ablation: two-stage grounding vs. single-stage VLM bbox")
    print(f"{'=' * 50}")
    if result["n"] == 0:
        print("  No examples with gold_box found -- add real labeled frames first.")
    else:
        print(f"  n = {result['n']} examples with ground-truth boxes\n")
        print(f"  Grounding IoU     -- two-stage: {result['two_stage_mean_iou']:.2f}  |  "
              f"single-stage: {result['single_stage_mean_iou']:.2f}")
        print(f"  Maneuver accuracy -- two-stage: {result['two_stage_maneuver_acc']:.2f}  |  "
              f"single-stage: {result['single_stage_maneuver_acc']:.2f}")
    print(f"{'=' * 50}\n")
