"""
Scores the pipeline against eval/labeled_examples.json.

Usage:
    python -m eval.run_eval

Reports, per difficulty tier:
    - maneuver accuracy (exact match against gold_maneuver)
    - grounding IoU (against gold_box, when one exists)
    - "correctly abstained" rate for adversarial examples (gold_maneuver ==
      "unknown" and the model also said "unknown", instead of hallucinating)

This is deliberately not just a single aggregate accuracy number -- the
breakdown by difficulty tier is the point, since "how does it do on the
easy cases" and "does it know when it doesn't know" are different, both
important, questions.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.decide import decide

EXAMPLES_PATH = Path(__file__).parent / "labeled_examples.json"


def iou(box_a: list, box_b: list) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run() -> list:
    data = json.loads(EXAMPLES_PATH.read_text())
    results = []

    for ex in data["examples"]:
        frame_path = Path(__file__).parent.parent / ex["frame_path"]
        if not frame_path.exists():
            print(f"[skip] {ex['id']}: {frame_path} not found -- add real labeled frames first")
            continue

        frame = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        decision = decide(frame, ex["command"])

        maneuver_correct = decision.maneuver == ex["gold_maneuver"]
        box_iou = iou(decision.box, ex.get("gold_box")) if ex.get("gold_box") else None
        correctly_abstained = (
            ex["gold_maneuver"] == "unknown" and decision.maneuver == "unknown"
        )

        results.append({
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "command": ex["command"],
            "predicted_maneuver": decision.maneuver,
            "gold_maneuver": ex["gold_maneuver"],
            "maneuver_correct": maneuver_correct,
            "predicted_box": decision.box,
            "gold_box": ex.get("gold_box"),
            "iou": box_iou,
            "correctly_abstained": correctly_abstained if ex["gold_maneuver"] == "unknown" else None,
        })

    return results


def summarize(results: list) -> None:
    if not results:
        print("No results to summarize -- did you add real labeled examples?")
        return

    by_tier = {}
    for r in results:
        by_tier.setdefault(r["difficulty"], []).append(r)

    print(f"\n{'=' * 50}")
    print("  Evaluation summary")
    print(f"{'=' * 50}")
    for tier, rs in by_tier.items():
        acc = np.mean([r["maneuver_correct"] for r in rs])
        ious = [r["iou"] for r in rs if r["iou"] is not None]
        iou_str = f"{np.mean(ious):.2f}" if ious else "n/a"
        print(f"  [{tier:11s}] n={len(rs):3d} | maneuver_acc={acc:.2f} | mean_iou={iou_str}")

    overall_acc = np.mean([r["maneuver_correct"] for r in results])
    print(f"\n  Overall maneuver accuracy: {overall_acc:.2f}")
    print(f"{'=' * 50}\n")

    print("Per-example detail (for failure analysis):")
    for r in results:
        status = "OK" if r["maneuver_correct"] else "FAIL"
        print(f"  [{status}] {r['id']} ({r['difficulty']}): "
              f"predicted={r['predicted_maneuver']!r} gold={r['gold_maneuver']!r} "
              f"iou={r['iou']}")


if __name__ == "__main__":
    results = run()
    summarize(results)
