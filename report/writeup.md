# VLA Driver Copilot: Grounding Natural-Language Driver Commands to Structured Driving Decisions

*Status: implementation complete, evaluation pending real labeled data -- fill in the TODO sections below once `eval/run_eval.py` and `eval/ablation.py` have run against real clips.*

## Problem

A human driver expresses intent informally and often ambiguously -- "pull
in behind that van," "is it safe to change lanes," "slow down, kid near
the road." Grounding that kind of natural-language expression in visual
context and converting it into a structured, actionable decision is
fundamentally a human-intent-understanding problem, with driving as the
application domain rather than the point itself. This project asks: can a
general-purpose vision-language model do this reliably, and where does it
fail?

## Approach

Two-stage pipeline: a hosted VLM (`reasoning/vlm_client.py`) interprets the
command against a frame and decides the maneuver + a target object
description; a dedicated open-vocabulary grounding model (GroundingDINO,
`perception/grounding.py`) localizes that description as a bounding box.
Distance is tracked via a local depth model across frames
(`perception/depth.py`) to support commands where proximity matters (e.g.
"stop near the van"). Full architecture diagram in `README.md`.

**Why two stages, not one big VLM call**: general VLMs are known to be
unreliable at precise spatial coordinates. Section "Ablation" below
quantifies this directly rather than assuming it.

## Dataset

TODO: nuScenes-mini scene IDs / clip sources used, once finalized. See
`data/README.md` for sourcing.

## Evaluation set

`eval/labeled_examples.json` -- hand-labeled (frame, command) pairs across
three difficulty tiers:
- **easy**: single unambiguous referent
- **ambiguous**: multiple plausible candidates for the referent
- **adversarial**: the referent isn't visible, or the command doesn't map
  to a clear maneuver -- correct behavior is `maneuver: "unknown"`, not a
  hallucinated answer

TODO: n = ___ examples, ___ easy / ___ ambiguous / ___ adversarial.

## Results

TODO -- run `python -m eval.run_eval` and paste the summary table here:

| Difficulty | n | Maneuver accuracy | Mean grounding IoU |
|---|---|---|---|
| easy | | | |
| ambiguous | | | |
| adversarial | | | |
| **overall** | | | |

## Ablation: two-stage grounding vs. single-stage VLM bbox

TODO -- run `python -m eval.ablation` and paste the result here:

| | Grounding IoU | Maneuver accuracy |
|---|---|---|
| Two-stage (VLM + GroundingDINO) | | |
| Single-stage (VLM outputs bbox directly) | | |

## Failure analysis

TODO once real results are in. Categorize failures, don't just report the
number that got them wrong:
- Ambiguous referent cases: did the model pick a plausible-but-wrong
  candidate, or correctly flag ambiguity?
- Adversarial cases: did the model correctly say "unknown," or hallucinate
  a confident answer for something not present? (This is arguably the most
  important number in the whole eval -- a system that never abstains is
  worse than one that does.)
- Occlusion / motion blur cases in the multi-frame tracking runs: does the
  box correctly show "lost" (red, per `viz/overlay.py`) rather than
  drifting onto the wrong object?

## Related work

- **Talk2Car** (Deruyttere et al.) -- grounding free-text driver commands
  to objects in real road scenes; closest task framing to this project.
- **DriveLM**, **NuScenes-QA** -- structured driving-scene VQA benchmarks.
- **RT-2, OpenVLA, PaLM-E** -- VLA systems built on pretrained VLM/LLM
  backbones for robot control, the same "adapt a pretrained backbone
  rather than train from scratch" approach used here.
- **nuScenes prediction challenge, Waymo Open Motion Dataset** -- open-loop
  motion/action prediction evaluated against logged data, the evaluation
  paradigm this project follows in the absence of a live simulator.

## Limitations and future work

- **Open-loop only**: no live vehicle or simulator; actions are predictions
  scored against ground truth, not executed. Closed-loop evaluation (e.g.
  in CARLA, using its built-in Traffic Manager as base driving competence)
  is the natural next step.
- **Relative depth, not calibrated distance**: `perception/depth.py`
  reports near/mid/far + trend rather than metric distance, since that
  needs known camera intrinsics (available per-clip in nuScenes, not in
  a generic dashcam clip).
- **Small, hand-labeled eval set**: a pilot-scale study, not a benchmark
  submission. Scaling the labeled set, or moving toward automatic labeling
  from a real driver's logged behavior (does the model's prediction match
  what a real driver actually did at that point in the clip), is a natural
  scaling path.
- **Single VLM call per clip**: the reasoning step runs once per clip and
  the target description is re-grounded every frame; it doesn't re-query
  the VLM if the scene changes meaningfully mid-clip (e.g. a new hazard
  appears). A trigger for re-invoking the reasoning stage is future work.
