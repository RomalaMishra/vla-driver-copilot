# VLA Driver Copilot: Grounding Natural-Language Driver Commands to Structured Driving Decisions

*Status: pilot results in, on a small hand-labeled set (n=5) -- see Limitations for what a larger eval would need.*

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

Two real driving clips (5 sec each, 10fps, sourced from free-license Pexels
footage -- see `data/clips/SOURCES.md`): `example_01`, a rural road under
heavy motion blur (kept deliberately, as a stress test), and `example_02`,
clear downtown driving with unambiguous vehicles/cyclists visible. nuScenes-mini
integration is planned (see Limitations) but not yet the eval source.

## Evaluation set

`eval/labeled_examples.json` -- 5 hand-labeled (frame, command) pairs, boxes
estimated by direct visual inspection of the raw frames (independent of any
model output), across three difficulty tiers:
- **easy** (n=2): single unambiguous referent
- **ambiguous** (n=1): multiple plausible candidates for the referent
- **adversarial** (n=2): the referent isn't visible, or isn't confidently
  identifiable even on close visual inspection -- correct behavior is
  `maneuver: "unknown"`, not a hallucinated answer

## Results

| Difficulty | n | Maneuver accuracy | Mean grounding IoU |
|---|---|---|---|
| easy | 2 | 1.00 | 0.43 |
| ambiguous | 1 | 1.00 | 0.30 |
| adversarial | 2 | **0.00** | n/a |
| **overall** | 5 | 0.60 | -- |

## Ablation: two-stage grounding vs. single-stage VLM bbox

| | Grounding IoU | Maneuver accuracy |
|---|---|---|
| Two-stage (VLM + GroundingDINO) | **0.25** | 0.67 |
| Single-stage (VLM outputs bbox directly) | **0.00** | 0.67 |

The single-stage VLM produced zero IoU overlap with ground truth on every
example with a gold box -- consistent with the known unreliability of
general VLMs at precise pixel coordinates, and a direct empirical
justification for the two-stage design rather than an assumed one.
Maneuver accuracy is identical between the two conditions, as expected --
that's the reasoning task, not the spatial one, and doesn't depend on which
component does the localizing.

## Failure analysis

**The headline finding is the adversarial-tier failure, not the accuracy
number.** Both adversarial examples got the maneuver wrong -- in both cases
the model confidently output `pull_over` for a command referencing an
object that was not present (`ex_adversarial_01`, no truck in frame at all)
or not confidently identifiable (`ex_adversarial_02`, too motion-blurred to
confirm). The system prompt explicitly instructs the model to output
`"unknown"` when uncertain rather than guess -- it did not do so in either
case. A system that never abstains is worse than one that does, and at n=2
this isn't decisive evidence, but it's a real, reproducible pattern worth
treating as the primary limitation of the current reasoning stage rather
than a footnote. A likely mitigation: lower-confidence outputs (the model
did report `confidence` below 0.9 on both adversarial cases even while
still committing to a maneuver) could be thresholded and overridden to
`"unknown"` post-hoc, rather than trusting the model's own decision to
abstain or not.

**Grounding IoU (0.30-0.44) is moderate, not tight**, on the tiers where a
box existed at all. Boxes are in approximately the right region but not
pixel-precise -- consistent with GroundingDINO being a general-purpose
open-vocabulary detector, not fine-tuned for driving scenes specifically.

**The ambiguous-tier example** (`ex_ambiguous_01`, two candidate cars) was
resolved correctly by defaulting to the nearer vehicle, matching the gold
label -- but n=1 here means this shouldn't be over-read; more ambiguous
examples are needed to say anything general about how the system resolves
multi-candidate references.

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
