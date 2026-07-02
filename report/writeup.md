# VLA Driver Copilot: Grounding Natural-Language Driver Commands to Structured Driving Decisions

*Status: pilot results in, on a small hand-labeled set (n=8, now including real nuScenes-mini data) -- see Limitations for what a larger eval would need.*

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

Three sources, kept deliberately varied rather than one clean set:
- `example_01` -- rural road, free-license Pexels footage, heavy motion blur
  (kept as a stress test, not a demo clip)
- `example_02` -- clear downtown driving, free-license Pexels footage
- `scene-0061` -- real **nuScenes-mini** scene ("Parked truck, construction,
  intersection, turn left, following a van"), extracted via
  `data/extract_nuscenes.py`

See `data/clips/SOURCES.md` for provenance.

## Evaluation set

`eval/labeled_examples.json` -- 8 hand-labeled (frame, command) pairs, boxes
estimated by direct visual inspection of the raw frames (independent of any
model output), across three difficulty tiers:
- **easy** (n=4): single unambiguous referent
- **ambiguous** (n=1): multiple plausible candidates for the referent
- **adversarial** (n=3): the referent isn't visible, or isn't confidently
  identifiable even on close visual inspection -- correct behavior is
  `maneuver: "unknown"`, not a hallucinated answer

## Results

| Difficulty | n | Maneuver accuracy | Mean grounding IoU |
|---|---|---|---|
| easy | 4 | 0.50 | 0.42 |
| ambiguous | 1 | 1.00 | 0.30 |
| adversarial | 3 | **0.00** | n/a |
| **overall** | 8 | 0.38 | -- |

Note: the first pilot pass (n=5) showed 0.60 overall accuracy; expanding to
n=8 (adding 3 more real examples, including nuScenes) dropped that to 0.38.
That drop is itself worth reporting rather than only keeping the more
flattering early number -- a 5-example eval was simply too small to trust.

## Ablation: two-stage grounding vs. single-stage VLM bbox

| | Grounding IoU | Maneuver accuracy |
|---|---|---|
| Two-stage (VLM + GroundingDINO) | **0.47** | **0.60** |
| Single-stage (VLM outputs bbox directly) | **0.02** | 0.40 |

The single-stage VLM produced essentially zero IoU overlap with ground
truth -- consistent with the known unreliability of general VLMs at precise
pixel coordinates, and a direct empirical justification for the two-stage
design rather than an assumed one. Maneuver accuracy is *also* lower
single-stage (0.40 vs 0.60), which wasn't predicted going in: asking one
prompt to both reason about intent and produce precise coordinates appears
to cost some reasoning quality too, not just localization precision --
plausibly a divided-attention effect, though n=5 is too small to be certain
that's the mechanism rather than noise.

## Failure analysis

**The headline finding is the adversarial-tier failure, and it's now a
robust pattern, not a coincidence.** All three adversarial examples failed
-- the model confidently produced a maneuver (`pull_over`, `pull_over`,
`turn_right`) for commands referencing objects or infrastructure that
weren't present (`ex_adversarial_01`: no truck in frame; `ex_adversarial_03`:
no roundabout, it's a straight intersection) or weren't confidently
identifiable (`ex_adversarial_02`: motion blur). The system prompt
explicitly instructs the model to output `"unknown"` when uncertain -- it
did not do so in any of the three cases, across two different data sources.
A system that never abstains is worse than one that does; a plausible
mitigation is thresholding on the model's own reported `confidence` (it was
below 0.9 in all three failures despite still committing to an answer)
rather than trusting the model's self-assessed decision to abstain or not.

**A second, softer pattern: near-miss maneuver confusions on cautionary
commands.** `ex_easy_02` ("watch out for the cyclist") predicted `yield`
against a gold label of `slow_down`; `ex_easy_03` ("wait behind the parked
truck") predicted `slow_down` against a gold label of `stop`. Both
predictions are defensible interpretations of an inherently soft natural-
language command -- this looks less like a model error and more like the
10-way maneuver taxonomy being finer-grained than these commands
unambiguously specify. Worth revisiting whether `yield`/`slow_down`/`stop`
should be collapsed for scoring purposes, or whether the eval should accept
a small set of acceptable maneuvers per example instead of one gold label.

**Grounding IoU (0.30-0.73) is moderate on average**, but `ex_easy_04` (the
nuScenes white van) scored 0.73 -- notably tighter than the Pexels examples.
Not enough data to claim nuScenes grounds better in general, but worth
watching as more nuScenes examples are added.

**The ambiguous-tier example** (`ex_ambiguous_01`, two candidate cars) was
resolved correctly by defaulting to the nearer vehicle -- but n=1 means this
shouldn't be over-read; more ambiguous examples are needed to say anything
general about how the system resolves multi-candidate references.

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
