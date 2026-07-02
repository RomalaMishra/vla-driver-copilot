# VLA Driver Copilot

A Vision-Language-**Action** system that grounds natural-language driver
commands ("park behind the white van") to a structured, executable driving
decision -- not just a description of the scene.

![demo](docs/demo.gif)

The clip above is real [nuScenes](https://www.nuscenes.org/nuscenes)
footage (`scene-0061`), played straight through with no edits or staging.
Three driver lines are issued back to back as the clip plays -- a mix of
questions and commands -- and the overlay shows the system's actual live
output for each: the grounded object (box), the decision, and the reasoning
behind it. Nothing in the video is scripted after the fact.

## Why this is VLA, not just VLM

A plain VLM demo stops at "the model can describe what it sees" or "the
model can answer a question about the image" -- language *about* vision,
nothing more. This project makes language *produce a decision that would
change what the vehicle does*: a driver command goes in, a structured
action comes out (maneuver + grounded target object + rationale), and it's
evaluated as a prediction task against hand-labeled ground truth -- the
same way real motion-prediction research (nuScenes prediction challenge,
Waymo Open Motion Dataset) evaluates planning without needing a live
simulator. The moment language changes a *decision* instead of just
generating text, you've crossed from VLM into VLA.

## Architecture

```
 Driving frame(s) + driver command ("park behind the white van")
              |
              v
   ┌─────────────────────────┐
   │  reasoning/vlm_client.py │   <- the one hosted API call: a VLM decides
   │  (Groq / Claude / GPT)   │      WHAT to do and WHAT object it refers to
   └─────────────────────────┘
              |  target_description="white van...", maneuver="pull_over"
              v
   ┌─────────────────────────┐
   │ perception/grounding.py │   <- GroundingDINO (local, open-vocabulary)
   │      (GroundingDINO)    │      finds WHERE that object is in the frame
   └─────────────────────────┘
              |  box=[x1,y1,x2,y2]
              v
   ┌─────────────────────────┐
   │  perception/depth.py    │   <- local depth model judges near/mid/far
   └─────────────────────────┘      + closing/opening trend across frames
              |
              v
   ┌─────────────────────────┐
   │    viz/overlay.py       │   <- HUD-style overlay + annotated video
   └─────────────────────────┘
```

**Why two stages instead of one big VLM call for everything**: general
VLMs are often unreliable at precise pixel coordinates. `eval/ablation.py`
quantifies this directly -- it compares this two-stage pipeline against
asking the VLM to output the bounding box itself in a single call, so the
design choice is backed by a measured result, not just asserted.

**What's an API call vs. a local model vs. code we wrote**, since that
distinction matters for understanding the actual contribution here:

| Component | What it is | Hosted API? |
|---|---|---|
| VLM reasoning (intent + maneuver decision) | Groq/Claude/GPT | Yes -- the one API call |
| Grounding (GroundingDINO) | Local pretrained model | No |
| Depth estimation | Local pretrained model (Depth-Anything-V2) | No |
| Tracking, overlay, evaluation harness, ablation | Code in this repo | N/A -- this is the actual system design |

## Evaluated open-loop, on purpose

There's no live vehicle or simulator here -- clips are real, pre-recorded
footage, so the system can't literally execute an action on them. What it
produces instead is a *prediction* of the correct action at each point in
the clip, scored against hand-labeled ground truth. This is standard
practice in AV motion-prediction/planning research, not a workaround --
see `report/writeup.md` for more on this and on what closed-loop control
would take as future work.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (or switch VLM_BACKEND)
```

See `data/README.md` for how to get driving clips in place.

## Usage

```bash
# Run the pipeline on one clip + command, save an annotated video
# (--fps 2 matches nuScenes' native keyframe rate; Pexels clips are already 10fps)
python scripts/run_example.py --clip data/clips/scene-0061 --command "follow the white van ahead" --fps 2

# Score against the hand-labeled eval set
python -m eval.run_eval

# Run the two-stage vs. single-stage ablation
python -m eval.ablation

# Generate the demo reel (real nuScenes footage + a running driver Q&A)
python -m demo.make_demo_video
```

## Results

Pilot eval (n=8, including real nuScenes-mini data): 0.38 overall maneuver
accuracy, but the real finding is in the breakdown -- 0/3 on adversarial
commands referencing absent objects (the model hallucinates a maneuver
instead of abstaining), reproducing consistently across two data sources.
See `report/writeup.md` for the full failure analysis. Ablation: two-stage
grounding scores 0.47 IoU vs. 0.02 for asking the VLM to output the
bounding box directly.

## Project structure

```
reasoning/    VLM reasoning client (the one hosted API call)
perception/   Grounding (GroundingDINO) + depth estimation (local models)
pipeline/     Orchestration: single-frame decision + multi-frame tracking
viz/          HUD overlay + video rendering
eval/         Hand-labeled test set, scoring, and the two-stage ablation
demo/         Assembles the demo reel (nuScenes + running driver Q&A)
data/         Curated clips + nuScenes extraction script
scripts/      CLI entry points
report/       Write-up
```
