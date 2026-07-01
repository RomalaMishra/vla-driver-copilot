# Data

This project expects short driving clips as sequences of extracted frames:

```
data/clips/<clip_name>/frame_0000.jpg
data/clips/<clip_name>/frame_0001.jpg
...
```

`data/clips/` is gitignored (raw frames shouldn't live in the repo) --
populate it locally before running `scripts/run_example.py`, `eval/run_eval.py`,
or `demo/make_demo_video.py`.

## Recommended source: nuScenes-mini

[nuScenes](https://www.nuscenes.org/nuscenes) is the standard dataset for
this kind of work in the AV research community -- real driving scenes with
3D annotations, which is what lets `perception/depth.py` be checked against
real distances rather than only relative depth. The `mini` split (10 scenes)
is small enough for this project's scope.

1. Register at nuscenes.org (free, requires agreeing to their terms --
   this has to be done by a real person, not automated).
2. Download the `v1.0-mini` split.
3. Use `nuscenes-devkit` to extract camera frames for a scene into
   `data/clips/<clip_name>/frame_%04d.jpg`.

## Alternative: any short dashcam clip

The pipeline doesn't hard-depend on nuScenes -- any short (5-10 sec) driving
clip works, split into sequential JPEG frames with `ffmpeg`:

```bash
ffmpeg -i your_clip.mp4 -vf fps=10 data/clips/your_clip_name/frame_%04d.jpg
```

Just make sure whatever source clips you use are appropriately licensed if
this repo/demo is going to be public.
