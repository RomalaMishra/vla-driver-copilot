# Data

This project expects short driving clips as sequences of extracted frames:

```
data/clips/<clip_name>/frame_0000.jpg
data/clips/<clip_name>/frame_0001.jpg
...
```

Curated clips (small JPEG sequences, not raw video) are committed directly
in `data/clips/` for reproducibility -- see `data/clips/SOURCES.md` for
provenance. `data/raw/` (source `.mp4`s) and `data/nuscenes/` (the full
nuScenes-mini download) are gitignored -- too large to commit, and easy to
regenerate/re-download.

## nuScenes-mini (in use)

`scene-0061` is already extracted and in the eval set. To add more scenes:

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes) (free,
   requires agreeing to their terms -- has to be done by a real person).
2. Download `v1.0-mini.tgz`, extract into `data/nuscenes/`.
3. `pip install nuscenes-devkit`
4. `python -m data.extract_nuscenes --list` to see available scenes, then
   `python -m data.extract_nuscenes --scene scene-XXXX` to extract one into
   `data/clips/scene-XXXX/`.

## Alternative: any short dashcam clip

The pipeline doesn't hard-depend on nuScenes -- `example_01`/`example_02`
are free-license Pexels clips, extracted frame-by-frame via OpenCV (see
`data/clips/SOURCES.md`). Any short (5-10 sec) driving clip works, split
into sequential JPEGs; just make sure the source is appropriately licensed
if this repo/demo is going to be public.
