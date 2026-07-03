Frame sequences in this directory are extracted from free-license Pexels videos (no attribution required, but noted here for provenance):

- `example_01/` -- from [pexels.com/video/5384116](https://www.pexels.com/video/traveling-a-road-in-an-urban-area-5384116/), rural road, motion-blurred (kept as a pipeline stress test)
- `example_02/` -- from [pexels.com/video/4644521](https://www.pexels.com/video/dash-cam-footage-in-city-driving-4644521/) by German Korb, downtown Montreal, clear/sharp -- primary demo clip

All 10 scenes in nuScenes-mini, extracted via `data/extract_nuscenes.py`:

- `scene-0061/` -- "Parked truck, construction, intersection, turn left, following a van"
- `scene-0103/` -- "Many peds right, wait for turning car, long bike rack left, cyclist"
- `scene-0553/` -- "Wait at intersection, bicycle, large truck, peds crossing crosswalk, ped with stroller"
- `scene-0655/` -- "Parking lot, parked cars, jaywalker, bendy bus, gardening vehicles"
- `scene-0757/` -- "Arrive at busy intersection, bus, wait at intersection, bicycle, peds"
- `scene-0796/` -- "Scooter, peds on sidewalk, bus, cars, truck, fake construction worker, bicycle, cross intersection, car overtaking us"
- `scene-0916/` -- "Parking lot, bicycle rack, parked bicycles, bus, many peds, parked scooters, parked motorcycle"
- `scene-1077/` -- "Night, big street, bus stop, high speed, construction vehicle"
- `scene-1094/` -- "Night, after rain, many peds, PMD, ped with bag, jaywalker, truck, scooter"
- `scene-1100/` -- "Night, peds in sidewalk, peds cross crosswalk, scooter, PMD, difficult lighting"

`webapp/app.py` picks a random clip directory (any `example_*`/`scene-*` under `data/clips/`)
on each run when `--clip` isn't passed, so every scene above is in rotation.

nuScenes usage requires citation per their Terms of Use:

```bibtex
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora
          and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan
          and Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
```
