"""
Extracts CAM_FRONT keyframes from a nuScenes-mini scene into
data/clips/<scene_name>/frame_%04d.jpg, matching the frame-sequence
convention the rest of the pipeline expects.

Usage:
    python -m data.extract_nuscenes --scene scene-0061
    python -m data.extract_nuscenes --list   # show available scenes
"""

import argparse
import shutil
from pathlib import Path

import cv2

DATAROOT = Path(__file__).parent / "nuscenes"
CLIPS_DIR = Path(__file__).parent / "clips"
TARGET_W = 640


def _load_nusc():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version="v1.0-mini", dataroot=str(DATAROOT), verbose=False)


def list_scenes():
    nusc = _load_nusc()
    for scene in nusc.scene:
        print(f"{scene['name']:12s} ({scene['nbr_samples']:2d} samples) -- {scene['description']}")


def extract_scene(scene_name: str) -> Path:
    nusc = _load_nusc()
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"Scene {scene_name!r} not found. Run with --list to see options.")

    out_dir = CLIPS_DIR / scene_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    sample_token = scene["first_sample_token"]
    i = 0
    while sample_token:
        sample = nusc.get("sample", sample_token)
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data = nusc.get("sample_data", cam_token)
        img_path = DATAROOT / cam_data["filename"]

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        scale = TARGET_W / w
        frame = cv2.resize(frame, (TARGET_W, int(h * scale)))
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), frame)

        i += 1
        sample_token = sample["next"]

    print(f"Extracted {i} keyframes ({scene['description']}) -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", help="Scene name, e.g. scene-0061")
    p.add_argument("--list", action="store_true", help="List available scenes and exit")
    args = p.parse_args()

    if args.list:
        list_scenes()
    elif args.scene:
        extract_scene(args.scene)
    else:
        p.error("Provide --scene <name> or --list")
