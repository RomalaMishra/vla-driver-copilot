"""
Monocular depth estimation using Depth Anything V2 Small (HuggingFace).

Runs locally (CPU or GPU), lazy-loaded and cached for the session. Used
to judge relative distance to the grounded target -- e.g. is the car
getting closer to the white van, and is it now close enough to stop.

Note: this gives *relative* depth, not calibrated metric distance (that
would require known camera intrinsics, which nuScenes provides per-clip
but a generic dashcam clip does not). We report a near/mid/far bin plus
a frame-to-frame trend (closing/steady/opening) rather than claiming
precise meters -- see report/writeup.md for why that's the honest choice.

Public API:
    estimate(frame)                       -> (H, W) float32 relative depth map
    depth_bin(depth_map, box)             -> "near" | "mid" | "far" | "unknown"
    trend(prev_bin, curr_bin)             -> "closing" | "opening" | "steady"
"""

import numpy as np
from PIL import Image

_pipe = None

_BIN_ORDER = {"near": 0, "mid": 1, "far": 2}


def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline as hf_pipeline

        print("[depth] Loading Depth Anything V2 Small -- first call only...")
        _pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        print("[depth] Model loaded.")
    return _pipe


def estimate(frame: np.ndarray) -> np.ndarray:
    """
    Args:
        frame: (H, W, 3) uint8 RGB image

    Returns:
        (H, W) float32 relative depth map -- larger values = further away.
    """
    pipe = _load()
    result = pipe(Image.fromarray(frame))
    return np.array(result["depth"], dtype=np.float32)


def depth_bin(depth_map: np.ndarray, box: list) -> str:
    """
    Classify relative depth of a bounding box region.

    Args:
        depth_map : (H, W) float32 from estimate()
        box       : [x1, y1, x2, y2]

    Returns:
        "near" | "mid" | "far" | "unknown"
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    d_min, d_max = depth_map.min(), depth_map.max()
    norm = (roi.mean() - d_min) / (d_max - d_min + 1e-6)
    if norm < 0.33:
        return "near"
    if norm < 0.66:
        return "mid"
    return "far"


def trend(prev_bin: str, curr_bin: str) -> str:
    """Frame-to-frame distance trend, for the live HUD readout."""
    if prev_bin not in _BIN_ORDER or curr_bin not in _BIN_ORDER:
        return "steady"
    delta = _BIN_ORDER[curr_bin] - _BIN_ORDER[prev_bin]
    if delta < 0:
        return "closing"
    if delta > 0:
        return "opening"
    return "steady"


if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) > 1:
        frame = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    else:
        print("No image provided. Using random 480x640 frame.")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = estimate(frame)
    print(f"Depth map shape : {depth.shape}")
    print(f"Depth range     : {depth.min():.2f} - {depth.max():.2f}")
    print(f"Depth bin [50,50,200,200]: {depth_bin(depth, [50, 50, 200, 200])}")
