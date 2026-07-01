"""
Open-vocabulary object grounding using GroundingDINO (HuggingFace port).

Unlike a closed-vocabulary detector (e.g. plain YOLO, which only knows a
fixed class list), GroundingDINO takes an arbitrary free-text phrase and
localizes it in the image -- it is itself a vision-language model, just
specialized for localization rather than open-ended reasoning. The VLM
(reasoning/vlm_client.py) decides *what* to look for and *what to do*;
this module only answers *where is it*.

Model is lazy-loaded on first call and cached for the session.
"""

import numpy as np
from PIL import Image

_processor = None
_model = None
_device = None

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def _load():
    global _processor, _model, _device
    if _model is None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[grounding] Loading {MODEL_ID} on {_device} -- first call only...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(_device)
        print("[grounding] Model loaded.")
    return _processor, _model


def ground(
    frame: np.ndarray,
    text_description: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> list:
    """
    Localize a free-text description in a frame.

    Args:
        frame: (H, W, 3) uint8 RGB image
        text_description: e.g. "white van parked on the right side of the road"
        box_threshold: minimum box confidence to keep a detection
        text_threshold: minimum per-token text-match confidence

    Returns:
        list of {"box": [x1, y1, x2, y2], "score": float, "label": str},
        sorted by score descending. Empty list if nothing matched --
        callers should treat this as "target not found", not an error.
    """
    import torch

    processor, model = _load()
    image = Image.fromarray(frame)

    query = text_description.strip().lower()
    if not query.endswith("."):
        query += "."

    inputs = processor(images=image, text=query, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = [
        {"box": [round(v) for v in box.tolist()], "score": round(float(score), 3), "label": label}
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
    ]
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) > 2:
        frame = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
        query = sys.argv[2]
    else:
        print("Usage: python -m perception.grounding <image_path> <text query>")
        print("No args given -- running smoke test on a random frame.")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        query = "white van"

    dets = ground(frame, query)
    print(f"Query: {query!r}")
    print(f"Detections: {dets}")
