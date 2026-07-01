"""
VLM reasoning client -- sends a driving frame + a natural-language driver
command to a hosted vision-language model and gets back a structured
intent: what maneuver is appropriate and what object it refers to.

This is the one genuine hosted-API call in the pipeline. Grounding and
depth (see perception/) run local pretrained models instead -- see the
architecture section in report/writeup.md for why the pipeline is split
this way.
"""

import base64
import json
import os

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

VALID_MANEUVERS = [
    "stop", "slow_down", "yield", "turn_left", "turn_right",
    "change_lane_left", "change_lane_right", "pull_over", "proceed", "unknown",
]

SYSTEM_PROMPT = """You are the reasoning module of a driving assistant. You are given \
a single camera frame from a car and a natural-language command from the driver. \
Your job is NOT to describe the scene -- it is to decide what the car should do.

Respond with ONLY a JSON object, no other text, with these exact fields:
{
  "target_description": "<a short, specific visual description of the object the \
command refers to, suitable for an object detector to search for -- e.g. 'white \
van parked on the right side of the road'. Use null if the command does not refer \
to a specific object.>",
  "maneuver": "<one of: stop, slow_down, yield, turn_left, turn_right, \
change_lane_left, change_lane_right, pull_over, proceed, unknown>",
  "rationale": "<one short sentence explaining the decision>",
  "confidence": <float 0-1>
}

If the command is ambiguous, refers to something not visible, or you are not \
confident, use "unknown" for maneuver and explain why in the rationale rather \
than guessing."""


def _encode_image(frame: np.ndarray) -> str:
    """frame: (H, W, 3) uint8 RGB -> base64 data URI (JPEG)."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text)
    if result.get("maneuver") not in VALID_MANEUVERS:
        result["maneuver"] = "unknown"
    return result


def interpret(frame: np.ndarray, command: str, model: str = None) -> dict:
    """
    Args:
        frame: (H, W, 3) uint8 RGB image
        command: natural-language driver command, e.g. "park behind the white van"
        model: override the default vision model for the active backend

    Returns:
        {
            "target_description": str | None,
            "maneuver": str,       # one of VALID_MANEUVERS
            "rationale": str,
            "confidence": float,
        }
    """
    backend = os.getenv("VLM_BACKEND", "groq").lower()
    if backend == "groq":
        return _interpret_groq(frame, command, model)
    if backend == "anthropic":
        return _interpret_anthropic(frame, command, model)
    if backend == "openai":
        return _interpret_openai(frame, command, model)
    raise ValueError(f"Unknown VLM_BACKEND: {backend}")


def _interpret_groq(frame: np.ndarray, command: str, model: str = None) -> dict:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = model or os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    image_uri = _encode_image(frame)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Driver command: "{command}"'},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return _parse_response(response.choices[0].message.content)


def _interpret_anthropic(frame: np.ndarray, command: str, model: str = None) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = model or os.getenv("ANTHROPIC_VLM_MODEL", "claude-sonnet-5")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": f'Driver command: "{command}"'},
            ],
        }],
    )
    return _parse_response(response.content[0].text)


SINGLE_STAGE_SYSTEM_PROMPT = """You are the reasoning module of a driving assistant. You are given \
a single camera frame from a car and a natural-language command from the driver. \
Decide what the car should do AND localize the relevant object yourself.

Respond with ONLY a JSON object, no other text:
{
  "maneuver": "<one of: stop, slow_down, yield, turn_left, turn_right, \
change_lane_left, change_lane_right, pull_over, proceed, unknown>",
  "rationale": "<one short sentence>",
  "confidence": <float 0-1>,
  "box": [x1, y1, x2, y2] or null if no specific object / not visible
}

Box coordinates are pixel coordinates in the given image (top-left origin). \
Use null for box if the command doesn't refer to a specific object or it isn't visible."""


def interpret_single_stage(frame: np.ndarray, command: str, model: str = None) -> dict:
    """
    Ablation baseline: ask the VLM to both decide the maneuver AND output
    the target bounding box directly, in one call, instead of delegating
    localization to perception.grounding. See eval/ablation.py -- this
    exists to measure how much the dedicated grounding model actually
    buys you over asking a general VLM for coordinates.
    """
    backend = os.getenv("VLM_BACKEND", "groq").lower()
    image_uri = _encode_image(frame)

    if backend == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = model or os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SINGLE_STAGE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f'Driver command: "{command}"'},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ]},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        text = response.choices[0].message.content
    else:
        raise NotImplementedError(f"Single-stage ablation not wired up for backend={backend!r} yet")

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text)
    if result.get("maneuver") not in VALID_MANEUVERS:
        result["maneuver"] = "unknown"
    return result


def _interpret_openai(frame: np.ndarray, command: str, model: str = None) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = model or os.getenv("OPENAI_VLM_MODEL", "gpt-4o")
    image_uri = _encode_image(frame)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Driver command: "{command}"'},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return _parse_response(response.choices[0].message.content)
