"""
Interactive web UI: the clip plays continuously in a browser and a driver
can type a command at any moment -- the pipeline actually reruns on
whatever comes in, live, instead of the pre-scripted demo reel in
demo/make_demo_video.py.

Usage:
    python -m webapp.app --clip data/clips/scene-0061
    (then open http://127.0.0.1:5000)
"""

import argparse
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from flask import Flask, Response, jsonify, render_template, request

from pipeline.track import FrameResult, track_clip_iter

app = Flask(__name__)

STREAM_FPS = 8

MANEUVER_COLORS = {
    "stop": (60, 60, 230),
    "slow_down": (0, 165, 255),
    "yield": (0, 165, 255),
    "turn_left": (255, 200, 0),
    "turn_right": (255, 200, 0),
    "change_lane_left": (255, 200, 0),
    "change_lane_right": (255, 200, 0),
    "pull_over": (220, 220, 0),
    "proceed": (0, 220, 0),
    "unknown": (150, 150, 150),
    "idle": (110, 110, 110),
}

_lock = threading.Lock()
_state = {
    "frames": [],
    "results": [],
    "chat": [],
    "processing": False,
    "playhead": 0,  # frame index currently on screen, kept live by _gen_frames
}


def _neutral_results(n: int) -> list:
    return [
        FrameResult(frame_index=i, box=None, grounding_score=None,
                    distance_bin="unknown", distance_trend="steady",
                    maneuver="idle", rationale="Waiting for a driver command...",
                    confidence=0.0)
        for i in range(n)
    ]


def _load_frames(clip_dir: Path) -> list:
    paths = sorted(clip_dir.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.jpg files found in {clip_dir}")
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def _draw_overlay(frame, result: FrameResult):
    out = frame.copy()

    if result.box is not None:
        x1, y1, x2, y2 = result.box
        grounded_ok = bool(result.grounding_score and result.grounding_score > 0.3)
        color = (0, 220, 0) if grounded_ok else (60, 60, 230)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        tag = f"{result.distance_bin} / {result.distance_trend}"
        cv2.putText(out, tag, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    chip_color = MANEUVER_COLORS.get(result.maneuver, (150, 150, 150))
    chip_text = result.maneuver.upper().replace("_", " ")
    (tw, th), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (10 + tw + 24, 10 + th + 20), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    cv2.rectangle(out, (10, 10), (10 + tw + 24, 10 + th + 20), chip_color, 2)
    cv2.putText(out, chip_text, (22, 10 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, chip_color, 2)

    return out


def _gen_frames():
    idx = 0
    while True:
        with _lock:
            frames = _state["frames"]
            results = _state["results"]
        if not frames:
            time.sleep(0.1)
            continue
        pos = idx % len(frames)
        frame = frames[pos]
        result = results[pos]
        with _lock:
            _state["playhead"] = pos
        annotated = _draw_overlay(frame, result)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        idx += 1
        time.sleep(1 / STREAM_FPS)


def _run_command(command: str, start_index: int = 0):
    with _lock:
        frames = _state["frames"]
    try:
        # track_clip_iter yields the start frame as soon as the one hosted
        # VLM call returns -- that's the driver-facing answer, so post it to
        # the chat immediately instead of waiting for every remaining frame
        # to be re-grounded (which is what made the old blocking
        # track_clip() feel like the copilot had gone silent for tens of
        # seconds). start_index is wherever the clip was actually playing
        # when the driver spoke, so the answer is judged against *that*
        # frame instead of always frame 0.
        it = track_clip_iter(frames, command, start_index=start_index)
        first = next(it)
        with _lock:
            _state["results"][first.frame_index] = first
            _state["chat"].append({
                "role": "copilot",
                "text": first.rationale,
                "maneuver": first.maneuver,
                "confidence": first.confidence,
                "grounded": first.box is not None,
                "distance_bin": first.distance_bin,
            })
        for result in it:
            with _lock:
                _state["results"][result.frame_index] = result
    except Exception as exc:  # surface pipeline errors in the chat instead of a silent hang
        with _lock:
            _state["chat"].append({
                "role": "copilot",
                "text": f"Pipeline error: {exc}",
                "maneuver": "unknown",
                "confidence": 0.0,
                "grounded": False,
                "distance_bin": "unknown",
            })
    finally:
        with _lock:
            _state["processing"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/command", methods=["POST"])
def api_command():
    command = ((request.json or {}).get("command") or "").strip()
    if not command:
        return jsonify({"error": "empty command"}), 400
    with _lock:
        if _state["processing"]:
            return jsonify({"error": "still processing the previous command"}), 429
        _state["processing"] = True
        start_index = _state["playhead"]
        _state["chat"].append({"role": "driver", "text": command})
    threading.Thread(target=_run_command, args=(command, start_index), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify({
            "processing": _state["processing"],
            "chat": _state["chat"],
        })


def _warm_up_models():
    # grounding and depth are both lazy-loaded on first call; without this,
    # that load (weight download/init, tens of seconds) happens during the
    # driver's first command instead of at startup, which is what made the
    # very first answer in a session feel far laggier than every one after.
    from perception import depth, grounding

    print("Warming up perception models (grounding + depth)...")
    grounding._load()
    depth._load()
    print("Perception models ready.")


def _pick_random_clip(clips_root: Path) -> Path:
    candidates = sorted(
        d for d in clips_root.iterdir()
        if d.is_dir() and any(d.glob("frame_*.jpg"))
    )
    if not candidates:
        raise FileNotFoundError(f"No clip directories with frame_*.jpg found under {clips_root}")
    return random.choice(candidates)


def main():
    p = argparse.ArgumentParser(description="Interactive VLA driver-copilot web UI")
    p.add_argument("--clip", default=None,
                    help="Directory of frame_*.jpg files. Default: pick a random "
                         "clip from data/clips/ each run.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    clip_dir = Path(args.clip) if args.clip else _pick_random_clip(Path("data/clips"))
    frames = _load_frames(clip_dir)
    with _lock:
        _state["frames"] = frames
        _state["results"] = _neutral_results(len(frames))

    print(f"Loaded {len(frames)} frames from {clip_dir}")
    _warm_up_models()
    print(f"Open http://{args.host}:{args.port} and type a driver command.")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
