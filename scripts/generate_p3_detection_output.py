"""
generate_p3_detection_output.py
Produces real detect_objects() output on clip3, per known event window,
for P3's extract_features() wiring (replacing mock/random object_flag values).

Run from repo root:
    uv run python scripts/generate_p3_detection_output.py

Requires: models/phone_detector_v3.pt already trained, clip3 video file
present locally (adjust VIDEO_PATH below to your actual file).
"""
import csv
import sys
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.track_det.detector import detect_objects
VIDEO_PATH = REPO_ROOT / "data" / "college_dataset" / "03.CCTV Mobile Usage.mkv"
OUT_CSV = REPO_ROOT / "p3_detection_output_clip3.csv"
EXAM_MODE = "CBT"

# Known event windows from ground_truth_events.csv (clip3) — each is
# (event_id, start_s, end_s). Padded +/-1.5s around instant events so the
# window spans a realistic occlude-reveal cycle, matching P2 doc §4.1.
EVENT_WINDOWS = [
    ("c3_e1", -1.5, 1.5),      # phone hidden under shirt, start of clip
    ("c3_e2", 54.5, 57.5),     # takes phone out from under shirt
    ("c3_e3", 63.5, 66.5),     # photographs screen, hides under desk
    ("c3_e4", 79.5, 90.5),     # takes phone out again to photograph
    ("c3_e5", 91.5, 101.5),    # takes phone out to copy
    ("c3_e6", 105.5, 108.5),   # takes phone out, photographs screen
    ("c3_e7", 122.5, 125.5),   # takes phone out, photographs screen
    ("c3_e8", 129.5, 132.5),   # hides phone in pants
]


def extract_window_crops(cap, fps, start_s, end_s, sample_every_n_frames=2):
    """Grabs frames across [start_s, end_s], returns list of full frames.
    Real ROI cropping isn't available yet (P1's fusion stage), so this
    passes full frames as crops — a known simplification, noted in the
    handoff message. Swap in real ROI boxes once P1's windows land."""
    start_s = max(0, start_s)
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    crops = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % sample_every_n_frames == 0:
            crops.append(frame)
        frame_idx += 1
    return crops


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {VIDEO_PATH} — check path")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    rows = []
    for event_id, start_s, end_s in EVENT_WINDOWS:
        crops = extract_window_crops(cap, fps, start_s, end_s)
        if not crops:
            rows.append({
                "event_id": event_id, "window_start_s": max(0, start_s),
                "window_end_s": end_s, "class": None, "confidence": None,
                "note": "no frames extracted — check VIDEO_PATH/timestamps",
            })
            continue

        results = detect_objects(crops, EXAM_MODE)  # real frozen contract call
        if not results:
            rows.append({
                "event_id": event_id, "window_start_s": max(0, start_s),
                "window_end_s": end_s, "class": None, "confidence": 0.0,
                "note": "detect_objects() returned [] for this window",
            })
        else:
            for r in results:
                rows.append({
                    "event_id": event_id, "window_start_s": max(0, start_s),
                    "window_end_s": end_s, "class": r["class"],
                    "confidence": round(r["confidence"], 4), "note": "",
                })

    cap.release()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "window_start_s", "window_end_s", "class", "confidence", "note"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print("NOTE for P3: these are real detect_objects() calls on the real trained")
    print("model, but crops are full frames (P1's real ROI windows aren't wired")
    print("in yet) — say this plainly when handing off, don't claim ROI-gated output.")


if __name__ == "__main__":
    main()