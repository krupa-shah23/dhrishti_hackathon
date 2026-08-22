"""
generate_p3_detection_output_clip1.py
Produces real detect_objects() output on clip1, per known event window,
for P3's extract_features() wiring (same pattern as the clip3 script).

Run from repo root:
    uv run python scripts/generate_p3_detection_output_clip1.py

Requires: models/phone_detector_v3.pt already trained, clip1 video file
present locally (adjust VIDEO_PATH below if your filename/path differs).

ASSUMPTION (confirm before trusting output): clip1 = the file
"01.Candidate was found using a mobile phone in the examination hall..mkv"
(31,946,378 bytes in college_dataset listing), matching Krupa's original
WhatsApp transcription for clip 1 (takeout 6s / copying 12-16s / copying
51s / intervention 55s / leaves scene 1m8s). Flag this mapping to the team
when handing off results, same way clip7/clip3 mappings were explicitly
confirmed rather than assumed.
"""
import csv
import sys
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.track_det.detector import detect_objects

VIDEO_PATH = REPO_ROOT / "data" / "college_dataset" / "01.Candidate was found using a mobile phone in the examination hall..mkv"
OUT_CSV = REPO_ROOT / "p3_detection_output_clip1.csv"
EXAM_MODE = "CBT"

# Known event windows from Krupa's clip1 transcription. Padded +/-1.5s
# around instant events (matching the clip3 script's convention, per
# P2 doc §4.1's occlude-reveal window logic). c1_e2 is already a range
# (12-16s) so it gets a smaller +/-1.5s pad on each end rather than a
# full window built around a single instant.
#
# c1_e4 (intervention) and c1_e5 (leaves scene) are NOT expected to
# contain a phone detection — they're included deliberately as a
# false-positive sanity check, not because an object is expected there.
EVENT_WINDOWS = [
    ("c1_e1", 4.5, 7.5, "takeout"),
    ("c1_e2", 10.5, 17.5, "copying (photo of screen)"),
    ("c1_e3", 49.5, 52.5, "copying (photo of screen)"),
    ("c1_e4", 53.5, 56.5, "intervention -- no object expected, FP check"),
    ("c1_e5", 66.5, 69.5, "accused leaves scene -- no object expected, FP check"),
]


def extract_window_crops(cap, fps, start_s, end_s, sample_every_n_frames=2):
    """Grabs frames across [start_s, end_s], returns list of full frames.
    Real ROI cropping isn't available yet (P1's fusion stage), so this
    passes full frames as crops -- same known simplification as the
    clip3 script. Swap in real ROI boxes once P1's windows land."""
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
    for event_id, start_s, end_s, desc in EVENT_WINDOWS:
        crops = extract_window_crops(cap, fps, start_s, end_s)
        if not crops:
            rows.append({
                "event_id": event_id, "window_start_s": max(0, start_s),
                "window_end_s": end_s, "class": None, "confidence": None,
                "note": f"no frames extracted — check VIDEO_PATH/timestamps ({desc})",
            })
            continue

        results = detect_objects(crops, EXAM_MODE)  # real frozen contract call
        if not results:
            rows.append({
                "event_id": event_id, "window_start_s": max(0, start_s),
                "window_end_s": end_s, "class": None, "confidence": 0.0,
                "note": f"detect_objects() returned [] for this window ({desc})",
            })
        else:
            for r in results:
                rows.append({
                    "event_id": event_id, "window_start_s": max(0, start_s),
                    "window_end_s": end_s, "class": r["class"],
                    "confidence": round(r["confidence"], 4), "note": desc,
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
    print("NOTE: c1_e4 (intervention) and c1_e5 (leaves scene) are included as")
    print("false-positive sanity checks -- no object is expected in those windows.")
    print("If either fires a confident phone detection, flag it to P3 as a real FP,")
    print("not just noise to discard.")


if __name__ == "__main__":
    main()