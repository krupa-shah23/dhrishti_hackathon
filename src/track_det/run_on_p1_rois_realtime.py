"""
Time-correct real-CSV tracker test — fixes a real bug found in
run_on_p1_rois.py: that script only calls track() for frame_indices
PRESENT in the CSV. Since P1's motion pipeline omits frames with zero
detected ROIs entirely (confirmed in her handoff letter), a real 50-
frame gap of no motion was invisible to the tracker — track() was only
called once for the frame AFTER the gap, so each track's internal age
counter only incremented by 1, not by 50. This makes max_age=10 mean
"10 consecutive calls with no match" rather than "10 real video frames
with no match" — a much looser, unintended threshold once real footage
has sparse/bursty motion (confirmed: this CSV has a 6,681-frame gap
between frame 484 and frame 7165).

This version iterates over EVERY frame index from 0 to the max frame
seen in the CSV, calling track([]) for frames with no ROI, so max_age
is tested against real elapsed frames — the same way main.py's real
per-frame video loop will call it.

Run from the REPO ROOT:
    python -m src.track_det.run_on_p1_rois_realtime src/p1_rois.csv

Compare its printed max track_id and ID-churn pattern against the
original run_on_p1_rois.py's output on the same file — the difference
between the two IS the bug's impact, made visible.
"""
import sys
import csv
from collections import defaultdict

from .tracker import track, reset_tracker


def load_rois_by_frame(csv_path):
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_index"])
            box = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
            frames[frame_idx].append(box)
    return dict(frames)


def main(csv_path):
    frames = load_rois_by_frame(csv_path)
    if not frames:
        print(f"No rows loaded from {csv_path}.")
        return

    max_frame = max(frames.keys())
    reset_tracker()

    max_track_id_seen = 0
    total_calls = 0
    gap_events = []  # (gap_start, gap_end, gap_length) for gaps > max_age
    current_gap_start = None

    print(f"Simulating {max_frame + 1} REAL frames (0..{max_frame}), "
          f"calling track() every frame — including empty gaps.\n")

    for frame_idx in range(max_frame + 1):
        boxes = frames.get(frame_idx, [])
        tracks = track(boxes)
        total_calls += 1

        if boxes:
            if current_gap_start is not None:
                gap_len = frame_idx - current_gap_start
                if gap_len > 10:  # > max_age default
                    gap_events.append((current_gap_start, frame_idx, gap_len))
                current_gap_start = None
        else:
            if current_gap_start is None:
                current_gap_start = frame_idx

        for t in tracks:
            max_track_id_seen = max(max_track_id_seen, t["track_id"])

        # Only print frames where something is actually happening, to
        # keep output readable — real gaps are summarized separately
        # below rather than printed frame-by-frame.
        if boxes:
            ids = sorted(t["track_id"] for t in tracks)
            print(f"Frame {frame_idx}: {len(boxes)} input boxes -> "
                  f"{len(tracks)} tracks, IDs={ids}")

    print(f"\n--- Summary ---")
    print(f"Total real frames simulated: {max_frame + 1}")
    print(f"Highest track_id ever assigned: {max_track_id_seen}")
    print(f"Gaps longer than max_age=10 real frames (tracks SHOULD have "
          f"been dropped and reassigned a new ID after these): "
          f"{len(gap_events)}")
    for start, end, length in gap_events[:20]:
        print(f"  Gap from frame {start} to {end} ({length} real frames "
              f"with zero motion)")
    if len(gap_events) > 20:
        print(f"  ...and {len(gap_events) - 20} more gap(s)")

    print(
        "\nCompare 'Highest track_id ever assigned' here against the "
        "original run_on_p1_rois.py's max ID on the same file. If this "
        "version's max ID is HIGHER, that confirms the original script "
        "was under-counting real ID churn by not calling track() during "
        "empty-motion gaps — i.e. max_age was effectively meaningless "
        "for any gap the original script silently skipped over."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.track_det.run_on_p1_rois_realtime path/to/p1_rois.csv")
        sys.exit(1)
    main(sys.argv[1])