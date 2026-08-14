"""
Loads P1's ROI boxes from CSV and feeds them into track() one REAL
video frame at a time, including frames with zero ROIs — matching how
main.py's actual per-frame video loop will call track().

FIXED VERSION (replaces the original run_on_p1_rois.py): the original
only called track() for frame_indices PRESENT in the CSV. Since P1's
motion pipeline omits frames with zero detected ROIs entirely (per her
handoff letter), a real N-frame gap of no motion was invisible to the
tracker — track() was only called once for the frame AFTER the gap, so
each track's age counter incremented by 1 regardless of how large N
was. That made max_age=10 mean "10 consecutive calls with no match"
rather than "10 real video frames with no match" — silently wrong on
sparse/bursty real footage (confirmed on the real p1_rois.csv: gaps up
to 666 real frames long, 27 gaps total exceeding max_age=10).

This version walks every real frame index from 0 to the max frame seen
in the CSV, calling track([]) for empty ones, so max_age is evaluated
against true elapsed frames.

Expected CSV columns: frame_index, x1, y1, x2, y2

Run from the REPO ROOT (not from inside src/track_det/):
    python -m src.track_det.run_on_p1_rois src/p1_rois.csv

Add --quiet to suppress per-frame lines and only print the summary
(recommended for large files — the per-frame output on a multi-
thousand-frame CSV is unreadable in a terminal):
    python -m src.track_det.run_on_p1_rois src/p1_rois.csv --quiet
"""
import sys
import csv
from collections import defaultdict

from .tracker import track, reset_tracker

DEFAULT_MAX_AGE = 25  # CentroidTracker's built-in default, used only when
                        # no --max-age override is passed
DEFAULT_MAX_DISTANCE = 80.0  # CentroidTracker's built-in default



def load_rois_by_frame(csv_path):
    """
    Returns: dict {frame_index: [(x1, y1, x2, y2), ...]}
    Frames absent from the CSV mean zero ROIs that frame (per P1's
    contract) — NOT "not yet processed". Callers must account for that
    by iterating the full frame range, not just this dict's keys.
    """
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


def main(csv_path, quiet=False, max_distance=None, max_age=None):
    frames = load_rois_by_frame(csv_path)
    if not frames:
        print(f"No rows loaded from {csv_path}.")
        return

    # Effective max_age actually in effect this run -- the override if one
    # was passed, otherwise the tracker's real default. Gap-flagging below
    # must be evaluated against THIS value, not a hardcoded constant, or
    # the reported gap list silently stops matching the real tracking
    # behavior whenever --max-age is overridden.
    effective_max_age = max_age if max_age is not None else DEFAULT_MAX_AGE

    max_frame = max(frames.keys())
    reset_tracker(max_distance=max_distance, max_age=max_age)
    print(
        f"[run_on_p1_rois] Effective tuning: "
        f"max_distance={max_distance if max_distance is not None else DEFAULT_MAX_DISTANCE}, "
        f"max_age={effective_max_age}"
    )

    max_track_id_seen = 0
    gap_events = []
    current_gap_start = None

    print(f"Loaded ROI data spanning {len(frames)} active frames out of "
          f"{max_frame + 1} total real frames (0..{max_frame}).")
    print("Calling track() every real frame, including empty gaps.\n")

    for frame_idx in range(max_frame + 1):
        boxes = frames.get(frame_idx, [])
        tracks = track(boxes)

        if boxes:
            if current_gap_start is not None:
                gap_len = frame_idx - current_gap_start
                if gap_len > effective_max_age:
                    gap_events.append((current_gap_start, frame_idx, gap_len))
                current_gap_start = None
        elif current_gap_start is None:
            current_gap_start = frame_idx

        for t in tracks:
            max_track_id_seen = max(max_track_id_seen, t["track_id"])

        if boxes and not quiet:
            ids = sorted(t["track_id"] for t in tracks)
            print(f"Frame {frame_idx}: {len(boxes)} input boxes -> "
                  f"{len(tracks)} tracks, IDs={ids}")

    print(f"\n--- Summary ---")
    print(f"Total real frames simulated: {max_frame + 1}")
    print(f"Highest track_id ever assigned: {max_track_id_seen}")
    print(f"Gaps longer than max_age={effective_max_age} real frames "
          f"(track correctly dropped and reassigned a new ID after "
          f"these): {len(gap_events)}")
    for start, end, length in gap_events[:20]:
        print(f"  Gap from frame {start} to {end} ({length} real frames "
              f"with zero motion)")
    if len(gap_events) > 20:
        print(f"  ...and {len(gap_events) - 20} more gap(s)")

    print(
        "\nCheck: same physical person's activity bursts get a NEW "
        "track_id after any gap longer than max_age — this is expected "
        "and correct given current settings, not a bug. If your team "
        "decides bursts must link back to the same person across long "
        "gaps, max_age/tracking approach needs to change; see the "
        "team discussion on event-level vs person-level identity."
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    quiet_flag = "--quiet" in args

    max_distance_override = None
    max_age_override = None
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--quiet":
            i += 1
        elif args[i] == "--max-distance":
            max_distance_override = float(args[i + 1])
            i += 2
        elif args[i] == "--max-age":
            max_age_override = int(args[i + 1])
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if len(positional) != 1:
        print("Usage: python -m src.track_det.run_on_p1_rois path/to/p1_rois.csv "
              "[--quiet] [--max-distance FLOAT] [--max-age INT]")
        sys.exit(1)

    main(positional[0], quiet=quiet_flag,
         max_distance=max_distance_override, max_age=max_age_override)