"""
Stress test for track() / CentroidTracker — harder synthetic cases than
test_track.py's happy path. Each case isolates ONE failure mode so you
can tell exactly which mechanism (max_distance vs max_age vs greedy
matching) is responsible if something breaks.

Run from the REPO ROOT:
    python -m src.track_det.test_track_stress

This does NOT replace test_track.py — keep both. test_track.py is your
quick "did I break anything" smoke test; this is for characterizing
failure modes before real footage arrives.
"""

from .tracker import track, reset_tracker


def run_case(name, frames, expected_note):
    print(f"\n=== {name} ===")
    print(f"Expected: {expected_note}")
    reset_tracker()
    for i, boxes in enumerate(frames):
        tracks = track(boxes)
        # Sort by track_id for stable reading, and show each ID's box —
        # the ID SET alone (e.g. [1, 2]) can't reveal a swap, since a
        # swap changes WHICH box an ID is attached to, not how many IDs
        # exist. Printing centroid x makes left/right position obvious
        # at a glance.
        tracks_sorted = sorted(tracks, key=lambda t: t["track_id"])
        details = []
        for t in tracks_sorted:
            x1, y1, x2, y2 = t["box"]
            cx = (x1 + x2) / 2.0
            details.append(f"id={t['track_id']}@x={cx:.0f}")
        print(f"  Frame {i}: {len(boxes)} input boxes -> {len(tracks)} tracks: "
              f"{', '.join(details)}")


# ---------------------------------------------------------------------
# Case 1: Crossing paths
# Two boxes start apart, move toward each other, cross, then separate.
# Failure mode to watch for: IDs SWAPPING at/after the crossing point
# (person A's ID ending up on person B's box after they cross).
# ---------------------------------------------------------------------
crossing_frames = [
    [(10, 100, 40, 150), (400, 100, 430, 150)],    # far apart
    [(60, 100, 90, 150), (350, 100, 380, 150)],    # step <= 80px (50px move)
    [(110, 100, 140, 150), (300, 100, 330, 150)],  # 50px move
    [(160, 100, 190, 150), (250, 100, 280, 150)],  # 50px move, approaching
    [(210, 100, 240, 150), (200, 100, 230, 150)],  # near-overlap / crossing
    [(260, 100, 290, 150), (150, 100, 180, 150)],  # crossed — positions swapped
    [(310, 100, 340, 150), (100, 100, 130, 150)],  # separating, 50px steps
]

# ---------------------------------------------------------------------
# Case 2: Temporary occlusion
# One box present, then MISSING for 3 frames (simulating someone
# stepping behind another person / out of frame briefly), then
# reappears near where it left off.
# Failure mode to watch for: a NEW track ID being spawned on reappearance
# instead of the original ID surviving the gap (depends on max_age=10 —
# 3 missed frames should survive; this case should PASS with default
# settings. Extend the gap beyond 10 frames in a follow-up run to find
# where it actually breaks).
# ---------------------------------------------------------------------
occlusion_frames = [
    [(200, 200, 230, 260)],
    [(205, 202, 235, 262)],
    [],  # occluded — frame 2
    [],  # occluded — frame 3
    [],  # occluded — frame 4
    [(220, 210, 250, 270)],  # reappears
]

# ---------------------------------------------------------------------
# Case 3: Fast movement (large frame-to-frame jump)
# A box that moves further than max_distance=80.0 px in one step —
# simulates someone standing up quickly / a fast reach.
# Failure mode to watch for: this SHOULD spawn a new ID under current
# settings (jump > 80px), which may be the WRONG behavior for real
# footage — if this is common in real clips, max_distance needs raising.
# ---------------------------------------------------------------------
fast_move_frames = [
    [(50, 50, 80, 100)],
    [(60, 55, 90, 105)],     # normal small move
    [(250, 55, 280, 105)],   # jump of ~190px in one frame — exceeds max_distance
    [(255, 58, 285, 108)],   # continues near new position
]


if __name__ == "__main__":
    run_case(
        "Case 1: Crossing paths",
        crossing_frames,
        "Track id=1 starts on the LEFT (low x) and id=2 on the RIGHT "
        "(high x). Watch the x= values frame by frame: if id=1's x keeps "
        "climbing smoothly through the crossing and id=2's keeps "
        "dropping smoothly, no swap happened. If id=1's x suddenly jumps "
        "to match what id=2 was doing (or vice versa) around the "
        "crossing frame, that's the swap — a greedy-matching limitation "
        "(no Hungarian algorithm), flag it, don't just re-run.",
    )

    run_case(
        "Case 2: Temporary occlusion (3-frame gap, max_age=10)",
        occlusion_frames,
        "Original track ID should survive the 3-frame gap and reattach "
        "on reappearance (3 <= max_age=10). If a NEW id appears instead, "
        "max_age isn't behaving as expected — check age-increment logic.",
    )

    run_case(
        "Case 3: Fast movement (~190px jump, max_distance=80.0)",
        fast_move_frames,
        "Jump exceeds max_distance, so a NEW id is expected here under "
        "current settings. This is the actual real-world question: is "
        "80.0 too tight for your footage's resolution/frame rate? "
        "Decide once you have P1's real clip + FPS.",
    )

    print(
        "\nEach case above ran on a FRESH tracker (reset_tracker() called "
        "before each). IDs restart at 1 for every case now, so ID sets "
        "are directly comparable to each case's 'Expected' note above — "
        "no cross-case contamination."
    )