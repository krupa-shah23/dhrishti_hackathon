"""
Synthetic test for is_invigilator_track() — checks it correctly tells
apart a seated, fidgeting student from a patrolling invigilator, using
only track.history (centroid path), no appearance model.

Run from the REPO ROOT:
    python -m src.track_det.test_invigilator_filter

THIS TEST USES MADE-UP PIXEL VALUES. It exists to sanity-check the
DISCRIMINATION LOGIC (net_displacement + path_length together), not to
validate real thresholds — those need retuning against P1's real
footage once available (real camera resolution, real aisle length in
pixels, real frame rate for how many hits a full walk takes).
"""

from .invigilator_filter import is_invigilator_track, path_length, net_displacement


def make_seated_student_history(num_frames: int = 20):
    """
    Simulates a seated student fidgeting: small back-and-forth motion
    within roughly a 30x30px box, never leaving the seat area. Real
    path_length accumulates (lots of small back-and-forth steps) but
    net_displacement stays near zero (ends up close to where started).
    """
    history = []
    cx, cy = 300, 300  # seat position, roughly fixed
    for i in range(num_frames):
        # oscillate +/- 15px around the seat position
        offset = 15 if i % 2 == 0 else -15
        x1, y1 = cx + offset, cy + offset
        history.append((x1, y1, x1 + 30, y1 + 30))
    return history


def make_invigilator_history(num_frames: int = 20):
    """
    Simulates an invigilator walking steadily down a long aisle: large,
    consistent per-frame displacement in ONE direction. Both
    path_length AND net_displacement end up large, since motion is
    directional rather than back-and-forth.
    """
    history = []
    for i in range(num_frames):
        x1 = 50 + i * 20   # steady rightward walk, 20px/frame
        y1 = 100
        history.append((x1, y1, x1 + 30, y1 + 60))
    return history


def make_short_glitch_history():
    """
    Simulates a brief tracking glitch: only 3 frames of history with a
    big jump — should NOT be flagged as invigilator, because min_hits
    guards against judging on too little evidence.
    """
    return [
        (50, 50, 80, 100),
        (300, 50, 330, 100),
        (310, 55, 340, 105),
    ]


def main():
    student_hist = make_seated_student_history()
    invigilator_hist = make_invigilator_history()
    glitch_hist = make_short_glitch_history()

    print("=== Seated student (fidgeting, should NOT be flagged) ===")
    print(f"  path_length={path_length(student_hist):.1f} "
          f"net_displacement={net_displacement(student_hist):.1f}")
    print(f"  is_invigilator_track = {is_invigilator_track(student_hist)}  "
          f"(expected: False)")

    print("\n=== Invigilator (steady walk, SHOULD be flagged) ===")
    print(f"  path_length={path_length(invigilator_hist):.1f} "
          f"net_displacement={net_displacement(invigilator_hist):.1f}")
    print(f"  is_invigilator_track = {is_invigilator_track(invigilator_hist)}  "
          f"(expected: True)")

    print("\n=== Short tracking glitch (big jump, too little history) ===")
    print(f"  path_length={path_length(glitch_hist):.1f} "
          f"net_displacement={net_displacement(glitch_hist):.1f}")
    print(f"  is_invigilator_track = {is_invigilator_track(glitch_hist)}  "
          f"(expected: False — min_hits=15 not met, only 3 frames)")

    print(
        "\nIf student shows True or invigilator shows False, the "
        "min_net_displacement/min_path_length thresholds in "
        "invigilator_filter.py need adjusting for THIS synthetic data "
        "before you even get to real footage. If the glitch case shows "
        "True, min_hits isn't guarding correctly."
    )


if __name__ == "__main__":
    main()