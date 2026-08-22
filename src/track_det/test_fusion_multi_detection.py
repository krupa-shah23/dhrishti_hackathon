# src/track_det/test_fusion_multi_detection.py
"""
Tests fuse_track_detections() behavior when TWO detections legitimately fall
inside the SAME track's box in the same frame (e.g. a phone AND a paper chit
both visible on one person). Hungarian assignment enforces 1:1 matching, so
this checks whether the second detection is silently dropped, and whether
that's actually a problem in practice (per the open TODO in master doc §18).

Detections use the real detect_objects() contract: (box, class_name, conf)
tuples -- NOT dicts. (Earlier version of this script incorrectly used dicts,
which silently unpacked as 3 dict-key strings instead of a box, producing a
confusing downstream ValueError rather than a clear type error.)

Usage:
    uv run python -m src.track_det.test_fusion_multi_detection
"""
from .fusion import fuse_track_detections


def make_track(track_id, box):
    return {"track_id": track_id, "box": box}


def run_case(name, tracks, detections):
    print(f"\n=== {name} ===")
    print(f"Tracks: {tracks}")
    print(f"Detections: {detections}")
    result = fuse_track_detections(tracks, detections)
    print(f"Result: {result}")
    return result


def main():
    # Case 1: one large track box, two legitimate detections inside it
    # (phone in one hand, chit-like paper in the other), same frame.
    track_box = (100, 100, 400, 400)
    phone_det = ((150, 150, 200, 200), "phone", 0.90)
    chit_det = ((300, 300, 350, 350), "paper-chit", 0.85)

    result = run_case(
        "Two legit detections, one track box",
        tracks=[make_track(1, track_box)],
        detections=[phone_det, chit_det],
    )

    n_matched = sum(1 for r in result if r.get("class") is not None)
    print(f"\n--- Verdict ---")
    print(f"Detections matched to track: {n_matched} of 2")
    if n_matched == 1:
        print("CONFIRMED LIMITATION: Hungarian's 1:1 constraint drops the second "
              "legitimate detection. One of phone/chit is silently lost this frame.")
    elif n_matched == 2:
        print("No limitation found -- both detections retained.")
    else:
        print("UNEXPECTED -- neither detection matched. Investigate before trusting "
              "either outcome above.")

    # Case 2: same scenario across a short sequence, to check for flicker in
    # which detection wins frame-to-frame.
    print(f"\n=== Multi-frame flicker check ===")
    for frame_i in range(3):
        result = fuse_track_detections(
            [make_track(1, track_box)],
            [phone_det, chit_det],
        )
        classes = [r.get("class") for r in result]
        print(f"Frame {frame_i}: assigned classes = {classes}")

    # Case 3: same two detections, but swap which one has higher confidence,
    # to check whether confidence or list-order decides the winner.
    print(f"\n=== Tiebreak check: does confidence or order decide the winner? ===")
    phone_det_lower_conf = ((150, 150, 200, 200), "phone", 0.60)
    chit_det_higher_conf = ((300, 300, 350, 350), "paper-chit", 0.95)
    result = fuse_track_detections(
        [make_track(1, (100, 100, 400, 400))],
        [phone_det_lower_conf, chit_det_higher_conf],
    )
    print(f"Result (chit now has higher confidence): {result}")

    # Case 4: same confidences as Case 3, but swap LIST ORDER instead of confidence,
    # to isolate whether the winner is decided by position in the list.
    print(f"\n=== Order check: does swapping list position change the winner? ===")
    result = fuse_track_detections(
        [make_track(1, (100, 100, 400, 400))],
        [chit_det_higher_conf, phone_det_lower_conf],  # chit now listed first
    )
    print(f"Result (chit listed first, same confidences as Case 3): {result}")


if __name__ == "__main__":
    main()