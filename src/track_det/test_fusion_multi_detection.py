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
from .fusion import fuse_track_detections, merge_invigilator_flags
from .invigilator_filter import is_invigilator_track
from ..integration.p2_p3_bridge import P2P3Bridge


def make_track(track_id, box, **kwargs):
    t = {"track_id": track_id, "box": box}
    t.update(kwargs)
    return t


def make_seated_student_history(num_frames: int = 20):
    history = []
    cx, cy = 300, 300
    for i in range(num_frames):
        offset = 15 if i % 2 == 0 else -15
        x1, y1 = cx + offset, cy + offset
        history.append((x1, y1, x1 + 30, y1 + 30))
    return history


def make_invigilator_history(num_frames: int = 20):
    history = []
    for i in range(num_frames):
        x1 = 50 + i * 20
        y1 = 100
        history.append((x1, y1, x1 + 30, y1 + 60))
    return history


def run_case(name, tracks, detections):
    print(f"\n=== {name} ===")
    print(f"Tracks: {tracks}")
    print(f"Detections: {detections}")
    result = fuse_track_detections(tracks, detections)
    print(f"Result: {result}")
    return result


def test_invigilator_flag_end_to_end():
    """
    Proves invigilator_flag flows end-to-end from:
    invigilator_filter -> fuse_track_detections -> P2P3Bridge -> event's is_invigilator field.
    """
    student_hist = make_seated_student_history(20)
    invigilator_hist = make_invigilator_history(20)

    # 1. Verify invigilator_filter evaluates them correctly
    assert not is_invigilator_track(student_hist)
    assert is_invigilator_track(invigilator_hist)

    # 2. Test fusion with track_histories dict (Precedence 2)
    tracks = [
        make_track(1, (100, 100, 200, 200)),
        make_track(2, (400, 100, 500, 200)),
    ]
    track_histories = {1: student_hist, 2: invigilator_hist}
    detections = [((110, 110, 150, 150), "phone", 0.9)]

    fused = fuse_track_detections(tracks, detections, track_histories=track_histories)
    assert len(fused) == 2
    assert fused[0]["track_id"] == 1
    assert fused[0]["invigilator_flag"] is False
    assert fused[0]["class"] == "phone"
    assert fused[1]["track_id"] == 2
    assert fused[1]["invigilator_flag"] is True

    # 3. Test P2P3Bridge receives and processes fused tracks
    bridge = P2P3Bridge(fps=10.0)
    bridge.process_fused_tracks(fused, frame_index=0)
    bridge.flush()

    events = bridge.get_completed_events()
    assert len(events) == 2

    event_by_id = {e["track_id"]: e for e in events}
    assert event_by_id[1]["is_invigilator"] is False
    assert event_by_id[1]["object_detected"] is True
    assert event_by_id[2]["is_invigilator"] is True

    # 4. Test Precedence 1: Explicit invigilator_flags dict override
    fused_override = fuse_track_detections(
        tracks,
        [],
        track_histories=track_histories,
        invigilator_flags={1: True, 2: False},
    )
    assert fused_override[0]["invigilator_flag"] is True
    assert fused_override[1]["invigilator_flag"] is False

    # 5. Test Precedence 3: Pre-existing flag on track dict
    tracks_preflagged = [
        make_track(3, (50, 50, 100, 100), invigilator_flag=True),
        make_track(4, (100, 100, 150, 150)),
    ]
    fused_preflagged = fuse_track_detections(tracks_preflagged, [])
    assert fused_preflagged[0]["invigilator_flag"] is True
    assert fused_preflagged[1]["invigilator_flag"] is False


def test_merge_invigilator_flags_helper():
    fused = [
        {"track_id": 1, "box": (0, 0, 10, 10), "class": None, "confidence": None},
        {"track_id": 2, "box": (20, 20, 30, 30), "class": None, "confidence": None},
    ]
    track_histories = {
        1: make_seated_student_history(20),
        2: make_invigilator_history(20),
    }
    merged = merge_invigilator_flags(fused, track_histories=track_histories)
    assert merged[0]["invigilator_flag"] is False
    assert merged[1]["invigilator_flag"] is True


def main():
    # Run unit/integration test
    test_invigilator_flag_end_to_end()
    test_merge_invigilator_flags_helper()
    print("=== End-to-end invigilator_flag test passed successfully! ===")

    # Case 1: one large track box, two legitimate detections inside it
    # (phone in one hand, chit-like paper in the other), same frame.
    track_box = (100, 100, 400, 400)
    phone_det = ((150, 150, 200, 200), "phone", 0.90)
    chit_det = ((300, 300, 350, 350), "chit", 0.85)

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
    chit_det_higher_conf = ((300, 300, 350, 350), "chit", 0.95)
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