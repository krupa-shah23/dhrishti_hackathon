"""
Multi-frame test for fuse_track_detections() — checks that detection-to-
track attachment stays correct as BOTH the track and the detection move
together across frames, and correctly clears to None when no detection
overlaps a track.

fusion.py has so far only been verified with a single static frame
(one phone box fully inside one larger track box). This test drives it
frame-by-frame the way main.py actually will: track() called every
frame to get current track boxes, then fuse_track_detections() called
with those tracks + that frame's synthetic detections.

Run from the REPO ROOT:
    python -m src.track_det.test_fusion_sequence
"""

from .tracker import track, reset_tracker
from .fusion import fuse_track_detections


# One synthetic "person" (track box) moving right across 5 frames.
# Track box is deliberately larger than the phone box, matching the
# real scenario: a motion-ROI box (whole person/upper body) vs a tight
# YOLO phone box, which is why fusion.py uses containment, not IoU.
track_boxes_per_frame = [
    [(50, 50, 150, 250)],    # frame 0
    [(70, 50, 170, 250)],    # frame 1
    [(90, 50, 190, 250)],    # frame 2
    [(110, 50, 210, 250)],   # frame 3
    [(130, 50, 230, 250)],   # frame 4
]

# Synthetic phone detections. Deliberately includes:
# - frames where the phone box sits INSIDE the track box (should attach)
# - a frame with NO detection at all (should read back as None, not
#   carry over stale data from the previous frame)
# - a frame where the phone box is OUTSIDE any track box (should NOT
#   attach — tests that containment_thresh correctly rejects it, not
#   just that it accepts good matches)
detections_per_frame = [
    [((70, 100, 100, 140), "phone", 0.91)],   # frame 0: inside track box -> attach
    [((90, 100, 120, 140), "phone", 0.88)],   # frame 1: inside -> attach
    [],                                        # frame 2: no detection -> None
    [((400, 400, 430, 440), "phone", 0.95)],  # frame 3: nowhere near any track -> None
    [((150, 100, 180, 140), "phone", 0.93)],  # frame 4: inside -> attach
]


def main():
    reset_tracker()
    print("Frame-by-frame fusion results:")
    print("(class=None + confidence=None is CORRECT for frames 2 and 3 — "
          "that's the point of including them)\n")

    for i, (t_boxes, dets) in enumerate(zip(track_boxes_per_frame, detections_per_frame)):
        tracks = track(t_boxes)
        fused = fuse_track_detections(tracks, dets, containment_thresh=0.5)

        for f in fused:
            print(f"  Frame {i}: track_id={f['track_id']} box={f['box']} "
                  f"class={f['class']} confidence={f['confidence']}")

    print(
        "\nCheck: frame 0,1,4 should show class='phone' with the given "
        "confidence. Frame 2 should show class=None (no detection that "
        "frame at all). Frame 3 should ALSO show class=None even though "
        "a detection existed that frame — because it doesn't overlap "
        "the track box by the containment threshold. If frame 3 shows "
        "a phone attached, containment_thresh or the containment "
        "calculation is too permissive."
    )


if __name__ == "__main__":
    main()