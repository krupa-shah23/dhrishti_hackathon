"""
Same as test_detect_webcam.py, but with cv2.imshow/waitKey removed —
gives a clean INFERENCE-ONLY FPS number, not inflated/deflated by
display rendering overhead. Use this number for the computational-
analysis slide; use the original test_detect_webcam.py when you want
to visually see detections happening live.

Run from the REPO ROOT:
    python -m src.track_det.test_detect_webcam_fps_only

Runs silently for the fixed duration below, then prints the summary.
No window will pop up — hold your phone up blind for the duration, or
just let it run on an empty scene to measure pure detection-loop speed
regardless of content.
"""
import time
import cv2

from .detector import detect_objects

DURATION_SECONDS = 20
CAMERA_INDEX = 0

# detect_objects() now needs exam_mode -- CBT allows both phone and
# paper-chit, the closest match to this script's original class-agnostic
# "detect anything" behavior.
EXAM_MODE = "CBT"


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}.")
        return

    print(f"Running for {DURATION_SECONDS} seconds, NO display window "
          f"(pure inference-loop timing). Hold your phone up if you "
          f"want detection-rate numbers too, or ignore this print and "
          f"let it run on whatever's in view.")

    start = time.time()
    frame_count = 0
    frames_with_detection = 0
    total_detections = 0
    max_conf_seen = 0.0

    while time.time() - start < DURATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam — stopping.")
            break

        frame_count += 1
        # Each live frame is wrapped as its own 1-crop window -- see the
        # comment in test_detect_real_footage.py / test_detect_webcam.py for
        # why (detect_objects() is frozen to take a window of crops, not a
        # single frame).
        detections = detect_objects([frame], EXAM_MODE)
        if detections:
            frames_with_detection += 1
            total_detections += len(detections)
            for det in detections:
                max_conf_seen = max(max_conf_seen, det["confidence"])

    cap.release()

    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    print(f"\n--- Summary (inference-only, no display overhead) ---")
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Pure inference FPS: {fps:.1f}")
    print(f"Frames with at least one detection: {frames_with_detection}")
    print(f"Total detections: {total_detections}")
    print(f"Highest confidence seen: {max_conf_seen:.3f}")
    print(
        "\nCompare this FPS against the earlier ~15.4 FPS measurement "
        "(which included cv2.imshow/waitKey display overhead). This "
        "number is the more honest one for the computational-analysis "
        "slide, since your real offline batch pipeline won't render a "
        "live window either."
    )


if __name__ == "__main__":
    main()