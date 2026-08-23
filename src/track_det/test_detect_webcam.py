"""
Tests detect_objects() on a live video STREAM (webcam) instead of a
saved video file — functionally identical code path to reading a real
.mp4 via cv2.VideoCapture, since a webcam is just another VideoCapture
source. This is the "Remaining: Video Inference" item flagged in the
original setup summary — detect_objects() has so far only been run on
static test IMAGES, never on a frame-by-frame video stream.

What this checks that image-only testing couldn't:
- Does detect_objects() handle the frame format cv2.VideoCapture.read()
  actually produces (BGR numpy array), across MANY frames in a row,
  without crashing or slowing down unacceptably?
- Does detection confidence/behavior look reasonable on live, slightly
  blurry, real-lighting frames vs. the clean static test images from
  Phase 19?

Run from the REPO ROOT:
    python -m src.track_det.test_detect_webcam

Hold your phone up to the webcam for a few seconds after it starts.
Press 'q' in the video window to stop early, or let it run for the
fixed duration below.

If no webcam is available (e.g. running over SSH with no camera),
this will fail fast with a clear error rather than hanging.
"""
import time
import cv2

from .detector import detect_objects

DURATION_SECONDS = 45

CAMERA_INDEX = 0  # change to 1, 2... if you have multiple cameras


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}. "
              f"If running over SSH/remote, no webcam is accessible this way.")
        return

    print(f"Webcam opened. Running for {DURATION_SECONDS} seconds — "
          f"hold your phone up to the camera. Press 'q' to stop early.")

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
        detections = detect_objects(frame)

        if detections:
            frames_with_detection += 1
            total_detections += len(detections)
            for box, cls_name, conf in detections:
                max_conf_seen = max(max_conf_seen, conf)
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("detect_objects() live test — press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"Frames processed: {frame_count}")
    print(f"Effective FPS (detection + display included): {fps:.1f}")
    print(f"Frames with at least one detection: {frames_with_detection}")
    print(f"Total detections across all frames: {total_detections}")
    print(f"Highest confidence seen: {max_conf_seen:.3f}")

    if frames_with_detection == 0:
        print(
            "\nNo detections at all — either the phone wasn't held up to "
            "the camera in view, or worth checking detect_objects() is "
            "actually loading the weights (look for the earlier "
            "'[detector] Loaded detector weights: ...' startup message)."
        )
    else:
        print(
            "\nCheck: does the FPS number look usable for your planned "
            "pipeline speed (real exam videos are hours long — this "
            "tells you roughly how long full-video inference will take)? "
            "Does confidence look reasonably stable across frames, or "
            "does it flicker a lot frame-to-frame for the same held-up "
            "phone (flickering would suggest video-frame conditions are "
            "meaningfully harder than the clean static test images)?"
        )


if __name__ == "__main__":
    main()