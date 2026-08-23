"""
calibration_tool.py — P4's grid/ROI calibration UI (minimal, click-to-tag version)

Lets you grab a still frame from a real clip, draw a box around each seat/desk
region by click-drag, and type the seat_id for that box in the terminal.
Saves to JSON in the frozen format both P1 and P2 (ground truth) depend on:

{
  "camera_id": "cam_01",
  "grid_cells": [
    {"seat_id": "seat_01", "bbox": [x1, y1, x2, y2]},
    {"seat_id": "seat_12", "bbox": [x1, y1, x2, y2]}
  ]
}

Usage:
    python calibration_tool.py --video path/to/06_phone_use.mp4 --camera-id cam_clip7 \
        --frame-time 30 --out config/cam_clip7_grid.json

Sanity check to run first: use clip 7 (06_phone_use.mp4) at any timestamp between
26s-1m22s. You already know that region is seat_12 — if your box lands there and
you label it seat_12, the tool + config format are proven correct before you touch
any other clip.

Controls (once the window opens):
    Left-click and drag  -> draw a box
    (release mouse)       -> terminal prompts for seat_id, type it + Enter
    's'                    -> save progress to --out without quitting
    'u'                    -> undo the last box
    Right arrow / 'l'      -> jump forward 5s (search for a clearer frame)
    Left arrow  / 'j'      -> jump back 5s
    Up arrow    / 'i'      -> jump forward 1s (fine search)
    Down arrow  / 'k'      -> jump back 1s
    'q'                    -> save and quit
"""
import cv2
import json
import argparse
import os

drawing = False
ix, iy = -1, -1
boxes = []
frame = None
display = None
cap = None
fps = 25
current_time = 0.0


def redraw():
    global display
    display = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, b["seat_id"], (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, display
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        redraw()
        cv2.rectangle(display, (ix, iy), (x, y), (0, 200, 255), 1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if x2 - x1 < 5 or y2 - y1 < 5:
            print("Box too small, ignored.")
            return
        seat_id = input(f"Seat ID for box ({x1},{y1})-({x2},{y2}): ").strip()
        if not seat_id:
            print("Empty seat_id, box discarded.")
            return
        boxes.append({"seat_id": seat_id, "bbox": [x1, y1, x2, y2]})
        redraw()


def load_existing(out_path):
    """Load previously-saved boxes if --out already exists, so restarting the
    script never silently wipes prior work."""
    global boxes
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            boxes = existing.get("grid_cells", [])
            print(f"Loaded {len(boxes)} existing region(s) from {out_path}: "
                  f"{[b['seat_id'] for b in boxes]}")
        except Exception as e:
            print(f"WARNING: could not load existing {out_path} ({e}). "
                  f"Starting fresh — check this file isn't corrupted before saving over it.")
    else:
        print(f"No existing file at {out_path} — starting fresh.")


def save(camera_id, out_path):
    config = {"camera_id": camera_id, "grid_cells": boxes}
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {len(boxes)} regions to {out_path}")


def seek_to(time_sec):
    """Re-grab a frame at time_sec, clamped to >= 0. Returns True if successful."""
    global frame, display, current_time
    time_sec = max(0.0, time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
    ret, new_frame = cap.read()
    if not ret:
        print(f"Could not read frame at {time_sec:.1f}s (may be past end of video).")
        return False
    frame = new_frame
    current_time = time_sec
    redraw()
    print(f"Now at {current_time:.1f}s")
    return True


def main():
    global frame, display, cap, fps, current_time
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frame-time", type=float, default=5.0,
                         help="seconds into the video to grab the initial still frame from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    load_existing(args.out)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    current_time = args.frame_time
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame_time * fps))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read a frame at {args.frame_time}s from {args.video}")

    display = frame.copy()
    redraw()  # draw any previously-loaded boxes immediately on the initial frame
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("Draw boxes by click-drag. Type seat_id in the terminal after each box.")
    print("'s' = save, 'u' = undo, arrows/i-j-k-l = scrub, 'q' = save and quit.")
    print(f"Starting at {current_time:.1f}s")

    while True:
        cv2.imshow("Calibration", display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            save(args.camera_id, args.out)
        elif key == ord('u'):
            if boxes:
                removed = boxes.pop()
                print(f"Removed {removed['seat_id']}")
                redraw()
        elif key in (83, ord('l')):        # right arrow / l -> forward 5s
            seek_to(current_time + 5)
        elif key in (81, ord('j')):        # left arrow / j -> back 5s
            seek_to(current_time - 5)
        elif key in (82, ord('i')):        # up arrow / i -> forward 1s
            seek_to(current_time + 1)
        elif key in (84, ord('k')):        # down arrow / k -> back 1s
            seek_to(current_time - 1)
        elif key == ord('q'):
            save(args.camera_id, args.out)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()