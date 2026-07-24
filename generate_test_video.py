"""
generate_test_video.py — Creates a synthetic exam-hall-like video for pipeline testing.
Simulates students as moving rectangles on a static desk background, with occasional
"high-motion" events to trigger risk detection.
"""

import cv2  # type: ignore[import-not-found]
import numpy as np
import os
import random
from typing import TypedDict


class _Student(TypedDict):
    cx: int
    cy: int
    w: int
    h: int
    vx: int
    vy: int
    color: tuple[int, int, int]

def generate_test_video(
    output_path: str = "data/demo_clips/raw/sample_exam.mp4",
    width: int = 1280,
    height: int = 720,
    fps: float = 25.0,
    duration_s: int = 30,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(fps * duration_s)

    # Desk-like static background
    background = np.full((height, width, 3), (200, 195, 180), dtype=np.uint8)
    # Draw desk lines
    for i in range(0, width, 200):
        cv2.line(background, (i, 0), (i, height), (180, 175, 160), 1)
    for j in range(0, height, 150):
        cv2.line(background, (0, j), (width, j), (180, 175, 160), 1)

    # Define 3 "students" as animated rectangles
    students: list[_Student] = [
        {"cx": 200, "cy": 200, "w": 80, "h": 50, "vx": 0, "vy": 0, "color": (60, 80, 150)},
        {"cx": 640, "cy": 360, "w": 80, "h": 50, "vx": 0, "vy": 0, "color": (80, 150, 60)},
        {"cx": 1050, "cy": 550, "w": 80, "h": 50, "vx": 0, "vy": 0, "color": (150, 80, 60)},
    ]

    rng = random.Random(42)

    for f in range(total_frames):
        frame = background.copy()

        # Every 3–6 seconds, trigger a "suspicious" high-motion event (fast velocity)
        for s in students:
            if f % int(fps * rng.uniform(3, 6)) == 0:
                s["vx"] = rng.randint(-12, 12)
                s["vy"] = rng.randint(-8, 8)
            else:
                # Dampen velocity
                s["vx"] = int(s["vx"] * 0.85)
                s["vy"] = int(s["vy"] * 0.85)

            s["cx"] = int(np.clip(s["cx"] + s["vx"], s["w"], width - s["w"]))
            s["cy"] = int(np.clip(s["cy"] + s["vy"], s["h"], height - s["h"]))

            x1 = s["cx"] - s["w"] // 2
            y1 = s["cy"] - s["h"] // 2
            x2 = s["cx"] + s["w"] // 2
            y2 = s["cy"] + s["h"] // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), s["color"], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)
            cv2.putText(frame, "Student", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        # Add timestamp overlay
        ts = f"Frame: {f:04d} | {f/fps:.2f}s"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

        writer.write(frame)

    writer.release()
    print(f"[generator] Synthetic test video saved -> {output_path}")
    print(f"[generator] Duration: {duration_s}s | {total_frames} frames @ {fps}fps")


if __name__ == "__main__":
    generate_test_video()
