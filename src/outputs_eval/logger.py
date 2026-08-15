"""
logger.py — P4: Outputs, Evaluation & Integration
CSV event logger that writes structured detection events to disk.
Schema: timestamp, frame_index, roi_box, motion_score, audio_level, risk_score, confidence
"""

import csv
import os
from pathlib import Path

# Canonical CSV column order (frozen — matches docs/function_contracts.md schema)
CSV_FIELDNAMES = [
    "timestamp",
    "frame_index",
    "roi_box",
    "motion_score",
    "audio_level",
    "risk_score",
    "confidence",
]


def init_log(csv_path: str) -> None:
    """
    Creates a fresh CSV log file with the header row.
    Overwrites any existing file at the given path.

    Args:
        csv_path (str): Full output file path (e.g., 'outputs/video_events.csv').
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
    print(f"[logger] Initialized log -> {csv_path}")


def log_event(csv_path: str, event_data: dict) -> None:
    """
    Appends a single event record to the CSV log file.

    Args:
        csv_path (str): Full output file path.
        event_data (dict): Dict with keys matching CSV_FIELDNAMES:
            - timestamp   (str)   : HH:MM:SS.mmm
            - frame_index (int)   : Frame number
            - roi_box     (str)   : "(x,y,w,h)" formatted string
            - motion_score(float) : Motion intensity 0.0–100.0
            - audio_level (float) : Audio energy / dB level
            - risk_score  (float) : Behavioral risk 0.0–1.0
            - confidence  (float) : Overall event confidence 0.0–1.0
    """
    # Gracefully fill any missing optional fields
    row = {field: event_data.get(field, "") for field in CSV_FIELDNAMES}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def format_timestamp(frame_index: int, fps: float) -> str:
    """
    Converts a frame index to a HH:MM:SS.mmm timestamp string.

    Args:
        frame_index (int): Current frame number.
        fps (float):       Frames per second of the source video.

    Returns:
        str: Timestamp string in HH:MM:SS.mmm format.
    """
    total_seconds = frame_index / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def format_roi_box(box: tuple[int, int, int, int]) -> str:
    """
    Formats a (x, y, w, h) box tuple as a string for CSV storage.

    Args:
        box (tuple): Bounding box (x, y, w, h).

    Returns:
        str: Formatted string '(x,y,w,h)'.
    """
    return f"({box[0]},{box[1]},{box[2]},{box[3]})"
