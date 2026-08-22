"""
thumbnail_heatmap_writer.py

Writes the two media artifacts Backend expects at fixed disk paths and
returns the path strings to attach to the Event payload's thumbnail_path /
heatmap_ref fields (schemas.Event, both Optional[str] — untouched here).

Backend-confirmed storage convention: absolute local disk paths, no object
storage, served via Backend's own static file route, ONE FILE PER EVENT
(keyed by timestamp_start, not event_id -- Backend generates the MongoDB
_id only after the ML service has already written these files, so event_id
isn't available at write time):

    ml-service/outputs/thumbnails/<videoId>_t<timestamp_start>.jpg
    ml-service/outputs/heatmaps/<videoId>_t<timestamp_start>.png

e.g. abc123_t142.jpg = event starting at second 142 of video abc123.
timestamp_start is rounded to an int with int(round(...)) before going into
the filename -- Backend's example is "_t142", not "_t142.35".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("thumbnail_heatmap_writer")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THUMBNAIL_DIR = REPO_ROOT / "ml-service" / "outputs" / "thumbnails"
HEATMAP_DIR = REPO_ROOT / "ml-service" / "outputs" / "heatmaps"


def _dedupe_path(path: Path) -> Path:
    """
    If `path` already exists (two events in the same video rounded to the
    same integer start second -- plausible with fast-fragmenting tracks,
    e.g. clip 1's 202 micro-events averaging 3.68s apart), append _2, _3, ...
    to the stem rather than silently overwriting. Logs a warning so this is
    visible rather than silent.
    """
    if not path.exists():
        return path

    logger.warning(f"thumbnail_heatmap_writer: {path.name} already exists for this video/timestamp "
                   f"(two events rounded to the same start second) -- disambiguating with a suffix")
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def save_event_thumbnail(video_id: str, frame, timestamp_start: float, event_id: Optional[str] = None) -> Optional[str]:
    """
    Writes `frame` (BGR numpy array — the same convention frames already use
    everywhere else in this repo: cv2.VideoCapture/cv2.imread output, sliced
    directly into crops in p1_p2_tracker.py) as a JPEG to
    ml-service/outputs/thumbnails/<video_id>_t<timestamp_start>.jpg.

    One file per EVENT (per Backend's confirmed convention), keyed by the
    event's rounded start second rather than event_id, since event_id (a
    MongoDB _id) doesn't exist yet when the ML service writes this file.
    event_id is still accepted here for logging only. If two events in the
    same video round to the same start second, _dedupe_path() appends a
    _2/_3/... suffix and logs a warning instead of overwriting.

    Returns the absolute path string, or None if no usable frame was given
    (caller should leave thumbnail_path=None rather than write a broken file).
    """
    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return None

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _dedupe_path(THUMBNAIL_DIR / f"{video_id}_t{int(round(timestamp_start))}.jpg")

    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        return None
    return str(out_path.resolve())


def save_event_heatmap(video_id: str, heatmap_array, timestamp_start: float) -> Optional[str]:
    """
    Writes a 2D motion/activity accumulation array as a colorized PNG to
    ml-service/outputs/heatmaps/<video_id>_t<timestamp_start>.png.

    One file per EVENT, same timestamp_start-keyed convention as
    save_event_thumbnail() (see its docstring for why, and for the
    collision/dedupe behavior).

    STOPGAP, NOT THE REAL THING: no accumulated motion heatmap producer
    exists anywhere in this repo today. P1's MotionEstimator
    (src/motion/motion.py) only exposes a per-frame binary mask via
    get_motion_mask(frame) plus a scalar motion_intensity(mask) — nothing
    sums that across frames into a 2D map. `heatmap_array` here is expected
    to already be an accumulated 2D array (e.g. a running sum of per-frame
    motion masks over one event's frame range) built by the caller; this
    function only colorizes + writes whatever array it's given. It does not
    do the accumulating itself, and does not touch P1's motion code.
    Real heatmap generation belongs on P1's motion layer long-term — see
    report.

    Returns the absolute path string, or None if no usable array was given.
    """
    if heatmap_array is None or not hasattr(heatmap_array, "size") or heatmap_array.size == 0:
        return None

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _dedupe_path(HEATMAP_DIR / f"{video_id}_t{int(round(timestamp_start))}.png")

    arr = np.asarray(heatmap_array, dtype=np.float32)
    max_val = float(arr.max())
    normalized = (arr / max_val * 255.0) if max_val > 0 else arr
    colorized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

    ok = cv2.imwrite(str(out_path), colorized)
    if not ok:
        return None
    return str(out_path.resolve())
