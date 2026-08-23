"""
extract_clip08_frames.py

Extracts ALL frames within each ground-truth event window (start->end)
for the new "06.Candidate was found using a mobile phone..." video
(clip08), instead of a single still at each labeled timestamp.

Run from repo root:
    dhrishti_hackathon> uv run python scripts/extract_clip08_frames.py

Requires ffmpeg on PATH (same as your other extraction scripts).
"""

import subprocess
from collections import defaultdict
from pathlib import Path

# --- CONFIRM THESE BEFORE RUNNING ---
VIDEO = Path("data/college_dataset/06.Candidate was found using a mobile phone in the examination hall.mp4")
OUTDIR = Path("data/frames_new/clip08_mobile_usage")

# Optional: cap extraction rate instead of pulling every native frame.
# None = extract every frame at the video's native fps (can be a LOT of
# images for longer events, e.g. a 24s event at 25fps = ~600 frames).
# Set e.g. 5 to sample at 5 frames/sec instead.
FPS_LIMIT = None
# --------------------------------------

# Same raw (label, timestamp) pairs as before -- only used to derive each
# event's start/end window. Labels are grouped by their evN prefix.
EVENTS = [
    ("ev1_start", "00:05:57"), ("ev1_end", "00:06:21"),
    ("ev2_start", "00:07:59"), ("ev2_end", "00:08:19"),
    ("ev3_start", "00:11:03"), ("ev3_end", "00:11:17"),
    ("ev4_start", "00:19:41"), ("ev4_end", "00:19:52"),
    ("ev5_start", "00:20:16"), ("ev5_end", "00:20:32"),
    ("ev6_start", "00:37:39"), ("ev6_mid", "00:38:20"), ("ev6_end", "00:39:03"),
    ("ev7_start", "00:43:23"), ("ev7_end", "00:43:33"),
    ("ev8_start", "01:21:15"), ("ev8_end", "01:21:34"),
    ("ev9_start", "01:23:42"), ("ev9_end", "01:23:49"),
    ("ev10_start", "01:24:07"), ("ev10_end", "01:24:14"),
    ("ev11_start", "01:24:41"), ("ev11_end", "01:24:50"),
    ("ev12_start", "01:25:07"), ("ev12_end", "01:25:11"),
    ("ev13_start", "01:25:24"), ("ev13_end", "01:25:27"),
    ("ev14_start", "01:25:47"), ("ev14_end", "01:25:55"),
    ("ev15_start", "01:26:13"), ("ev15_end", "01:26:19"),
    ("ev16_start", "01:26:38"), ("ev16_end", "01:26:44"),
    ("ev17_start", "01:26:56"), ("ev17_end", "01:27:00"),
    ("ev18_start", "01:27:03"), ("ev18_end", "01:27:07"),
    ("ev19_start", "01:27:50"), ("ev19_end", "01:28:08"),
    ("ev20_start", "01:28:13"), ("ev20_end", "01:28:45"),
    ("ev21_start", "01:28:45"), ("ev21_end", "01:28:49"),
    ("ev22_start", "01:29:11"), ("ev22_end", "01:29:14"),
    ("ev23_start", "01:29:24"), ("ev23_end", "01:29:40"),
    ("ev24_start", "01:29:40"), ("ev24_end", "01:30:00"),
    ("ev25_start", "01:30:06"), ("ev25_mid", "01:32:45"), ("ev25_end", "01:35:25"),
]


def parse_ts_to_seconds(ts: str) -> int:
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def group_events(events):
    """Group (label, ts) pairs by their evN prefix -> (start_ts, end_ts)."""
    groups = defaultdict(list)
    for label, ts in events:
        event_id = label.split("_")[0]  # "ev1", "ev6", ...
        groups[event_id].append(ts)

    windows = {}
    for event_id, timestamps in groups.items():
        # Fixed-width HH:MM:SS strings sort chronologically as strings too.
        windows[event_id] = (min(timestamps), max(timestamps))
    return windows


def extract_window(event_id: str, start_ts: str, end_ts: str):
    event_dir = OUTDIR / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    duration = parse_ts_to_seconds(end_ts) - parse_ts_to_seconds(start_ts)
    if duration <= 0:
        print(f"[SKIP] {event_id}: end <= start ({start_ts} -> {end_ts})")
        return

    out_pattern = event_dir / f"{event_id}_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-ss", start_ts,      # fast input seek to window start
        "-i", str(VIDEO),
        "-t", str(duration),  # pull the whole window, not a single frame
        "-q:v", "2",
    ]
    if FPS_LIMIT:
        cmd += ["-vf", f"fps={FPS_LIMIT}"]
    cmd += [str(out_pattern), "-y"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAILED] {event_id} @ {start_ts}-{end_ts}\n{result.stderr[-500:]}")
    else:
        n_frames = len(list(event_dir.glob(f"{event_id}_*.jpg")))
        print(f"[OK] {event_id} @ {start_ts}-{end_ts} -> {n_frames} frames in {event_dir}")


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found at {VIDEO.resolve()}. "
            "Check the filename/extension and that you're running from repo root."
        )
    OUTDIR.mkdir(parents=True, exist_ok=True)

    windows = group_events(EVENTS)
    for event_id in sorted(windows, key=lambda e: int(e[2:])):
        start_ts, end_ts = windows[event_id]
        extract_window(event_id, start_ts, end_ts)

    print(f"\nDone. Frames written under {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()