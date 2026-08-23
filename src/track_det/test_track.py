"""
Quick sanity test for track() using synthetic boxes — two "people"
moving across frames, one static "person" to check no spurious ID churn.

Run from the REPO ROOT (not from inside src/track_det/):
    python -m src.track_det.test_track

Expected: 3 stable track IDs, one per synthetic subject, across all
5 frames — matches your Day 2 "Check result": same person keeps the
same ID across a clip.
"""

from .tracker import track, reset_tracker

# Simulate 5 frames, 3 moving/static boxes each (x1, y1, x2, y2)
synthetic_frames = [
    [(10, 10, 40, 60), (200, 200, 230, 260), (400, 50, 430, 100)],
    [(15, 12, 45, 62), (205, 202, 235, 262), (400, 50, 430, 100)],
    [(20, 14, 50, 64), (210, 205, 240, 265), (400, 50, 430, 100)],
    [(25, 16, 55, 66), (215, 207, 245, 267), (400, 50, 430, 100)],
    [(30, 18, 60, 68), (220, 210, 250, 270), (400, 50, 430, 100)],
]

if __name__ == "__main__":
    for i, boxes in enumerate(synthetic_frames):
        tracks = track(boxes)
        ids = sorted(t["track_id"] for t in tracks)
        print(f"Frame {i}: {len(tracks)} tracks, IDs={ids}")

    print("\nCheck: IDs should stay the same set across all 5 frames "
          "(e.g. [1, 2, 3] every time) — no new IDs spawning for the "
          "same moving subject.")
