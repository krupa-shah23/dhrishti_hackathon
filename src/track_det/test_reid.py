"""
Tests for src/track_det/reid.py.

Run: python -m src.track_det.test_reid
     pytest src/track_det/test_reid.py

Two kinds of tests here:
  1. Unit tests with synthetic crops — prove the matching LOGIC is correct
     (same synthetic crop matches itself, different synthetic crops don't,
     running-average update works, persistence round-trips).
  2. Threshold tuning sweep — a MANUAL step you run once against real crops
     of known people to actually pick your threshold. This is NOT optional
     before trusting Re-ID in the real pipeline; the unit tests below only
     prove the code works, not that 0.6 is the right number for your data.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from .reid import PersonStore, cosine_similarity, extract_embedding


def make_solid_color_crop(color, size=(128, 64, 3)):
    """Cheap synthetic 'person crop' — a solid color block. Not a real photo,
    but good enough to prove matching logic: same color -> high similarity,
    very different color -> lower similarity, deterministic and fast."""
    return np.full(size, color, dtype=np.uint8)


def test_same_crop_matches_itself():
    store = PersonStore(threshold=0.9)
    crop = make_solid_color_crop((200, 50, 50))
    emb1 = extract_embedding(crop)
    emb2 = extract_embedding(crop)  # same crop, extracted twice

    pid1 = store.match_or_create(emb1, video_id="v1")
    pid2 = store.match_or_create(emb2, video_id="v1")

    assert pid1 == pid2, "Identical crop should match itself, not create a new person"
    assert len(store.persons) == 1


def test_different_crops_create_different_persons():
    store = PersonStore(threshold=0.95)  # tight threshold for this test
    crop_a = make_solid_color_crop((200, 50, 50))   # reddish
    crop_b = make_solid_color_crop((50, 50, 200))   # bluish

    pid_a = store.match_or_create(extract_embedding(crop_a), video_id="v1")
    pid_b = store.match_or_create(extract_embedding(crop_b), video_id="v1")

    assert pid_a != pid_b, "Visually distinct crops should not merge into one person"
    assert len(store.persons) == 2


def test_cross_video_tracking():
    store = PersonStore(threshold=0.9)
    crop = make_solid_color_crop((100, 150, 100))

    pid_v1 = store.match_or_create(extract_embedding(crop), video_id="video_A")
    pid_v2 = store.match_or_create(extract_embedding(crop), video_id="video_B")

    assert pid_v1 == pid_v2
    cross_video = store.get_cross_video_persons()
    assert pid_v1 in cross_video
    assert set(cross_video[pid_v1]) == {"video_A", "video_B"}


def test_seat_id_tracking():
    store = PersonStore(threshold=0.9)
    crop = make_solid_color_crop((80, 80, 80))
    pid = store.match_or_create(extract_embedding(crop), video_id="v1", seat_id="14")
    assert "14" in store.persons[pid]["seat_ids"]


def test_persistence_round_trip():
    tmpdir = tempfile.mkdtemp()
    try:
        path = Path(tmpdir) / "persons.json"
        store = PersonStore(path=str(path), threshold=0.9)
        crop = make_solid_color_crop((120, 60, 200))
        pid = store.match_or_create(extract_embedding(crop), video_id="v1")
        store.save()

        store2 = PersonStore(path=str(path), threshold=0.9)
        assert pid in store2.persons
        assert store2.persons[pid]["video_ids"] == ["v1"]
    finally:
        shutil.rmtree(tmpdir)


def test_match_log_records_every_decision():
    store = PersonStore(threshold=0.9)
    crop = make_solid_color_crop((10, 200, 10))
    store.match_or_create(extract_embedding(crop), video_id="v1")
    store.match_or_create(extract_embedding(crop), video_id="v2")
    assert len(store.match_log) == 2
    assert all("best_similarity" in m for m in store.match_log)


def test_cosine_similarity_bounds():
    a = extract_embedding(make_solid_color_crop((255, 0, 0)))
    b = extract_embedding(make_solid_color_crop((255, 0, 0)))
    sim = cosine_similarity(a, b)
    assert -1.0 <= sim <= 1.0
    assert sim > 0.9  # near-identical inputs should be near-identical embeddings


# ---------------------------------------------------------------------------
# THRESHOLD TUNING SWEEP — run manually against REAL crops before trusting
# this in production. This is a script, not an assert-based test; it prints
# results for you to eyeball, matching the plan's "log borderline matches,
# tune empirically" requirement.
# ---------------------------------------------------------------------------

def threshold_sweep(crops_same_person: list, crops_different_people: list,
                     thresholds=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """
    crops_same_person: list of crops known to be the SAME real person
                        (e.g. multiple frames of one student across 2 videos)
    crops_different_people: list of crops known to be DIFFERENT real people

    For each threshold, reports:
      - how many same-person pairs correctly matched (want: high)
      - how many different-person pairs incorrectly matched (want: low, ideally 0)
    Pick the threshold that maximizes correct same-person matches while
    keeping false merges at/near zero — false merges are usually worse than
    missed matches for an investigation tool (better to show two person_ids
    for one real person than to wrongly conflate two different people).
    """
    same_embeddings = [extract_embedding(c) for c in crops_same_person]
    diff_embeddings = [extract_embedding(c) for c in crops_different_people]

    same_pair_sims = [
        cosine_similarity(same_embeddings[i], same_embeddings[j])
        for i in range(len(same_embeddings))
        for j in range(i + 1, len(same_embeddings))
    ]
    diff_pair_sims = [
        cosine_similarity(a, b) for a in same_embeddings for b in diff_embeddings
    ]

    print(f"\n{'Threshold':<10}{'Same-person matched':<25}{'Diff-person false merges':<25}")
    for t in thresholds:
        same_matched = sum(1 for s in same_pair_sims if s >= t)
        diff_merged = sum(1 for s in diff_pair_sims if s >= t)
        print(f"{t:<10}{same_matched}/{len(same_pair_sims):<24}{diff_merged}/{len(diff_pair_sims)}")

    print(f"\nRaw same-person similarities: {[round(s, 3) for s in same_pair_sims]}")
    print(f"Raw diff-person similarities: {[round(s, 3) for s in diff_pair_sims]}")
    print("\nPick the smallest threshold where diff-person false merges = 0, "
          "or the best trade-off if 0 is unreachable with this embedding model.")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
    print("\n--- Threshold sweep requires real crops; see docstring for usage ---")
    print("Example:")
    print("""
    from src.track_det.test_reid import threshold_sweep
    import cv2

    same_person_crops = [
        cv2.imread('data/debug_crops/student14_video1_frame200.jpg'),
        cv2.imread('data/debug_crops/student14_video2_frame050.jpg'),
    ]
    different_people_crops = [
        cv2.imread('data/debug_crops/student14_video1_frame200.jpg'),
        cv2.imread('data/debug_crops/student22_video1_frame300.jpg'),
    ]
    threshold_sweep(same_person_crops, different_people_crops)
    """)