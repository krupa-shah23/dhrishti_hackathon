"""
P2 module — Re-ID (cross-video person matching).

Design:
    - Appearance embedding via pretrained ResNet18 (torchvision), penultimate
      layer output (512-dim), NOT fine-tuned. No training needed to start.
    - PersonStore holds {person_id: embedding} and matches new track crops
      against it via cosine similarity, either reusing an existing person_id
      or minting a new one.
    - JSON-backed persistence so persons survive across process runs / videos,
      independent of whatever backend (Mongo, etc.) eventually owns this data.
      Swap _load/_save for a real DB call later without touching match logic.

Contract (mirrors the tracker.py / detector.py style already in this repo):
    extract_embedding(crop) -> np.ndarray, shape (512,), L2-normalized
    PersonStore.match_or_create(embedding, video_id, seat_id=None) -> person_id
    PersonStore.save() / PersonStore.load(path)

Usage (see integration notes at bottom of this file):
    store = PersonStore(path="data/persons_store.json", threshold=0.6)
    embedding = extract_embedding(crop)  # crop = person-box image, e.g. from track box
    person_id = store.match_or_create(embedding, video_id="clip_07", seat_id="14")
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# --- Embedding extractor (lazy singleton, mirrors detector.py's pattern) ---

_embedder = None
_transform = None

_DEVICE = "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"


def _load_embedder():
    """Loads pretrained ResNet18, strips the final FC layer so forward()
    returns the 512-dim pooled feature vector instead of class logits."""
    global _embedder, _transform
    if _embedder is not None:
        return

    if not _HAS_TORCH:
        print("[reid] torch/torchvision not available — extract_embedding() "
              "will raise until installed. Run: pip install torch torchvision")
        return

    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    base.fc = nn.Identity()  # strip classifier -> outputs 512-dim pooled features
    base.eval()
    base.to(_DEVICE)
    _embedder = base

    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),  # standard person-ReID aspect ratio (tall crop)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"[reid] Loaded ResNet18 embedding extractor on {_DEVICE}.")


def extract_embedding(crop: np.ndarray) -> np.ndarray:
    """
    crop: person-box image crop, HxWx3, RGB or BGR (doesn't matter much for
          a pretrained general-purpose net, but be CONSISTENT — if detector.py
          reads frames as BGR via cv2, keep this the same across your pipeline).
    returns: L2-normalized 512-dim embedding, np.ndarray shape (512,)
    """
    _load_embedder()
    if _embedder is None:
        raise RuntimeError("torch/torchvision not installed — cannot extract embeddings. "
                            "Run: pip install torch torchvision")

    tensor = _transform(crop).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        feat = _embedder(tensor).squeeze(0).cpu().numpy()

    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both embeddings are assumed pre-normalized (extract_embedding does this),
    so this is just a dot product — kept as a named function for clarity/tests."""
    return float(np.dot(a, b))


# --- Person store: matching + persistence ---

class PersonStore:
    """
    Holds known persons and their embeddings. Matches new embeddings against
    the store via cosine similarity; reuses person_id above `threshold`,
    otherwise mints a new one.

    IMPORTANT — threshold is the single most consequential tuning knob here:
        too low  -> false merges (two different people become one person_id)
        too high -> never links the same person across videos (defeats the point)
    Do NOT guess this value — tune it empirically (see test_reid.py's
    threshold sweep) against real cross-video footage of known people before
    trusting this in the actual pipeline. 0.6 below is a reasonable STARTING
    point for cosine similarity on L2-normalized ResNet features, not a
    validated final value.
    """

    def __init__(self, path: Optional[str] = None, threshold: float = 0.6):
        self.path = Path(path) if path else None
        self.threshold = threshold
        self.persons: Dict[str, Dict] = {}  # person_id -> {embedding, video_ids, seat_ids, thumbnail_path}
        self.match_log: List[Dict] = []  # every match decision, for threshold tuning/audit
        if self.path and self.path.exists():
            self.load(self.path)

    def match_or_create(
        self,
        embedding: np.ndarray,
        video_id: str,
        seat_id: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
    ) -> str:
        best_id, best_sim = None, -1.0
        for pid, data in self.persons.items():
            sim = cosine_similarity(embedding, np.array(data["embedding"], dtype=np.float32))
            if sim > best_sim:
                best_id, best_sim = pid, sim

        if best_id is not None and best_sim >= self.threshold:
            matched = True
            person_id = best_id
            # Running-average update: keeps the stored embedding representative
            # as more observations come in, rather than freezing on the first crop.
            old = np.array(self.persons[person_id]["embedding"], dtype=np.float32)
            updated = (old + embedding) / 2.0
            updated = updated / (np.linalg.norm(updated) + 1e-8)
            self.persons[person_id]["embedding"] = updated.tolist()
            if video_id not in self.persons[person_id]["video_ids"]:
                self.persons[person_id]["video_ids"].append(video_id)
        else:
            matched = False
            person_id = str(uuid.uuid4())[:8]
            self.persons[person_id] = {
                "embedding": embedding.tolist(),
                "video_ids": [video_id],
                "seat_ids": [],
                "thumbnail_path": thumbnail_path,
            }

        if seat_id and seat_id not in self.persons[person_id]["seat_ids"]:
            self.persons[person_id]["seat_ids"].append(seat_id)

        # Log EVERY decision (not just borderline ones) — cheap, and lets you
        # rebuild a full precision/recall picture later if ground truth becomes
        # available, without having re-run inference.
        self.match_log.append({
            "person_id": person_id,
            "video_id": video_id,
            "best_similarity": round(best_sim, 4),
            "matched_existing": matched,
            "threshold_used": self.threshold,
        })

        return person_id

    def get_cross_video_persons(self) -> Dict[str, List[str]]:
        """person_id -> list of video_ids they appear in. Directly answers
        the 'seen in other sessions' frontend feature (Master Doc §6.2)."""
        return {pid: data["video_ids"] for pid, data in self.persons.items()
                if len(data["video_ids"]) > 1}

    def get_borderline_matches(self, low: float = 0.0, high: float = None) -> List[Dict]:
        """Returns log entries near the threshold — the ones most worth
        manually eyeballing when tuning. Defaults to threshold +/- 0.1."""
        high = high if high is not None else self.threshold + 0.1
        low = max(low, self.threshold - 0.1)
        return [m for m in self.match_log if low <= m["best_similarity"] <= high]

    def save(self, path: Optional[str] = None):
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path set for PersonStore — pass one to save().")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump({"threshold": self.threshold, "persons": self.persons}, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.persons = data.get("persons", {})
        # Keep the threshold you explicitly constructed with unless you want
        # the saved file to override it — explicit param wins if passed.


# ---------------------------------------------------------------------------
# INTEGRATION NOTES — where this plugs into your existing pipeline
# ---------------------------------------------------------------------------
#
# In p1_p2_tracker.py, at the point a track is FINALIZED (i.e. right where
# P2P3Bridge._finalize_track() fires, or in your own track-lifecycle code):
#
#   from src.track_det.reid import extract_embedding, PersonStore
#   person_store = PersonStore(path="data/persons_store.json", threshold=0.6)
#
#   # crop = the person's bounding-box region from a representative frame
#   # (e.g. the frame with the largest/clearest box for that track)
#   embedding = extract_embedding(crop)
#   person_id = person_store.match_or_create(
#       embedding, video_id=video_id, seat_id=seat_id
#   )
#
#   # attach person_id onto the event dict before it reaches event_adapter.py
#   # (schemas.Person model already expects person_id, video_id, seat_id —
#   # matches directly)
#
# Call person_store.save() once per video after processing completes, so
# persons persist for the NEXT video's matching (this is what makes cross-
# video Re-ID actually work — a fresh PersonStore() every run would never
# find a match).