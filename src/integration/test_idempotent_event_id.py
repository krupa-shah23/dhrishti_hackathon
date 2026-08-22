"""
test_idempotent_event_id.py

Tests event idempotency by posting the same event_id twice with different payloads,
asserting that the backend/store performs an upsert where the latest payload for the event_id wins
and no duplicate event_id entries exist.
"""

import unittest
from typing import Dict, Any

class MockEventStore:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def post_event(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates HTTP POST /internal/event.
        Upserts payload keyed by event_id.
        """
        event_id = event_payload.get("event_id")
        if not event_id:
            raise ValueError("Event payload missing required 'event_id'")
        
        # Upsert: latest payload for event_id overwrites existing payload
        self.store[event_id] = dict(event_payload)
        return {"status": "SUCCESS", "event_id": event_id, "upserted": True}

    def get_event(self, event_id: str) -> Dict[str, Any]:
        return self.store.get(event_id)

    def get_all_events(self) -> list:
        return list(self.store.values())


class TestIdempotentEventID(unittest.TestCase):

    def setUp(self):
        self.event_store = MockEventStore()

    def test_idempotent_upsert_same_event_id(self):
        event_id = "ev_test_001"

        payload_v1 = {
            "event_id": event_id,
            "video_id": "01_phone_use.mkv",
            "camera_id": "Camera04",
            "seat_id": "seat_01",
            "person_ids": ["person_01"],
            "timestamps": [{"start": 10.0, "end": 14.0}],
            "activities": ["phone_use"],
            "avg_motion_intensity": 0.15,
            "peak_intensity": 0.35,
            "mog2_foreground_ratio": 0.20,
            "duration": 4.0,
            "repetition_count": 1,
            "object_detected": True,
            "object_confidence": 0.85,
            "invigilator_excluded": False,
            "exam_phase": "active",
            "severity_score": 65.0,
            "risk_label": "MEDIUM",
            "explanation": "Initial detection",
            "confidence": 0.85,
            "color_tag": "YELLOW",
            "thumbnail_path": None,
            "bbox_overlay": [],
            "heatmap_ref": None,
            "related_event_id": None,
            "event_type": "phone_use",
        }

        payload_v2 = {
            "event_id": event_id,  # Same event_id
            "video_id": "01_phone_use.mkv",
            "camera_id": "Camera04",
            "seat_id": "seat_01",
            "person_ids": ["person_01"],
            "timestamps": [{"start": 10.0, "end": 16.0}],  # Updated timestamp end
            "activities": ["phone_use", "looking_down"],  # Updated activity
            "avg_motion_intensity": 0.25,
            "peak_intensity": 0.50,
            "mog2_foreground_ratio": 0.30,
            "duration": 6.0,
            "repetition_count": 2,
            "object_detected": True,
            "object_confidence": 0.92,  # Updated confidence
            "invigilator_excluded": False,
            "exam_phase": "active",
            "severity_score": 85.0,  # Updated severity
            "risk_label": "HIGH",  # Updated risk label
            "explanation": "Confirmed persistent phone usage",
            "confidence": 0.92,
            "color_tag": "RED",
            "thumbnail_path": "thumbnails/ev_test_001.jpg",
            "bbox_overlay": [],
            "heatmap_ref": None,
            "related_event_id": None,
            "event_type": "phone_use",
        }

        # 1. First POST
        res1 = self.event_store.post_event(payload_v1)
        self.assertEqual(res1["status"], "SUCCESS")
        self.assertEqual(len(self.event_store.get_all_events()), 1)
        stored_ev1 = self.event_store.get_event(event_id)
        self.assertEqual(stored_ev1["severity_score"], 65.0)

        # 2. Second POST with same event_id and updated payload
        res2 = self.event_store.post_event(payload_v2)
        self.assertEqual(res2["status"], "SUCCESS")

        # 3. Assert upsert behavior: count remains 1 (no duplicates) and latest payload won
        all_events = self.event_store.get_all_events()
        self.assertEqual(len(all_events), 1, "Duplicate event_ids created after second POST!")
        
        stored_ev2 = self.event_store.get_event(event_id)
        self.assertEqual(stored_ev2["severity_score"], 85.0, "Latest payload did not overwrite previous payload!")
        self.assertEqual(stored_ev2["risk_label"], "HIGH")
        self.assertEqual(stored_ev2["confidence"], 0.92)


if __name__ == "__main__":
    unittest.main()
