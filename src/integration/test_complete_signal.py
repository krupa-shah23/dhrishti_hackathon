"""
test_complete_signal.py

Simulates POST request to /internal/complete/:video_id and asserts that the response
payloads for 'Complete' and 'Timeline' match the exact JSON schemas (field names and types)
defined in ML Master Doc §9.
"""

import unittest
from typing import Dict, Any, List

def mock_handle_complete_signal(video_id: str, events: List[Dict[str, Any]], processing_time_sec: float) -> Dict[str, Any]:
    """
    Simulates endpoint logic for POST /internal/complete/:video_id
    Returns dict containing 'complete' payload and 'timelines' payloads.
    """
    # Complete payload
    complete_payload = {
        "video_id": video_id,
        "status": "COMPLETED",
        "total_events": len(events),
        "processing_time_sec": float(processing_time_sec),
    }

    # Extract unique person_ids from events and generate timelines
    person_timelines = []
    person_ids = set()
    for ev in events:
        for p_id in ev.get("person_ids", []):
            person_ids.add(p_id)

    if not person_ids:
        # Default person timeline if no person_ids in events
        person_ids.add("person_01")

    for p_id in sorted(list(person_ids)):
        t_entries = []
        for ev in events:
            if p_id in ev.get("person_ids", []):
                for ts in ev.get("timestamps", []):
                    t_entries.append({
                        "timestamp": float(ts.get("start", 0.0)),
                        "confidence": float(ev.get("confidence", 1.0)),
                    })
        person_timelines.append({
            "person_id": str(p_id),
            "video_id": str(video_id),
            "timeline": t_entries,
        })

    return {
        "complete": complete_payload,
        "timelines": person_timelines,
    }


class TestCompleteSignalSchema(unittest.TestCase):

    def test_complete_and_timeline_schemas(self):
        video_id = "01_phone_use.mkv"
        dummy_events = [
            {
                "event_id": "ev_101",
                "person_ids": ["person_01"],
                "timestamps": [{"start": 12.0, "end": 16.0}],
                "confidence": 0.95,
            }
        ]

        response = mock_handle_complete_signal(video_id, dummy_events, processing_time_sec=4.5)

        # 1. Assert Complete Schema
        complete = response["complete"]
        expected_complete_fields = {
            "video_id": str,
            "status": str,
            "total_events": int,
            "processing_time_sec": (float, int),
        }
        for field, expected_type in expected_complete_fields.items():
            self.assertIn(field, complete, f"Missing field in Complete payload: {field}")
            self.assertIsInstance(
                complete[field],
                expected_type,
                f"Field '{field}' in Complete payload has wrong type: {type(complete[field])}, expected {expected_type}",
            )

        # 2. Assert Timeline Schema
        timelines = response["timelines"]
        self.assertIsInstance(timelines, list, "Timelines payload must be a list")
        self.assertGreater(len(timelines), 0, "Timelines list must not be empty")

        expected_timeline_fields = {
            "person_id": str,
            "video_id": str,
            "timeline": list,
        }
        for t_payload in timelines:
            for field, expected_type in expected_timeline_fields.items():
                self.assertIn(field, t_payload, f"Missing field in Timeline payload: {field}")
                self.assertIsInstance(
                    t_payload[field],
                    expected_type,
                    f"Field '{field}' in Timeline payload has wrong type: {type(t_payload[field])}",
                )

            # Assert inner timeline list entry schema
            for entry in t_payload["timeline"]:
                self.assertIn("timestamp", entry)
                self.assertIn("confidence", entry)
                self.assertIsInstance(entry["timestamp"], (float, int))
                self.assertIsInstance(entry["confidence"], (float, int))


if __name__ == "__main__":
    unittest.main()
