def explain(event: dict) -> str:
    obj = event.get("object_detected") or "none"
    return (f"Seat {event['seat_id']}: motion duration {event['duration']}s, "
            f"intensity {event.get('avg_motion_intensity', 0):.1f}, object detected: {obj}")

if __name__ == "__main__":
    from mock_event_data import generate_mock_events
    for e in generate_mock_events(3):
        print(explain(e))