# ============================================================
# checkpoint_event.py
# ============================================================
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    video_id TEXT,
    camera_id TEXT,
    seat_id TEXT,
    start_time REAL,
    end_time REAL,
    grid_cells TEXT,
    avg_motion_intensity REAL,
    peak_intensity REAL,
    mog2_foreground_ratio REAL,
    duration REAL,
    repetition_count INTEGER,
    object_detected TEXT,
    object_confidence REAL,
    invigilator_excluded INTEGER,
    exam_phase TEXT,
    exam_mode TEXT,
    severity_score REAL,
    risk_label TEXT,
    explanation TEXT,
    audio_corroborated INTEGER,
    intervention_detected INTEGER,
    related_event_id TEXT,
    event_type TEXT
);
"""

FIELDS = [
    "event_id", "video_id", "camera_id", "seat_id", "start_time", "end_time",
    "grid_cells", "avg_motion_intensity", "peak_intensity", "mog2_foreground_ratio",
    "duration", "repetition_count", "object_detected", "object_confidence",
    "invigilator_excluded", "exam_phase", "exam_mode", "severity_score",
    "risk_label", "explanation", "audio_corroborated", "intervention_detected",
    "related_event_id", "event_type",
]

def get_connection(db_path="events.db"):
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn

def checkpoint_event(event, db_conn):
    """Incrementally write/update one event. None -> NULL, lists -> comma string."""
    row = []
    for f in FIELDS:
        v = event.get(f)
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        if isinstance(v, bool):
            v = int(v)
        row.append(v)

    placeholders = ",".join(["?"] * len(FIELDS))
    cols = ",".join(FIELDS)
    db_conn.execute(
        f"INSERT OR REPLACE INTO events ({cols}) VALUES ({placeholders})", row
    )
    db_conn.commit()
    return None


if __name__ == "__main__":
    from mock_event_data import generate_mock_events
    conn = get_connection("test.db")
    events = generate_mock_events(5)
    for e in events:
        checkpoint_event(e, conn)
    cur = conn.execute("SELECT event_id, seat_id, start_time FROM events")
    for r in cur.fetchall():
        print(r)
    conn.close()