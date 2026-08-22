from typing import Dict, Any, Optional
from fastapi import FastAPI, Query

app = FastAPI(title="Fake Backend")

_events: Dict[str, Any] = {}
_persons: Dict[str, Any] = {}
_completed: Dict[str, Any] = {}


@app.post("/internal/events")
async def events(payload: dict):
    event_id = payload.get("event_id")
    is_new = event_id not in _events
    if event_id is not None:
        _events[event_id] = payload
    return {"ok": True, "is_new": is_new, "stored_count": len(_events)}


@app.post("/internal/persons")
async def persons(payload: dict):
    person_id = payload.get("person_id")
    is_new = person_id not in _persons
    if person_id is not None:
        _persons[person_id] = payload
    return {"ok": True, "is_new": is_new, "stored_count": len(_persons)}


@app.post("/internal/complete/{video_id}")
async def complete(video_id: str, payload: dict):
    _completed[video_id] = payload
    return {"ok": True}


@app.get("/_debug/events")
async def debug_events(video_id: Optional[str] = Query(default=None)):
    if video_id is not None:
        filtered = [e for e in _events.values() if e.get("video_id") == video_id]
        return {"events": filtered}
    return {"events": list(_events.values())}


@app.get("/_debug/persons")
async def debug_persons(video_id: Optional[str] = Query(default=None)):
    if video_id is not None:
        filtered = [p for p in _persons.values() if p.get("video_id") == video_id]
        return {"persons": filtered}
    return {"persons": list(_persons.values())}


@app.get("/_debug/complete/{video_id}")
async def debug_complete(video_id: str):
    return _completed.get(video_id, {"status": "not_complete"})


@app.post("/_debug/reset")
async def debug_reset():
    _events.clear()
    _persons.clear()
    _completed.clear()
    return {"ok": True}