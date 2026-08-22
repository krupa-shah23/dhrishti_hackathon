import os
import asyncio
import logging
import httpx
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from schemas import Event, Person, CompleteSignal
from mock_data import generate_mock_events, generate_mock_person

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_bridge")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
USE_MOCK = os.environ.get("USE_MOCK", "true").lower() == "true"

app = FastAPI(title="Drishti P2b FastAPI Bridge")

# in-memory tracking for idempotency/testing/debugging
_posted_event_ids: set[str] = set()


class ProcessRequest(BaseModel):
    video_id: str
    path: str


async def post_event(client: httpx.AsyncClient, event: Event):
    try:
        resp = await client.post(f"{BACKEND_URL}/internal/events", json=event.model_dump())
        resp.raise_for_status()
        _posted_event_ids.add(event.event_id)
        logger.info(f"Posted event {event.event_id} ({event.event_type}) for {event.video_id}")
    except Exception as e:
        logger.error(f"Failed to post event {event.event_id}: {e}")
        raise


async def post_person(client: httpx.AsyncClient, person: Person):
    try:
        resp = await client.post(f"{BACKEND_URL}/internal/persons", json=person.model_dump())
        resp.raise_for_status()
        logger.info(f"Posted person {person.person_id} for {person.video_id}")
    except Exception as e:
        logger.error(f"Failed to post person {person.person_id}: {e}")


async def post_complete(client: httpx.AsyncClient, video_id: str, total_events: int):
    signal = CompleteSignal(video_id=video_id, total_events=total_events)
    try:
        resp = await client.post(f"{BACKEND_URL}/internal/complete/{video_id}", json=signal.model_dump())
        resp.raise_for_status()
        logger.info(f"Marked {video_id} complete ({total_events} events)")
    except Exception as e:
        logger.error(f"Failed to post complete signal for {video_id}: {e}")


async def run_pipeline(video_id: str, path: str):
    """
    Runs the ML pipeline and posts events incrementally as they're produced.
    Swap generate_mock_events() for the real pipeline generator once live —
    endpoint contracts below shouldn't need to change.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        person = generate_mock_person(video_id) if USE_MOCK else None
        if person:
            await post_person(client, person)

        events = generate_mock_events(video_id) if USE_MOCK else []
        # TODO: replace with real pipeline call, e.g.:
        # async for event in real_pipeline.stream_events(video_id, path):
        for event in events:
            await post_event(client, event)
            await asyncio.sleep(0.1)  # simulate incremental production

        await post_complete(client, video_id, len(events))


@app.post("/process")
async def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline, req.video_id, req.path)
    return {"status": "accepted", "video_id": req.video_id}


@app.get("/health")
async def health():
    return {"status": "ok", "backend_url": BACKEND_URL, "mock_mode": USE_MOCK}


@app.get("/_debug/posted_events")
async def debug_posted_events():
    """Test-only introspection endpoint — remove before final integration."""
    return {"count": len(_posted_event_ids), "event_ids": list(_posted_event_ids)}