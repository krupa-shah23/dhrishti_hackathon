import os
import sys
import asyncio
import logging
from pathlib import Path
import httpx
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from schemas import Event, Person, CompleteSignal
from mock_data import generate_mock_events, generate_mock_person

# Repo root (fastapi_bridge/ -> fastapi_bridge -> integration -> src -> repo root)
# needed on sys.path to import the real src.* pipeline modules below, since
# this app is launched as `uvicorn main:app` from within fastapi_bridge/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_bridge")

# BACKEND_WEBHOOK_URL is the primary/documented name; BACKEND_URL kept as a
# fallback alias for existing test/tooling that still sets it. Default is
# unchanged (localhost:5000) so existing tests don't break.
BACKEND_URL = os.environ.get(
    "BACKEND_WEBHOOK_URL", os.environ.get("BACKEND_URL", "http://localhost:5000")
)
USE_MOCK = os.environ.get("USE_MOCK", "true").lower() == "true"

app = FastAPI(title="Drishti P2b FastAPI Bridge")

# in-memory tracking for idempotency/testing/debugging
_posted_event_ids: set[str] = set()

# Fix 4 -- per-video_id in-flight guard. Plain in-memory set is sufficient
# for this single-process scope (matches _posted_event_ids' existing pattern
# above); a duplicate /process on an already-running video_id is rejected
# rather than risking two concurrent runs upserting onto the same
# mlEventId documents (mlEventId is "{video_id}_event_{track_id}", not
# per-run-unique, so two overlapping runs WOULD collide).
_in_progress_video_ids: set[str] = set()


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


async def post_failed(client: httpx.AsyncClient, video_id: str, error_message: str):
    """
    Fix 3: previously a pipeline failure just logged server-side and left
    the video stuck 'queued' forever with the client having already gotten
    a false 200 'accepted'. Reuses /internal/complete/:video_id (extended
    with status='failed') rather than adding a whole new route, since it's
    already "the pipeline is done, here's the outcome" -- see
    src/routes/internal.js.
    """
    try:
        resp = await client.post(
            f"{BACKEND_URL}/internal/complete/{video_id}",
            json={"status": "failed", "error_message": error_message},
        )
        resp.raise_for_status()
        logger.info(f"Marked {video_id} failed: {error_message}")
    except Exception as e:
        logger.error(f"Failed to post failure signal for {video_id}: {e}")


async def post_node_event(client: httpx.AsyncClient, payload: dict):
    """
    Fix 2: posts the real Node-shape payload (mlEventId/videoId/seatId/
    objectDetected/objectConfidence/timestamps/bboxOverlay) built by
    adapt_bridge_event_to_node_payload -- NOT schemas.Event.model_dump()
    (event_id/video_id/start/end/...), which is what this used to send and
    is why real events 400'd against Node's actual /internal/events
    contract. post_event() above is unchanged/still used for the USE_MOCK
    path against fake_backend.py, which doesn't enforce Node's real shape.
    """
    try:
        resp = await client.post(f"{BACKEND_URL}/internal/events", json=payload)
        resp.raise_for_status()
        _posted_event_ids.add(payload["mlEventId"])
        logger.info(f"Posted event {payload['mlEventId']} for {payload['videoId']}")
    except Exception as e:
        logger.error(f"Failed to post event {payload.get('mlEventId')}: {e}")
        raise


def _run_real_pipeline_sync(video_id: str, path: str) -> list[dict]:
    """
    Runs the real P1xP2 pipeline (motion -> track -> window-detect -> fuse ->
    severity/motion enrichment) over the whole clip and returns Node-shape
    event payload dicts (Fix 2: via adapt_bridge_event_to_node_payload, not
    the old adapt_bridge_event_to_schema -- that produced schemas.Event's
    internal shape, which Node's real /internal/events rejects with 400).

    Synchronous/CPU-bound by nature (cv2 + YOLO + mediapipe) -- callers MUST
    run this via loop.run_in_executor (Fix 1), never awaited directly in the
    event loop, or it blocks the ASGI response flush for the request that
    triggered it (confirmed live: a 150-frame clip made /process itself take
    >100s before the client even got its "accepted" response).
    """
    import cv2
    from src.integration.p1_p2_tracker import P1P2TrackerPipeline
    from src.integration.p2_p3_bridge import P2P3Bridge
    from src.integration.event_clustering import enrich_event_with_motion_fields
    from src.integration.event_adapter import adapt_bridge_event_to_node_payload

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pipeline = P1P2TrackerPipeline(clip_name=Path(path).name, fps=fps)
    bridge = P2P3Bridge(missing_threshold=15, fps=fps)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, frame_index)
        bridge.process_fused_tracks(
            result["fused_tracks"],
            frame_index=frame_index,
            pose_signals=result.get("pose_signals"),
            motion_intensity=result.get("motion_intensity"),
            mog2_foreground_ratio=result.get("mog2_foreground_ratio"),
        )
        frame_index += 1
    cap.release()
    bridge.flush()

    return [
        adapt_bridge_event_to_node_payload(enrich_event_with_motion_fields(ev), video_id=video_id)
        for ev in bridge.completed_events
    ]


async def run_pipeline(video_id: str, path: str):
    """
    Runs the ML pipeline and posts events after they're produced.

    Fix 1 (was BackgroundTasks alone): BackgroundTasks schedules this
    coroutine to start after the response is sent, but that guarantee only
    covers *starting* it -- Starlette's Response.__call__ does
    `await send(body); await self.background()` in the same task, on the
    single ASGI event loop thread. Confirmed live: since the real-pipeline
    branch below used to run fully synchronously right here, that
    synchronous CPU-bound call starved the event loop before it could
    finish flushing the response, so the client's socket read blocked for
    the pipeline's entire duration (>100s measured on a 150-frame clip) --
    BackgroundTasks did NOT make this non-blocking to the caller. Fixed by
    running the sync pipeline in a thread via loop.run_in_executor, so the
    event loop stays free to actually deliver the response.
    """
    loop = asyncio.get_running_loop()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            person = generate_mock_person(video_id) if USE_MOCK else None
            if person:
                await post_person(client, person)

            if USE_MOCK:
                events = generate_mock_events(video_id)
                for event in events:
                    await post_event(client, event)
                    await asyncio.sleep(0.1)  # simulate incremental production
                await post_complete(client, video_id, len(events))
            else:
                # Fix 1: off the event loop thread.
                node_payloads = await loop.run_in_executor(
                    None, _run_real_pipeline_sync, video_id, path
                )
                # Fix 2: real Node shape, not schemas.Event.
                for payload in node_payloads:
                    await post_node_event(client, payload)
                await post_complete(client, video_id, len(node_payloads))
    except Exception as e:
        # Fix 3: previously unhandled -- logged only, video stuck 'queued'
        # forever, client already had a false 200 'accepted'.
        logger.error(f"Pipeline failed for {video_id}: {e}")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await post_failed(client, video_id, str(e))
        except Exception as report_err:
            logger.error(f"Also failed to report failure for {video_id}: {report_err}")
    finally:
        # Fix 4: release the in-flight guard regardless of success/failure.
        _in_progress_video_ids.discard(video_id)


@app.post("/process")
async def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    # Fix 4: reject a duplicate /process on an already-running video_id
    # instead of letting two concurrent runs upsert onto the same
    # mlEventId documents.
    if req.video_id in _in_progress_video_ids:
        return JSONResponse(
            status_code=409,
            content={"status": "already_processing", "video_id": req.video_id},
        )
    _in_progress_video_ids.add(req.video_id)
    background_tasks.add_task(run_pipeline, req.video_id, req.path)
    return {"status": "accepted", "video_id": req.video_id}


@app.get("/health")
async def health():
    return {"status": "ok", "backend_url": BACKEND_URL, "mock_mode": USE_MOCK}


@app.get("/_debug/posted_events")
async def debug_posted_events():
    """Test-only introspection endpoint — remove before final integration."""
    return {"count": len(_posted_event_ids), "event_ids": list(_posted_event_ids)}