"""
DRISHTI ML Microservice — FastAPI Application

Provides:
1. Health check endpoint
2. Manual pipeline trigger endpoint (for testing)
3. Starts the BullMQ worker on application startup

Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import threading

from config import ML_SERVICE_PORT
from worker import get_worker
from pipeline_runner import run_pipeline
from db import get_db


# ─── Lifespan: start/stop the worker ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the ML worker on app startup, stop on shutdown."""
    worker = get_worker()
    try:
        worker.start()
    except Exception as e:
        print(f"⚠️  Worker failed to start: {e}")
    yield
    worker.stop()


app = FastAPI(
    title="DRISHTI ML Microservice",
    description="AI Video Analytics — ML Processing Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ───
class ProcessRequest(BaseModel):
    video_id: str
    filepath: str
    filename: Optional[str] = "unknown"


class HealthResponse(BaseModel):
    status: str
    service: str
    worker_running: bool
    db_connected: bool


# ─── Routes ───
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — verifies DB connectivity and worker status."""
    worker = get_worker()
    db_ok = False
    try:
        get_db().command("ping")
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok and worker.running else "degraded",
        service="drishti-ml-service",
        worker_running=worker.running,
        db_connected=db_ok,
    )


@app.post("/process")
async def trigger_processing(req: ProcessRequest):
    """
    Manually trigger pipeline processing for a video.
    Useful for testing without Redis/BullMQ.
    Runs in a background thread to avoid blocking the API.
    """
    def _run():
        result = run_pipeline(req.video_id, req.filepath, req.filename)
        print(f"📊  Manual trigger result: {result}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "success": True,
        "message": f"Pipeline started for video {req.video_id}",
        "note": "Processing runs in background. Check /api/videos/:id for status.",
    }


@app.get("/queue/status")
async def queue_status():
    """Check the current state of the ML processing queue."""
    worker = get_worker()
    try:
        import redis as redis_lib
        from config import REDIS_HOST, REDIS_PORT
        r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        waiting = r.llen("bull:ml-processing-queue:wait")
        return {
            "worker_running": worker.running,
            "jobs_waiting": waiting,
        }
    except Exception as e:
        return {
            "worker_running": worker.running,
            "jobs_waiting": "unknown",
            "error": str(e),
        }


# ─── Main ───
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=ML_SERVICE_PORT, reload=True)
