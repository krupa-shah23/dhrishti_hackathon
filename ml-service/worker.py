"""
DRISHTI ML Microservice — Redis Job Worker

Listens on the BullMQ 'ml-processing-queue' for new video processing jobs.
When a job arrives, runs the ML pipeline on the video.

Uses a simplified Redis list-based approach compatible with BullMQ's
internal queue structure.
"""
import json
import time
import threading
import redis
from config import REDIS_HOST, REDIS_PORT
from pipeline_runner import run_pipeline


class MLWorker:
    """
    Worker that polls the BullMQ processing queue and runs the ML pipeline.
    """

    def __init__(self):
        # Same fix as status_updater.py's client: bounded connect/socket
        # timeouts. BRPOP's own timeout=5 (in _poll_loop) only bounds the
        # blocking read once connected -- the initial _connect() had no
        # timeout of its own before this.
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=6,  # > BRPOP's timeout=5 so it doesn't cut the blocking read short
            retry_on_timeout=False,
        )
        self.running = False
        self._thread = None
        self.queue_name = "ml-processing-queue"

    def start(self):
        """Start the worker in a background thread."""
        self.running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"🔄  ML Worker started — listening on queue: {self.queue_name}")

    def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("🛑  ML Worker stopped")

    def _poll_loop(self):
        """
        Continuously poll the BullMQ queue for new jobs.
        Uses BRPOP for efficient blocking reads.
        """
        wait_key = f"bull:{self.queue_name}:wait"

        while self.running:
            try:
                # BRPOP blocks for up to 5 seconds waiting for a job
                result = self.redis_client.brpop(wait_key, timeout=5)
                if result is None:
                    continue  # Timeout, loop again

                _, raw_data = result

                # BullMQ stores jobs as IDs — we need to fetch the job data
                # For our simplified integration, we also check if raw_data is
                # directly a JSON job payload
                job_data = self._parse_job(raw_data)
                if job_data is None:
                    continue

                video_id = job_data.get("videoId")
                filepath = job_data.get("filepath")
                filename = job_data.get("filename", "unknown")

                if not video_id or not filepath:
                    print(f"⚠️  Invalid job data: {job_data}")
                    continue

                print(f"\n🎬  Processing job: {filename} (ID: {video_id})")
                result = run_pipeline(video_id, filepath, filename)

                if result.get("success"):
                    print(f"✅  Job completed: {filename} — {result.get('events_count', 0)} events")
                else:
                    print(f"❌  Job failed: {filename} — {result.get('error')}")

            except redis.ConnectionError as e:
                print(f"⚠️  Redis connection lost: {e}. Retrying in 5s...")
                time.sleep(5)
            except Exception as e:
                print(f"❌  Worker error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def _parse_job(self, raw_data: str) -> dict:
        """
        Parse a job from the Redis queue.
        BullMQ stores job IDs in the wait list and job data in separate hash keys.
        We handle both formats.
        """
        try:
            # Try parsing as direct JSON payload first
            data = json.loads(raw_data)
            if isinstance(data, dict) and "videoId" in data:
                return data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try fetching as a BullMQ job ID
        try:
            job_id = str(raw_data)
            job_key = f"bull:{self.queue_name}:{job_id}"
            job_hash = self.redis_client.hgetall(job_key)
            if job_hash and "data" in job_hash:
                return json.loads(job_hash["data"])
        except Exception:
            pass

        print(f"⚠️  Could not parse job data: {raw_data[:200]}")
        return None


# Singleton worker instance
_worker = None


def get_worker() -> MLWorker:
    global _worker
    if _worker is None:
        _worker = MLWorker()
    return _worker
