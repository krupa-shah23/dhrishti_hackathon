from fastapi import FastAPI
app = FastAPI()

@app.post("/internal/events")
async def events(payload: dict): return {"ok": True}

@app.post("/internal/persons")
async def persons(payload: dict): return {"ok": True}

@app.post("/internal/complete/{video_id}")
async def complete(video_id: str, payload: dict): return {"ok": True}