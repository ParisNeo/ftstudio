import asyncio
import base64
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel

from . import config, db, doctor, gguf_catalog, known_datasets, library, models, presets
from .runner import Runner
from .workers import KINDS


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    db.recover()
    runner = Runner()
    runner.start()
    yield
    runner.stop_event.set()
    runner.join(timeout=20)


app = FastAPI(title="ftstudio", lifespan=lifespan)


class JobCreate(BaseModel):
    kind: str
    params: dict = {}


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    static_dir = Path(__file__).parent / "static"
    for filename in ("favicon.ico", "favicon.png"):
        file_path = static_dir / filename
        if file_path.is_file():
            return FileResponse(str(file_path))
    # Fallback: 1x1 completely transparent PNG
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwAEhgGA" 
        "WjMl9wAAAABJRU5ErkJggg=="
    )
    return Response(
        content=tiny_png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/gguf")
def get_gguf():
    return {"local": gguf_catalog.local_ggufs(), "examples": gguf_catalog.examples(), "models_dir": str(gguf_catalog.models_dir())}


_doc = {"t": 0.0, "v": None}


@app.get("/doctor")
def get_doctor(refresh: bool = False):
    if refresh or _doc["v"] is None or time.time() - _doc["t"] > 60:
        _doc.update(t=time.time(), v=doctor.report())
    return _doc["v"]


@app.get("/library")
def get_libraries():
    return library.list_libraries()


@app.get("/library/{name}")
def get_library(name: str, offset: int = 0, limit: int = 50):
    try:
        rows = library.read_all(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": name, "total": len(rows), "entries": rows[offset:offset + limit]}


@app.delete("/library/{name}/{idx}")
def delete_entry(name: str, idx: int):
    try:
        library.delete(name, idx)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except IndexError:
        raise HTTPException(404, "no such entry")
    return {"deleted": idx}


@app.get("/datasets")
def get_datasets():
    return known_datasets.all_datasets()


@app.get("/models")
def get_models():
    return models.all_models()


@app.get("/presets")
def get_presets():
    return presets.all_presets()


@app.post("/jobs")
def create_job(body: JobCreate):
    if body.kind not in KINDS:
        raise HTTPException(400, f"unknown kind {body.kind!r}; known: {sorted(KINDS)}")
    return {"id": db.create_job(body.kind, body.params)}


@app.get("/jobs")
def list_jobs(limit: int = 50):
    return db.list_jobs(limit)


@app.get("/jobs/{jid}")
def get_job(jid: str):
    job = db.get_job(jid)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.post("/jobs/{jid}/cancel")
def cancel(jid: str):
    status = db.request_cancel(jid)
    if status is None:
        raise HTTPException(404, "no such job")
    return {"status": status}


@app.post("/jobs/{jid}/resume")
def resume(jid: str):
    job = db.get_job(jid)
    if job is None:
        raise HTTPException(404, "no such job")
    if not job.get("resumable") or not job.get("checkpoint"):
        raise HTTPException(409, "job is not resumable")
    updated = db.requeue(jid)
    if updated is None:
        raise HTTPException(409, "job is not resumable")
    return updated


@app.get("/jobs/{jid}/log")
def get_log(jid: str, offset: int = 0):
    path = config.runs_dir() / jid / "log.txt"
    if not path.exists():
        return {"offset": 0, "text": ""}
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    return {"offset": offset + len(data), "text": data.decode(errors="replace")}


@app.get("/jobs/{jid}/artifacts/{name}")
def artifact(jid: str, name: str):
    if db.get_job(jid) is None:
        raise HTTPException(404, "no such job")
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "bad artifact name")
    path = config.runs_dir() / jid / name
    if not path.is_file():
        raise HTTPException(404, "no such artifact")
    return FileResponse(path)


@app.websocket("/jobs/{jid}/stream")
async def stream(ws: WebSocket, jid: str):
    await ws.accept()
    path = config.runs_dir() / jid / "log.txt"
    pos = 0
    try:
        while True:
            job = db.get_job(jid)
            done = job is None or job["status"] in db.TERMINAL  # check BEFORE reading: no lost tail
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(pos)
                    data = f.read()
                pos += len(data)
                if data:
                    await ws.send_text(data.decode(errors="replace"))
            if done:
                break
            await asyncio.sleep(0.5)
        await ws.close()
    except WebSocketDisconnect:
        pass
