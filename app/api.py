"""FastAPI surface for the quality-score inference demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import model
from .engine import Engine, EngineError, MAX_PACE_MS

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_PATH = BASE_DIR / "data" / "samples.json"

app = FastAPI(
    title="产线缺陷分数推理",
    description="固定权重 NumPy 递推模型 + 单帧缺口补算",
    version="1.0.0",
)
engine = Engine()


def _load_samples() -> dict[str, Any]:
    with SAMPLES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


SAMPLES = _load_samples()


class FrameIn(BaseModel):
    camera: int = Field(..., ge=0)
    frame_no: int = Field(..., ge=1)
    features: list[float] = Field(..., min_length=model.FEATURES, max_length=model.FEATURES)


class FrameBody(BaseModel):
    frame_no: int = Field(..., ge=1)
    features: list[float] = Field(..., min_length=model.FEATURES, max_length=model.FEATURES)


class MissingBody(BaseModel):
    frame_no: int = Field(..., ge=1)


class FillBody(BaseModel):
    frame_no: int = Field(..., ge=1)
    features: list[float] = Field(..., min_length=model.FEATURES, max_length=model.FEATURES)
    pace_ms: float | None = Field(None, ge=0, le=MAX_PACE_MS)
    simulate_failure: bool = Field(
        False, description="演示用故障注入：补算第一帧后失败，用于验证原状态保留与重试"
    )


class ResetBody(BaseModel):
    confirm: Literal[True]


@app.exception_handler(EngineError)
async def engine_error_handler(_: Request, exc: EngineError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload())


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": model.info(), "cameras": [c["camera"] for c in engine.summary()["cameras"]]}


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    return model.info()


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    return SAMPLES


@app.get("/api/status")
def status() -> dict[str, Any]:
    return engine.summary()


@app.get("/api/cameras/{camera}/state")
def camera_state(camera: int) -> dict[str, Any]:
    return engine.snapshot(camera)


@app.get("/api/cameras/{camera}/scores")
def camera_scores(camera: int) -> dict[str, Any]:
    return engine.scores(camera)


@app.post("/api/cameras/{camera}/frames")
def submit_frame(camera: int, body: FrameBody) -> dict[str, Any]:
    return engine.ingest(camera, body.frame_no, body.features)


@app.post("/api/cameras/{camera}/missing")
def declare_missing(camera: int, body: MissingBody) -> dict[str, Any]:
    return engine.declare_missing(camera, body.frame_no)


@app.post("/api/cameras/{camera}/fill")
def fill_gap(camera: int, body: FillBody) -> dict[str, Any]:
    return engine.fill_gap(
        camera,
        body.frame_no,
        body.features,
        pace_ms=body.pace_ms,
        simulate_failure=body.simulate_failure,
    )


@app.post("/api/frames")
def submit_frame_flat(body: FrameIn) -> dict[str, Any]:
    """Convenience endpoint mirroring the camera/frame/feature payload."""
    return engine.ingest(body.camera, body.frame_no, body.features)


@app.post("/api/reset")
def reset(_: ResetBody) -> dict[str, Any]:
    return engine.reset()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
