"""HTTP-level tests for the FastAPI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app, engine

SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "app/data/samples.json").read_text("utf-8"))
CAM0 = {f["frame_no"]: f["features"] for f in SAMPLES["cameras"][0]["frames"]}
CAM1 = {f["frame_no"]: f["features"] for f in SAMPLES["cameras"][1]["frames"]}


@pytest.fixture(autouse=True)
def _reset():
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


def test_health_model_samples():
    assert client.get("/api/health").json()["status"] == "ok"
    info = client.get("/api/model").json()
    assert info["fixed_weights"] is True and info["features"] == 8 and info["hidden"] == 6
    samples = client.get("/api/samples").json()
    assert [c["camera"] for c in samples["cameras"]] == [0, 1]
    assert len(samples["cameras"][0]["frames"]) >= 18


def test_index_and_static():
    r = client.get("/")
    assert r.status_code == 200 and "产线缺陷分数推理" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_frame_endpoints_and_immutability():
    for n in (1, 2, 3):
        r = client.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
        assert r.status_code == 200 and r.json()["estimated"] is False
    dup = client.post("/api/cameras/0/frames", json={"frame_no": 3, "features": CAM0[3]})
    assert dup.status_code == 409
    body = dup.json()
    assert body["code"] == "conflict" and body["retryable"] is False
    assert "immutable" in body["message"]


def test_gap_fill_flow_over_http():
    for n in (1, 2, 3):
        client.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
    miss = client.post("/api/cameras/0/missing", json={"frame_no": 4})
    assert miss.status_code == 200 and miss.json()["estimated"] is True
    for n in (5, 6, 7, 8):
        r = client.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
        assert r.json()["estimated"] is True
    # Camera B keeps publishing independently.
    for n in (1, 2, 3):
        assert client.post("/api/cameras/1/frames", json={"frame_no": n, "features": CAM1[n]}).json()["estimated"] is False

    fill = client.post("/api/cameras/0/fill", json={"frame_no": 4, "features": CAM0[4], "pace_ms": 0})
    assert fill.status_code == 200
    data = fill.json()
    assert data["status"] == "replayed"
    assert data["replay"]["from"] == 4 and data["replay"]["to"] == 8
    assert data["published_scores_unchanged"] is True
    assert data["revisions"][0]["published_score"] == pytest.approx(data["gap_score"]["published_score"])
    assert data["gap_score"]["delta"] > 0.1

    snap = client.get("/api/cameras/0/state").json()
    assert snap["gap"] is None
    scores = snap["scores"]
    assert [s["frame_no"] for s in scores] == list(range(1, 9))
    # Published values are intact; revised values live alongside them.
    assert scores[3]["score"] == pytest.approx(data["gap_score"]["published_score"])
    assert scores[3]["revised_score"] == pytest.approx(data["gap_score"]["revised_score"])
    nxt = client.post("/api/cameras/0/frames", json={"frame_no": 9, "features": CAM0[9]}).json()
    assert nxt["estimated"] is False


def test_failed_fill_http_contract_then_retry():
    for n in (1, 2, 3):
        client.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
    client.post("/api/cameras/0/missing", json={"frame_no": 4})
    for n in (5, 6, 7):
        client.post("/api/cameras/0/frames", json={"frame_no": n, "features": CAM0[n]})
    before = client.get("/api/cameras/0/state").json()
    bad = client.post("/api/cameras/0/fill", json={"frame_no": 4, "features": CAM0[4], "simulate_failure": True})
    assert bad.status_code == 422
    err = bad.json()
    assert err["code"] == "replay_failed" and err["retryable"] is True and err["state_preserved"] is True
    after = client.get("/api/cameras/0/state").json()
    assert after["state_digest"] == before["state_digest"]
    assert after["gap"] == before["gap"]
    assert [s["score"] for s in after["scores"]] == [s["score"] for s in before["scores"]]
    # New frame still gets a score (estimated) while gap is open.
    keep = client.post("/api/cameras/0/frames", json={"frame_no": 8, "features": CAM0[8]})
    assert keep.status_code == 200 and keep.json()["estimated"] is True
    retry = client.post("/api/cameras/0/fill", json={"frame_no": 4, "features": CAM0[4]})
    assert retry.status_code == 200 and retry.json()["replay"]["to"] == 8


def test_validation_errors_over_http():
    r = client.post("/api/cameras/0/frames", json={"frame_no": 1, "features": [0.1] * 7})
    assert r.status_code == 422
    r = client.post("/api/cameras/7/frames", json={"frame_no": 1, "features": CAM0[1]})
    assert r.status_code == 422 and r.json()["code"] == "invalid_frame"
    r = client.post("/api/cameras/0/frames", json={"frame_no": 0, "features": CAM0[1]})
    assert r.status_code == 422
    r = client.post("/api/reset", json={})
    assert r.status_code == 422  # explicit confirmation required


def test_flat_frame_endpoint():
    r = client.post("/api/frames", json={"camera": 1, "frame_no": 1, "features": CAM1[1]})
    assert r.status_code == 200 and r.json()["camera"] == 1
    assert client.get("/api/status").json()["cameras"][1]["published"] == 1
