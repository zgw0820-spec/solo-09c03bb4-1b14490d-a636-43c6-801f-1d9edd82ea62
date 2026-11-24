"""HTTP-level tests for the FastAPI surface."""

from __future__ import annotations

import json
import threading
import time
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


# ------------------------------------------------------------- paired views
def pair_payload(n0: int, n1: int, *, delay: float | None = None) -> dict:
    body = {"items": [
        {"camera": 0, "frame_no": n0, "features": CAM0[n0]},
        {"camera": 1, "frame_no": n1, "features": CAM1[n1]},
    ]}
    if delay is not None:
        body["compute_delay_ms"] = delay
    return body


def test_pair_endpoint_happy_path_and_history():
    r = client.post("/api/pairs", json=pair_payload(1, 1))
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "paired" and data["pair_id"].startswith("pair-")
    assert [i["camera"] for i in data["items"]] == [0, 1]
    assert all(i["estimated"] is False for i in data["items"])

    r2 = client.post("/api/pairs", json=pair_payload(2, 2))
    assert r2.status_code == 200 and r2.json()["pair_id"] != data["pair_id"]

    # The pair id is attached to the score rows of both cameras.
    for camera in (0, 1):
        snap = client.get(f"/api/cameras/{camera}/state").json()
        assert [s["pair_id"] for s in snap["scores"]] == [data["pair_id"], r2.json()["pair_id"]]

    history = client.get("/api/pairs").json()
    assert history["count"] == 2
    assert history["pairs"][0]["pair_id"] == r2.json()["pair_id"]
    assert {i["camera"] for i in history["pairs"][0]["items"]} == {0, 1}

    status = client.get("/api/status").json()
    cams = {c["camera"]: c for c in status["cameras"]}
    assert cams[0]["paired_published"] == cams[1]["paired_published"] == 2
    assert status["pairs"]["count"] == 2
    assert status["pairs"]["last"]["pair_id"] == r2.json()["pair_id"]


def test_pair_validation_errors_change_nothing():
    before = client.get("/api/status").json()
    r = client.post("/api/pairs", json={"items": [
        {"camera": 0, "frame_no": 1, "features": CAM0[1]},
        {"camera": 1, "frame_no": 1, "features": [0.1] * 7},
    ]})
    assert r.status_code == 422
    r = client.post("/api/pairs", json={"items": [
        {"camera": 0, "frame_no": 1, "features": CAM0[1]},
    ]})
    assert r.status_code == 422 and r.json()["code"] == "invalid_pair"
    r = client.post("/api/pairs", json={"items": [
        {"camera": 0, "frame_no": 1, "features": CAM0[1]},
        {"camera": 0, "frame_no": 2, "features": CAM0[2]},
    ]})
    assert r.status_code == 422 and r.json()["code"] == "invalid_pair"
    after = client.get("/api/status").json()
    assert after == before
    assert client.get("/api/pairs").json()["count"] == 0


def test_pair_stale_frame_returns_conflict_and_writes_neither_side():
    client.post("/api/cameras/0/frames", json={"frame_no": 1, "features": CAM0[1]})
    # Cam 0 next frame is 2; sending 3 skips a frame.
    r = client.post("/api/pairs", json=pair_payload(3, 1))
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "conflict" and body["camera"] == 0
    assert body["expected_frame"] == 2
    # Camera 1 got nothing from the rejected pair.
    assert client.get("/api/cameras/1/state").json()["scores"] == []


def test_pair_conflict_during_compute_returns_409_refresh_required():
    client.post("/api/pairs", json=pair_payload(1, 1))
    outcome: dict = {}

    def slow_pair() -> None:
        outcome["response"] = client.post(
            "/api/pairs", json=pair_payload(2, 2, delay=400), timeout=30
        )

    worker = threading.Thread(target=slow_pair)
    worker.start()
    time.sleep(0.1)
    single = client.post("/api/cameras/0/frames", json={"frame_no": 2, "features": CAM0[2]})
    assert single.status_code == 200  # single-camera path keeps working
    worker.join(timeout=30)
    assert not worker.is_alive()
    resp = outcome["response"]
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "pair_conflict"
    assert body["retryable"] is True and body["refresh_required"] is True
    assert body["conflicts"][0]["camera"] == 0
    # Atomicity: camera 1 stays at frame 1 and no half pair entered history.
    assert client.get("/api/cameras/1/state").json()["last_frame"] == 1
    assert client.get("/api/pairs").json()["count"] == 1

    # Refreshed next frames (A->3, B->2) succeed on retry.
    retry = client.post("/api/pairs", json=pair_payload(3, 2))
    assert retry.status_code == 200
    assert retry.json()["pair_id"].startswith("pair-")
    assert client.get("/api/pairs").json()["count"] == 2


def test_pair_with_open_gap_publishes_estimated_then_fill_keeps_pair_id():
    client.post("/api/cameras/0/frames", json={"frame_no": 1, "features": CAM0[1]})
    client.post("/api/cameras/0/missing", json={"frame_no": 2})
    client.post("/api/cameras/0/frames", json={"frame_no": 3, "features": CAM0[3]})
    client.post("/api/cameras/1/frames", json={"frame_no": 1, "features": CAM1[1]})

    pair = client.post("/api/pairs", json=pair_payload(4, 2)).json()
    a = next(i for i in pair["items"] if i["camera"] == 0)
    b = next(i for i in pair["items"] if i["camera"] == 1)
    assert a["estimated"] is True and b["estimated"] is False

    fill = client.post("/api/cameras/0/fill", json={"frame_no": 2, "features": CAM0[2]})
    assert fill.status_code == 200
    snap = client.get("/api/cameras/0/state").json()
    record = next(s for s in snap["scores"] if s["frame_no"] == 4)
    assert record["pair_id"] == pair["pair_id"]
    assert record["revised_score"] is not None
    assert record["score"] != record["revised_score"]

    # After the fill, paired frames are measured again.
    again = client.post("/api/pairs", json=pair_payload(5, 3)).json()
    a2 = next(i for i in again["items"] if i["camera"] == 0)
    assert a2["estimated"] is False


def test_reset_during_slow_pair_returns_conflict_and_clears_pairs():
    client.post("/api/pairs", json=pair_payload(1, 1))
    outcome: dict = {}

    def slow_pair() -> None:
        outcome["response"] = client.post(
            "/api/pairs", json=pair_payload(2, 2, delay=400), timeout=30
        )

    worker = threading.Thread(target=slow_pair)
    worker.start()
    time.sleep(0.1)
    client.post("/api/reset", json={"confirm": True})
    worker.join(timeout=30)
    resp = outcome["response"]
    assert resp.status_code == 409
    assert resp.json()["code"] == "pair_conflict"
    assert resp.json()["refresh_required"] is True
    assert client.get("/api/pairs").json()["count"] == 0
    # Stream versions are non-zero after reset, so stale tokens never match.
    assert all(c["version"] >= 2 for c in client.get("/api/status").json()["cameras"])
