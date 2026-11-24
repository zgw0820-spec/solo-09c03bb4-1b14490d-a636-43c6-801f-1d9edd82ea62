"""Behavioural tests for the gap-tolerant inference engine."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app import model
from app.engine import (
    ConflictError,
    Engine,
    ReplayError,
    ValidationError,
)

SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "app/data/samples.json").read_text("utf-8"))


def frames(camera: int) -> dict[int, list[float]]:
    cam = next(c for c in SAMPLES["cameras"] if c["camera"] == camera)
    return {f["frame_no"]: f["features"] for f in cam["frames"]}


def clean_stream(camera: int, limit: int | None = None) -> tuple[list[float], list[float]]:
    """Reference: every frame submitted in order, no gaps."""
    state = model.initial_state()
    states, scores = [], []
    data = frames(camera)
    for n in sorted(data):
        if limit is not None and n > limit:
            break
        state = model.step(state, np.asarray(data[n]))
        states.append(state.copy())
        scores.append(model.score(state))
    return states, scores


def feed(engine: Engine, camera: int, numbers, data=None):
    data = data or frames(camera)
    out = []
    for n in numbers:
        out.append(engine.ingest(camera, n, data[n]))
    return out


def test_zero_substitution_marks_estimated_and_keeps_streaming():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    gap = e.declare_missing(0, 4)
    assert gap["estimated"] is True
    assert gap["feature_source"] == "zero_substitution"
    following = feed(e, 0, [5, 6, 7, 8])
    assert all(r["estimated"] for r in following)
    snap = e.snapshot(0)
    assert snap["gap"]["frame_no"] == 4
    assert snap["last_output_frame"] == 8
    assert len(snap["scores"]) == 8


def test_fill_replays_to_clean_state_and_never_edits_published_scores():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7, 8])
    published_before = {s["frame_no"]: s["score"] for s in e.snapshot(0)["scores"]}
    result = e.fill_gap(0, 4, frames(0)[4])
    assert result["status"] == "replayed"
    assert result["replay"] == {
        "from": 4,
        "to": 8,
        "frames": 5,
        "target_at_start": 8,
        "grew_during_replay": False,
        "caught_up_with_last_published_frame": True,
        "target_frame": 8,
    }
    snap = e.snapshot(0)
    assert snap["gap"] is None
    for s in snap["scores"]:
        assert s["score"] == pytest.approx(published_before[s["frame_no"]], abs=0)
        if s["frame_no"] >= 4:
            assert s["revised_score"] is not None
            assert s["final_estimated"] is False
            assert s["feature_source"] in ("submitted", "zero_substitution")
    assert "zero_substitution" in {
        s["feature_source"] for s in snap["scores"] if s["frame_no"] == 4
    } or snap["scores"][3]["revised_score"] is not None

    # Continued inference after the fill must match a gap-free run exactly.
    states, scores = clean_stream(0, limit=10)
    after = e.ingest(0, 9, frames(0)[9])
    assert after["estimated"] is False
    assert after["score"] == pytest.approx(scores[8], abs=1e-12)
    assert np.allclose(np.asarray(e.snapshot(0)["state"]), states[8], atol=1e-12)
    after10 = e.ingest(0, 10, frames(0)[10])
    assert after10["score"] == pytest.approx(scores[9], abs=1e-12)


def test_revised_values_match_gap_free_reference():
    e = Engine()
    feed(e, 1, [1, 2, 3, 4, 5, 6, 7])
    e.declare_missing(1, 8)
    feed(e, 1, [9, 10, 11, 12])
    e.fill_gap(1, 8, frames(1)[8])
    _, ref_scores = clean_stream(1, limit=12)
    snap = e.snapshot(1)
    for s in snap["scores"]:
        if s["frame_no"] >= 8:
            assert s["final_score"] == pytest.approx(ref_scores[s["frame_no"] - 1], abs=1e-12)
    # Gap score must move from the zero-substituted estimate to the real value.
    gap = snap["scores"][7]
    prev_state = np.asarray(snap["scores"][6]["state_after"])
    assert gap["score"] == pytest.approx(
        model.score(model.step(prev_state, np.zeros(model.FEATURES))), abs=1e-8
    )
    assert abs(gap["delta"]) > 0.1


def test_cameras_are_independent():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    feed(e, 1, [1, 2, 3, 4])
    e.declare_missing(0, 4)
    r = e.ingest(1, 5, frames(1)[5])
    assert r["estimated"] is False
    assert e.snapshot(1)["gap"] is None
    assert e.snapshot(0)["gap"]["frame_no"] == 4


def test_failed_replay_preserves_state_and_can_retry():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7])
    before = e.snapshot(0)
    with pytest.raises(ReplayError) as info:
        e.fill_gap(0, 4, frames(0)[4], simulate_failure=True)
    assert info.value.retryable is True
    assert info.value.payload()["state_preserved"] is True
    after = e.snapshot(0)
    assert after["state"] == before["state"]
    assert after["state_digest"] == before["state_digest"]
    assert after["gap"] == before["gap"]
    assert [s["score"] for s in after["scores"]] == [s["score"] for s in before["scores"]]
    assert all(s["revision"] == 0 for s in after["scores"])
    # Stream keeps working while the gap stays open.
    r = e.ingest(0, 8, frames(0)[8])
    assert r["estimated"] is True
    # Retry succeeds and catches up with frame 8 too.
    result = e.fill_gap(0, 4, frames(0)[4])
    assert result["replay"]["to"] == 8
    assert e.snapshot(0)["gap"] is None
    _, ref = clean_stream(0, limit=8)
    assert e.snapshot(0)["scores"][7]["final_score"] == pytest.approx(ref[7], abs=1e-12)


def test_replay_target_grows_with_frames_arriving_during_fill():
    """边补帧边进新帧：replay must chase frames published while it runs."""
    e = Engine()
    data = frames(0)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7, 8])

    seen = []
    e.replay_step_hook = lambda cam, fr: seen.append(fr)

    result: dict = {}

    def run_fill():
        result.update(e.fill_gap(0, 4, data[4], pace_ms=40))

    worker = threading.Thread(target=run_fill)
    worker.start()
    # Let replay get past the first frames, then keep publishing.
    deadline = time.time() + 5
    while len(seen) < 2 and time.time() < deadline:
        time.sleep(0.005)
    extra = feed(e, 0, [9, 10, 11])
    worker.join(timeout=10)
    assert not worker.is_alive()

    assert all(r["estimated"] for r in extra)  # scored from provisional state
    assert result["replay"]["target_at_start"] == 8
    assert result["replay"]["to"] >= 9
    assert result["replay"]["grew_during_replay"] is True
    assert result["replay"]["caught_up_with_last_published_frame"] is True
    snap = e.snapshot(0)
    assert snap["gap"] is None
    assert snap["scores"][8]["revised_score"] is not None
    # Every frame published before the commit is now corrected against reality.
    _, ref = clean_stream(0, limit=result["replay"]["to"])
    for s in snap["scores"]:
        if s["frame_no"] >= 4:
            assert s["final_estimated"] is False
            assert s["final_score"] == pytest.approx(ref[s["frame_no"] - 1], abs=1e-12)
    # New frames after commit continue from the corrected state.
    nxt = result["replay"]["to"] + 1
    after = e.ingest(0, nxt, data[nxt])
    assert after["estimated"] is False
    assert after["score"] == pytest.approx(clean_stream(0, limit=nxt)[1][-1], abs=1e-12)


def test_second_gap_inside_replay_window_fails_without_mutation():
    """A second zero-substituted frame in the window makes replay ambiguous."""
    e = Engine()
    # Craft a state where frame 5's stored features are synthetic.
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5])
    # Manually open a second gap in the window to exercise the guard.
    cam = e._camera(0)
    with cam.lock:
        cam.frames[5].feature_source = "zero_substitution"
    before = e.snapshot(0)
    with pytest.raises(ReplayError) as info:
        e.fill_gap(0, 4, frames(0)[4])
    assert info.value.retryable is True
    after = e.snapshot(0)
    assert after["state"] == before["state"]
    assert after["gap"] == before["gap"]
    assert e._camera(0).replay is None


def test_validation_and_conflicts():
    e = Engine()
    with pytest.raises(ValidationError):
        e.ingest(9, 1, [0.0] * model.FEATURES)
    with pytest.raises(ValidationError):
        e.ingest(0, 1, [0.0] * (model.FEATURES - 1))
    with pytest.raises(ValidationError):
        e.ingest(0, 1, [float("nan")] * model.FEATURES)
    with pytest.raises(ConflictError):
        e.ingest(0, 3, frames(0)[3])  # starts at frame 3 -> 2 missing
    e.ingest(0, 1, frames(0)[1])
    with pytest.raises(ConflictError):
        e.ingest(0, 1, frames(0)[1])
    with pytest.raises(ConflictError):
        e.ingest(0, 4, frames(0)[4])  # 2 missing (frames 2 and 3)
    e.declare_missing(0, 2)
    # The next frame may still arrive while the gap is open (estimated).
    assert e.ingest(0, 3, frames(0)[3])["estimated"] is True
    with pytest.raises(ConflictError):
        e.declare_missing(0, 4)  # a second missing frame is not allowed
    with pytest.raises(ConflictError):
        e.declare_missing(0, 3)
    with pytest.raises(ConflictError):
        e.fill_gap(0, 5, frames(0)[5])
    with pytest.raises(ConflictError):
        e.fill_gap(1, 2, frames(1)[2])  # camera 1 has no gap


def test_fill_on_camera_does_not_block_other_camera():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7])
    feed(e, 1, [1, 2])
    result: dict = {}

    def run_fill():
        result.update(e.fill_gap(0, 4, frames(0)[4], pace_ms=60))

    worker = threading.Thread(target=run_fill)
    worker.start()
    time.sleep(0.03)
    t0 = time.perf_counter()
    r = feed(e, 1, [3, 4, 5])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5
    assert all(x["estimated"] is False for x in r)
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert result["status"] == "replayed"


def test_reset():
    e = Engine()
    feed(e, 0, [1, 2])
    e.reset()
    snap = e.snapshot(0)
    assert snap["scores"] == [] and snap["gap"] is None and snap["last_frame"] == 0
    assert e.snapshot(0)["state_digest"] == e.snapshot(1)["state_digest"]


def test_gap_at_latest_frame_can_be_filled_immediately():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    result = e.fill_gap(0, 4, frames(0)[4])
    assert result["replay"]["from"] == 4 and result["replay"]["to"] == 4
    assert result["replay"]["frames"] == 1
    _, ref = clean_stream(0, limit=4)
    assert e.snapshot(0)["scores"][3]["final_score"] == pytest.approx(ref[3], abs=1e-12)
    assert e.ingest(0, 5, frames(0)[5])["score"] == pytest.approx(clean_stream(0, limit=5)[1][4], abs=1e-12)


def test_concurrent_ingest_during_replay_never_breaks_ordering():
    """Hammer one camera with frames while a paced replay runs."""
    e = Engine()
    data = frames(0)
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7, 8])
    result: dict = {}
    errors: list[BaseException] = []

    def run_fill():
        try:
            result.update(e.fill_gap(0, 4, data[4], pace_ms=5))
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    worker = threading.Thread(target=run_fill)
    worker.start()
    sent = []
    for n in range(9, 14):
        try:
            sent.append(e.ingest(0, n, data[n]))
        except ConflictError:
            pass
        time.sleep(0.008)
    worker.join(timeout=20)
    assert not worker.is_alive()
    assert not errors
    snap = e.snapshot(0)
    assert snap["gap"] is None
    target = result["replay"]["to"]
    _, ref = clean_stream(0, limit=target)
    frames_out = [s["frame_no"] for s in snap["scores"]]
    assert frames_out == sorted(set(frames_out))
    for s in snap["scores"]:
        if s["frame_no"] >= 4:
            assert s["final_estimated"] is False
            assert s["final_score"] == pytest.approx(ref[s["frame_no"] - 1], abs=1e-12)
    # The stream after the commit always continues from a corrected state.
    nxt = snap["last_output_frame"] + 1
    if nxt <= 14:
        r = e.ingest(0, nxt, data[nxt])
        assert r["estimated"] is False
        assert r["score"] == pytest.approx(clean_stream(0, limit=nxt)[1][-1], abs=1e-12)


def test_reset_during_replay_cancels_without_committing():
    e = Engine()
    feed(e, 0, [1, 2, 3])
    e.declare_missing(0, 4)
    feed(e, 0, [5, 6, 7])
    result: dict = {}
    outcome: list[BaseException] = []

    def run_fill():
        try:
            e.fill_gap(0, 4, frames(0)[4], pace_ms=80)
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    worker = threading.Thread(target=run_fill)
    worker.start()
    time.sleep(0.05)
    reset = e.reset()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert reset["cancelled_replays"] == [{"camera": 0, "frame_no": 4}]
    assert outcome and isinstance(outcome[0], ReplayError)
    snap = e.snapshot(0)
    assert snap["scores"] == [] and snap["gap"] is None and snap["last_frame"] == 0
