"""Behavioural tests for paired (two-view) submission.

Covers the guarantees from the pairing rules:

* the two views of a workpiece commit together and share one ``pair_id``;
* validation or computation failure on either side leaves both untouched;
* a new score, replay commit or reset on either camera while a pair is being
  computed rejects the whole pair (optimistic locking, HTTP-level 409
  ``pair_conflict``), again with no half-written state;
* single-camera ingest, gap handling and replay keep working during pair
  computation; estimated frames flow through pairs unchanged;
* the cross-camera summary never shows just one side of a pair.
"""

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
    PairConflictError,
    PairValidationError,
    ValidationError,
)

SAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "app/data/samples.json").read_text("utf-8")
)


def frames(camera: int) -> dict[int, list[float]]:
    cam = next(c for c in SAMPLES["cameras"] if c["camera"] == camera)
    return {f["frame_no"]: f["features"] for f in cam["frames"]}


def pair_items(nos: dict[int, int]) -> list[dict]:
    return [
        {"camera": cam, "frame_no": n, "features": frames(cam)[n]}
        for cam, n in sorted(nos.items())
    ]


def clean_score(camera: int, frame_no: int) -> float:
    state = model.initial_state()
    data = frames(camera)
    score = 0.0
    for n in range(1, frame_no + 1):
        state = model.step(state, np.asarray(data[n]))
        score = model.score(state)
    return score


def test_pair_commits_both_views_together_with_shared_pair_id():
    e = Engine()
    r = e.submit_pair(pair_items({0: 1, 1: 1}))
    assert r["status"] == "paired" and r["refresh_required"] is False
    pair_id = r["pair_id"]
    assert [it["camera"] for it in r["items"]] == [0, 1]
    for it in r["items"]:
        assert it["estimated"] is False
        assert it["score"] == pytest.approx(clean_score(it["camera"], 1), abs=1e-12)
    snap0, snap1 = e.snapshot(0), e.snapshot(1)
    assert snap0["scores"][0]["pair_id"] == pair_id
    assert snap1["scores"][0]["pair_id"] == pair_id
    assert snap0["version"] == 1 and snap1["version"] == 1
    history = e.list_pairs()
    assert history["count"] == 1
    assert history["pairs"][0]["pair_id"] == pair_id
    assert {i["camera"] for i in history["pairs"][0]["items"]} == {0, 1}


def test_pair_recurses_from_each_cameras_own_state():
    e = Engine()
    # Diverge the two streams first.
    e.ingest(0, 1, frames(0)[1])
    e.ingest(1, 1, frames(1)[1])
    e.ingest(1, 2, frames(1)[2])
    r = e.submit_pair(pair_items({0: 2, 1: 3}))
    by_cam = {it["camera"]: it for it in r["items"]}
    assert by_cam[0]["frame_no"] == 2 and by_cam[1]["frame_no"] == 3
    assert by_cam[0]["score"] == pytest.approx(clean_score(0, 2), abs=1e-12)
    assert by_cam[1]["score"] == pytest.approx(clean_score(1, 3), abs=1e-12)


def test_pair_requires_exactly_one_entry_per_camera():
    e = Engine()
    for bad in (
        [],
        [{"camera": 0, "frame_no": 1, "features": frames(0)[1]}],
        pair_items({0: 1, 1: 1}) + pair_items({0: 2, 1: 2}),
        [
            {"camera": 0, "frame_no": 1, "features": frames(0)[1]},
            {"camera": 0, "frame_no": 2, "features": frames(0)[2]},
        ],
        [
            {"camera": 0, "frame_no": 1, "features": frames(0)[1]},
            {"camera": 9, "frame_no": 1, "features": frames(1)[1]},
        ],
    ):
        with pytest.raises((PairValidationError, ValidationError)):
            e.submit_pair(bad)
    assert e.list_pairs()["count"] == 0
    assert e.snapshot(0)["version"] == 0 and e.snapshot(1)["version"] == 0


def test_pair_rejects_non_next_frame_without_mutating_either_side():
    e = Engine()
    e.submit_pair(pair_items({0: 1, 1: 1}))
    before0, before1 = e.snapshot(0), e.snapshot(1)
    # Camera 0 would skip frame 2; camera 1 payload is fine on its own.
    with pytest.raises(ConflictError) as info:
        e.submit_pair(pair_items({0: 3, 1: 2}))
    assert info.value.payload()["camera"] == 0
    assert info.value.payload()["expected_frame"] == 2
    after0, after1 = e.snapshot(0), e.snapshot(1)
    assert after0["version"] == before0["version"]
    assert after1["version"] == before1["version"]
    assert len(after0["scores"]) == 1 and len(after1["scores"]) == 1
    assert e.list_pairs()["count"] == 1


def test_pair_validation_failure_is_all_or_nothing():
    e = Engine()
    before = (e.snapshot(0), e.snapshot(1))
    bad_payload = [
        {"camera": 0, "frame_no": 1, "features": [0.1] * 7},
        {"camera": 1, "frame_no": 1, "features": frames(1)[1]},
    ]
    with pytest.raises(ValidationError):
        e.submit_pair(bad_payload)
    bad_payload[0]["features"] = [float("nan")] * 8
    with pytest.raises(ValidationError):
        e.submit_pair(bad_payload)
    for (cam, snap) in zip((0, 1), before):
        now = e.snapshot(cam)
        assert now["version"] == snap["version"]
        assert now["scores"] == [] and now["last_frame"] == 0
    assert e.list_pairs()["count"] == 0


def test_concurrent_single_ingest_on_either_camera_rejects_whole_pair():
    for racer_camera in (0, 1):
        e = Engine()
        e.submit_pair(pair_items({0: 1, 1: 1}))
        v0, v1 = e.snapshot(0)["version"], e.snapshot(1)["version"]
        outcome: dict = {}

        def slow_pair() -> None:
            try:
                e.submit_pair(pair_items({0: 2, 1: 2}), compute_delay_ms=300)
            except PairConflictError as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=slow_pair)
        worker.start()
        time.sleep(0.08)
        e.ingest(racer_camera, 2, frames(racer_camera)[2])
        worker.join(timeout=10)
        assert not worker.is_alive()
        exc = outcome["error"]
        assert isinstance(exc, PairConflictError)
        body = exc.payload()
        assert body["refresh_required"] is True and body["retryable"] is True
        assert body["conflicts"][0]["camera"] == racer_camera
        # The other camera must not carry any frame from the rejected pair.
        other = 1 - racer_camera
        assert e.snapshot(other)["last_frame"] == 1
        assert e.snapshot(other)["version"] == (v1 if other == 1 else v0)
        assert e.list_pairs()["count"] == 1
        # Refreshed next frames then succeed as a new pair.
        retry_nos = {0: 3 if racer_camera == 0 else 2, 1: 3 if racer_camera == 1 else 2}
        ok = e.submit_pair(pair_items(retry_nos))
        assert ok["pair_id"].startswith("pair-")
        assert e.list_pairs()["count"] == 2


def test_replay_commit_during_pair_computation_conflicts_and_keeps_scores():
    e = Engine()
    e.ingest(0, 1, frames(0)[1])
    e.declare_missing(0, 2)
    e.ingest(0, 3, frames(0)[3])
    e.ingest(1, 1, frames(1)[1])
    outcome: dict = {}

    def slow_pair() -> None:
        try:
            e.submit_pair(pair_items({0: 4, 1: 2}), compute_delay_ms=300)
        except PairConflictError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=slow_pair)
    worker.start()
    time.sleep(0.08)
    e.fill_gap(0, 2, frames(0)[2])
    worker.join(timeout=10)
    assert isinstance(outcome.get("error"), PairConflictError)
    # Nothing from the rejected pair landed; cam 1 is still at frame 1.
    assert e.snapshot(1)["last_frame"] == 1
    assert e.snapshot(0)["gap"] is None
    # A refreshed pair now works and camera 0 is no longer estimated.
    ok = e.submit_pair(pair_items({0: 4, 1: 2}))
    a = next(i for i in ok["items"] if i["camera"] == 0)
    assert a["estimated"] is False


def test_reset_during_pair_computation_rejects_pair_and_clears_history():
    e = Engine()
    e.submit_pair(pair_items({0: 1, 1: 1}))
    outcome: dict = {}

    def slow_pair() -> None:
        try:
            e.submit_pair(pair_items({0: 2, 1: 2}), compute_delay_ms=300)
        except PairConflictError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=slow_pair)
    worker.start()
    # Wait until the pair has passed its prepare phase (frame 2 is the next
    # frame on both cameras) so the reset races the commit, not validation.
    time.sleep(0.1)
    e.reset()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), PairConflictError)
    assert outcome["error"].payload()["refresh_required"] is True
    assert e.list_pairs()["count"] == 0
    # Versions keep increasing across reset, so stale tokens never match.
    assert all(c["version"] >= 2 for c in e.summary()["cameras"])


def test_gap_camera_publishes_estimated_pair_scores_then_fill_revises_them():
    e = Engine()
    e.ingest(0, 1, frames(0)[1])
    e.declare_missing(0, 2)
    e.ingest(0, 3, frames(0)[3])
    e.ingest(1, 1, frames(1)[1])
    pair = e.submit_pair(pair_items({0: 4, 1: 2}))
    a, b = (next(i for i in pair["items"] if i["camera"] == c) for c in (0, 1))
    assert a["estimated"] is True and b["estimated"] is False
    # Published score stays; fill attaches the revision and bumps the version.
    published = e.snapshot(0)["scores"][3]["score"]
    fill = e.fill_gap(0, 2, frames(0)[2])
    assert fill["version"] == e.snapshot(0)["version"]
    record4 = e.snapshot(0)["scores"][3]
    assert record4["score"] == pytest.approx(published, abs=0)
    assert record4["revised_score"] is not None
    assert record4["pair_id"] == pair["pair_id"]
    # The pair id is retained through revision and remains queryable.
    assert e.list_pairs()["pairs"][0]["pair_id"] == pair["pair_id"]


def test_single_camera_ingest_progresses_while_pair_is_computing():
    e = Engine()
    e.submit_pair(pair_items({0: 1, 1: 1}))
    outcome: dict = {}

    def slow_pair() -> None:
        try:
            # Both frame 2 values are valid next frames at prepare time.
            e.submit_pair(pair_items({0: 2, 1: 2}), compute_delay_ms=250)
        except PairConflictError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=slow_pair)
    worker.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    r = e.ingest(0, 2, frames(0)[2])
    elapsed = time.perf_counter() - t0
    worker.join(timeout=10)
    # The single-camera path was not blocked by the lock-free pair compute.
    assert elapsed < 1.0 and r["frame_no"] == 2
    assert isinstance(outcome.get("error"), PairConflictError)


def test_summary_is_a_consistent_two_camera_snapshot():
    e = Engine()
    for n in (1, 2, 3):
        e.submit_pair(pair_items({0: n, 1: n}))
    summary = e.summary()
    cams = {c["camera"]: c for c in summary["cameras"]}
    assert cams[0]["published"] == cams[1]["published"] == 3
    assert cams[0]["paired_published"] == cams[1]["paired_published"] == 3
    assert cams[0]["version"] == cams[1]["version"] == 3
    assert summary["pairs"]["count"] == 3
    assert summary["pairs"]["last"]["pair_id"] == e.list_pairs()["pairs"][0]["pair_id"]
    # Every paired score on both cameras carries the same three pair ids.
    ids0 = {s["pair_id"] for s in e.snapshot(0)["scores"]}
    ids1 = {s["pair_id"] for s in e.snapshot(1)["scores"]}
    assert ids0 == ids1 and len(ids0) == 3


def test_pair_ids_are_monotonic_and_unique_but_cleared_from_history_on_reset():
    e = Engine()
    first = e.submit_pair(pair_items({0: 1, 1: 1}))["pair_id"]
    e.reset()
    second = e.submit_pair(pair_items({0: 1, 1: 1}))["pair_id"]
    assert second != first
    history = e.list_pairs()
    assert history["count"] == 1 and history["pairs"][0]["pair_id"] == second
    # Fresh streams start scoring from frame 1 again.
    assert e.snapshot(0)["scores"][0]["frame_no"] == 1


def test_pair_history_is_capped():
    from app.engine import MAX_PAIR_HISTORY

    e = Engine()
    # Synthetic features (samples only ship 18 frames per camera); only frame
    # numbers and vector shape matter for the history bound.
    total = MAX_PAIR_HISTORY + 4
    for n in range(1, total + 1):
        vec = [0.01 * n] * 8
        e.submit_pair(
            [
                {"camera": 0, "frame_no": n, "features": vec},
                {"camera": 1, "frame_no": n, "features": vec},
            ]
        )
    data = e.list_pairs()
    assert data["count"] == MAX_PAIR_HISTORY
    assert len(data["pairs"]) == MAX_PAIR_HISTORY
    # Newest first.
    assert data["pairs"][0]["pair_id"] == f"pair-{total:06d}"
    assert data["pairs"][-1]["pair_id"] == f"pair-{total - MAX_PAIR_HISTORY + 1:06d}"
