"""In-memory per-camera ingestion engine with one-frame gap recovery.

Rules implemented here
----------------------
* Every accepted frame publishes a score immediately.  Published scores are
  immutable: a later replay never edits them, it only attaches a revised value
  next to the original.
* Each camera may have at most one outstanding missing frame ("gap").  While it
  is missing, the next frame is advanced with a zero-vector substitution and its
  score is flagged ``estimated``; following frames continue from that
  provisional state and are flagged estimated as well.
* Both cameras keep ingesting while one camera is replaying.  Frames that
  arrive on the replaying camera during replay are also scored from the
  provisional state and flagged estimated.
* When the original frame is supplied, replay starts from the state *before*
  the gap and walks the stored frames in frame order.  The corrected state is
  only committed once replay has caught up with the newest frame that already
  has a published score (a target that may grow while replay runs).  Only then
  does subsequent inference continue from the corrected state.
* If replay fails, nothing published is mutated: the provisional state, the
  open gap and every published score survive, and the fill can be retried.
* One lock per camera serialises ingest, missing-frame declaration and the
  short bookkeeping sections of replay.  Model arithmetic during replay runs
  outside the lock so the camera can keep accepting frames.

Paired (two-view) submission
----------------------------
A workpiece is scored from both cameras in one transaction
(``Engine.submit_pair``).  Each camera still recurses from its own state, but
the two frames/scores/states become visible together:

* prepare phase: both camera locks are taken in camera-id order, the payload is
  validated and each camera's state plus its monotonic ``version`` token are
  snapshotted; no mutation happens here;
* compute phase: model arithmetic runs with **no** locks held, so ordinary
  single-camera ingest and gap fills keep progressing;
* commit phase: both locks are re-taken in the same fixed order and every
  version token is re-checked.  If either camera published a new score,
  committed a replay or was reset meanwhile, the whole pair is rejected with a
  ``pair_conflict`` (HTTP 409, ``refresh_required=true``) and nothing is
  written on either side.  Otherwise both scores are emitted atomically and
  share one ``pair_id``.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np

from . import model

MAX_FRAMES_PER_CAMERA = 10_000
DEFAULT_CAMERAS = (0, 1)
SOURCE_SUBMITTED = "submitted"
SOURCE_ZERO = "zero_substitution"
REVISION_EPS = 1e-12
MAX_PACE_MS = 1000.0
# Demo-only artificial delay between the two model steps of a paired
# submission, so a concurrent single-camera change can reliably create an
# optimistic-locking conflict in the browser demo.
MAX_PAIR_DELAY_MS = 1000.0
MAX_PAIR_HISTORY = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EngineError(Exception):
    """Base error carrying an HTTP status and a machine-readable payload."""

    status_code = 400
    code = "engine_error"

    def __init__(self, message: str, *, retryable: bool = False, **extra: Any) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        body.update(self.extra)
        return body


class ValidationError(EngineError):
    status_code = 422
    code = "invalid_frame"


class ConflictError(EngineError):
    status_code = 409
    code = "conflict"


class PairConflictError(EngineError):
    """A paired submission lost its optimistic-locking race.

    At least one of the two cameras changed (new score, committed replay or
    reset) between the pair's prepare snapshot and its commit.  Neither side
    was mutated; the client must refresh and retry.
    """

    status_code = 409
    code = "pair_conflict"


class PairValidationError(EngineError):
    """The paired payload itself is malformed (not a state race)."""

    status_code = 422
    code = "invalid_pair"


class ReplayError(EngineError):
    status_code = 422
    code = "replay_failed"


@dataclass
class ScoreRecord:
    frame_no: int
    score: float
    estimated: bool
    feature_source: str
    published_at: str
    state_after: list[float]
    revision: int = 0
    revised_score: float | None = None
    revised_estimated: bool | None = None
    revised_at: str | None = None
    delta: float | None = None
    applied_to_stream: bool = False
    # Set when the frame was published as one side of a paired two-view
    # submission; None for ordinary single-camera ingest.
    pair_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_no": self.frame_no,
            "score": self.score,
            "estimated": self.estimated,
            "feature_source": self.feature_source,
            "published_at": self.published_at,
            "state_after": self.state_after,
            "revision": self.revision,
            "revised_score": self.revised_score,
            "revised_estimated": self.revised_estimated,
            "revised_at": self.revised_at,
            "delta": self.delta,
            "applied_to_stream": self.applied_to_stream,
            "pair_id": self.pair_id,
            "final_score": self.revised_score if self.revised_score is not None else self.score,
            "final_estimated": (
                self.revised_estimated if self.revised_estimated is not None else self.estimated
            ),
        }


@dataclass
class FrameRecord:
    frame_no: int
    features: list[float]
    feature_source: str


@dataclass
class Gap:
    frame_no: int
    state_before: list[float]
    opened_at: str


@dataclass
class ReplayJob:
    frame_no: int
    features: list[float]
    state_before: np.ndarray
    state: np.ndarray
    cursor: int
    scores: dict[int, float] = field(default_factory=dict)
    states: dict[int, np.ndarray] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    target_at_start: int = 0
    simulate_failure: bool = False
    cancelled: bool = False


@dataclass
class PairRecord:
    """One accepted two-view submission, one row per camera."""

    pair_id: str
    submitted_at: str
    items: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "submitted_at": self.submitted_at,
            "items": self.items,
        }


@dataclass
class CameraState:
    camera: int
    state: np.ndarray = field(default_factory=model.initial_state)
    last_frame: int = 0
    last_output_frame: int = 0
    gap: Gap | None = None
    replay: ReplayJob | None = None
    scores: dict[int, ScoreRecord] = field(default_factory=dict)
    frames: dict[int, FrameRecord] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Monotonic token bumped on every committed mutation (ingest, gap open,
    # replay commit, reset).  Paired submissions snapshot it in the prepare
    # phase and re-check it in the commit phase to detect concurrent changes.
    version: int = 0


class Engine:
    def __init__(self, cameras: tuple[int, ...] = DEFAULT_CAMERAS) -> None:
        self._states = {c: CameraState(camera=c) for c in cameras}
        self._registry_lock = threading.Lock()
        # Observability/test hook, called with (camera, frame_no) before every
        # replayed frame.  Raising from it simulates a replay failure.
        self.replay_step_hook: Callable[[int, int], None] | None = None
        # History of accepted paired submissions (newest last).  Cleared by
        # reset; the monotonic sequence counter survives resets on purpose so
        # pair ids are never reused within a process.
        self._pairs: list[PairRecord] = []
        self._pair_seq = 0

    @contextlib.contextmanager
    def _camera_locks(self, cams: Iterable[CameraState]):
        """Acquire per-camera locks in a fixed camera-id order.

        Every code path that holds more than one camera lock goes through
        here, so paired submission and cross-camera snapshots can never
        deadlock against each other.
        """
        ordered = sorted(cams, key=lambda c: c.camera)
        acquired: list[CameraState] = []
        try:
            for cam in ordered:
                cam.lock.acquire()
                acquired.append(cam)
            yield
        finally:
            for cam in reversed(acquired):
                cam.lock.release()

    # ------------------------------------------------------------------ helpers
    def _camera(self, camera: int) -> CameraState:
        with self._registry_lock:
            cam = self._states.get(camera)
        if cam is None:
            raise ValidationError(
                f"unknown camera {camera}; configured cameras: {sorted(self._states)}"
            )
        return cam

    @staticmethod
    def _validate_features(features: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if arr.shape != (model.FEATURES,):
            raise ValidationError(
                f"feature vector must contain exactly {model.FEATURES} numbers, got {arr.size}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValidationError("feature vector contains NaN or infinite values")
        if np.max(np.abs(arr)) > 1e6:
            raise ValidationError("feature values must be within +/-1e6")
        return arr

    @staticmethod
    def _check_frame_no(frame_no: int) -> None:
        if frame_no < 1:
            raise ValidationError("frame_no must be >= 1")

    @staticmethod
    def _check_pace(pace_ms: float | None) -> float:
        if pace_ms is None:
            return max(0.0, float(os.environ.get("QUALITY_REPLAY_PACE_MS", "0") or 0))
        if not (0.0 <= pace_ms <= MAX_PACE_MS):
            raise ValidationError(f"pace_ms must be between 0 and {MAX_PACE_MS:g}")
        return float(pace_ms)

    def _log(self, cam: CameraState, event: str, **payload: Any) -> None:
        entry = {"at": _now(), "camera": cam.camera, "event": event, **payload}
        cam.events.append(entry)
        if len(cam.events) > 500:
            del cam.events[: len(cam.events) - 500]

    def _emit(self, cam: CameraState, frame_no: int, features: list[float], source: str,
              state_after: np.ndarray, pair_id: str | None = None) -> ScoreRecord:
        record = ScoreRecord(
            frame_no=frame_no,
            score=model.score(state_after),
            estimated=(source == SOURCE_ZERO) or cam.gap is not None,
            feature_source=source,
            published_at=_now(),
            state_after=[round(float(v), 9) for v in state_after],
            pair_id=pair_id,
        )
        cam.scores[frame_no] = record
        cam.frames[frame_no] = FrameRecord(
            frame_no=frame_no, features=list(features), feature_source=source
        )
        cam.state = state_after
        cam.last_frame = frame_no
        cam.last_output_frame = frame_no
        # Every published frame changes the stream, so the optimistic-locking
        # token must move.  ``_open_gap`` emits through this same method.
        cam.version += 1
        return record

    # ------------------------------------------------------------------ ingest
    def ingest(self, camera: int, frame_no: int, features: list[float]) -> dict[str, Any]:
        cam = self._camera(camera)
        self._check_frame_no(frame_no)
        arr = self._validate_features(features)
        with cam.lock:
            if frame_no <= cam.last_frame:
                existing = cam.scores.get(frame_no)
                raise ConflictError(
                    f"frame {frame_no} was already scored; published scores are immutable",
                    frame_no=frame_no,
                    published_score=None if existing is None else existing.score,
                )
            if len(cam.frames) >= MAX_FRAMES_PER_CAMERA:
                raise ConflictError("camera frame buffer is full")
            skipped = frame_no - cam.last_frame - 1
            if skipped > 1:
                raise ConflictError(
                    f"jumped from frame {cam.last_frame} to {frame_no}; at most one frame "
                    "may be missing per camera at a time",
                    last_frame=cam.last_frame,
                )
            if skipped == 1:
                if cam.gap is not None:
                    raise ConflictError(
                        f"camera already has an open gap at frame {cam.gap.frame_no}; "
                        "fill it before another frame can be missing",
                        gap_frame=cam.gap.frame_no,
                    )
                self._open_gap(cam, cam.last_frame + 1)

            after = model.step(cam.state, arr)
            record = self._emit(cam, frame_no, arr.tolist(), SOURCE_SUBMITTED, after)
            gap_payload = self._gap_payload(cam)
            self._log(
                cam,
                "ingest",
                frame_no=frame_no,
                score=record.score,
                estimated=record.estimated,
                gap=gap_payload,
                state_digest=model.state_digest(after),
                replay_in_progress=cam.replay is not None,
            )
            return {
                "camera": camera,
                "frame_no": frame_no,
                "score": record.score,
                "estimated": record.estimated,
                "feature_source": record.feature_source,
                "state_digest": model.state_digest(after),
                "gap": gap_payload,
                "replay_in_progress": cam.replay is not None,
                "last_output_frame": cam.last_output_frame,
                "published_scores_unchanged": True,
                "version": cam.version,
            }

    def _open_gap(self, cam: CameraState, frame_no: int) -> ScoreRecord:
        before = cam.state
        cam.gap = Gap(frame_no=frame_no, state_before=[float(v) for v in before], opened_at=_now())
        zero = [0.0] * model.FEATURES
        after = model.step(before, np.asarray(zero))
        record = self._emit(cam, frame_no, zero, SOURCE_ZERO, after)
        self._log(
            cam,
            "gap_opened",
            frame_no=frame_no,
            score=record.score,
            state_digest=model.state_digest(after),
        )
        return record

    def declare_missing(self, camera: int, frame_no: int) -> dict[str, Any]:
        cam = self._camera(camera)
        self._check_frame_no(frame_no)
        with cam.lock:
            expected = cam.last_frame + 1
            if frame_no != expected:
                raise ConflictError(
                    f"missing frame must be the next frame ({expected}), got {frame_no}",
                    expected_frame=expected,
                )
            if cam.gap is not None:
                raise ConflictError(
                    f"camera already has an open gap at frame {cam.gap.frame_no}",
                    gap_frame=cam.gap.frame_no,
                )
            record = self._open_gap(cam, frame_no)
            return {
                "camera": camera,
                "frame_no": frame_no,
                "score": record.score,
                "estimated": True,
                "feature_source": record.feature_source,
                "state_digest": model.state_digest(cam.state),
                "gap": self._gap_payload(cam),
                "last_output_frame": cam.last_output_frame,
                "published_scores_unchanged": True,
                "version": cam.version,
            }

    # ------------------------------------------------------------- paired
    def submit_pair(
        self,
        items: list[dict[str, Any]],
        compute_delay_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Score the two views of one workpiece as a single transaction.

        ``items`` is a list of ``{"camera", "frame_no", "features"}`` with one
        entry per configured camera.  Each camera recurses from its own state,
        but the two frames are validated together, computed without any locks
        held, and committed atomically after an optimistic-locking re-check.
        If either side fails validation or loses its version race, nothing is
        written to either camera.
        """
        if not (0.0 <= float(compute_delay_ms) <= MAX_PAIR_DELAY_MS):
            raise PairValidationError(
                f"compute_delay_ms must be between 0 and {MAX_PAIR_DELAY_MS:g}"
            )
        if len(items) != len(self._states):
            raise PairValidationError(
                f"a paired submission must contain exactly {len(self._states)} camera entries "
                f"(one per camera), got {len(items)}"
            )
        seen: set[int] = set()
        cams: list[CameraState] = []
        for item in items:
            camera = item.get("camera")
            if not isinstance(camera, int) or camera in seen:
                raise PairValidationError(
                    "each camera must appear exactly once in a paired submission"
                )
            seen.add(camera)
            cams.append(self._camera(camera))  # raises on unknown camera
        if seen != set(self._states):
            raise PairValidationError(
                f"paired submission must cover cameras {sorted(self._states)}, got {sorted(seen)}"
            )
        # Feature vectors and frame numbers are validated up front as well, so
        # a malformed pair aborts before any state snapshot and leaves no trace.
        parsed: dict[int, tuple[int, np.ndarray]] = {}
        for item in items:
            camera = item["camera"]
            frame_no = item.get("frame_no")
            if not isinstance(frame_no, int) or isinstance(frame_no, bool):
                raise PairValidationError("frame_no must be an integer >= 1")
            self._check_frame_no(frame_no)
            arr = self._validate_features(item.get("features"))
            parsed[camera] = (frame_no, arr)

        # ---- prepare: validate against live state under both locks (fixed order)
        plans: list[dict[str, Any]] = []
        with self._camera_locks(cams):
            for cam in sorted(cams, key=lambda c: c.camera):
                frame_no, arr = parsed[cam.camera]
                if frame_no in cam.scores:
                    existing = cam.scores[frame_no]
                    raise ConflictError(
                        f"frame {frame_no} was already scored; published scores are immutable",
                        camera=cam.camera,
                        frame_no=frame_no,
                        published_score=existing.score,
                    )
                if len(cam.frames) >= MAX_FRAMES_PER_CAMERA:
                    raise ConflictError("camera frame buffer is full", camera=cam.camera)
                skipped = frame_no - cam.last_frame - 1
                if skipped != 0:
                    # Paired submissions only ever send the next frame of each
                    # stream; gaps are still handled through the single-camera
                    # missing/fill endpoints so the two flows stay explicit.
                    raise ConflictError(
                        f"paired submission expects the next frame {cam.last_frame + 1} on "
                        f"camera {cam.camera}, got {frame_no}; declare a missing frame on that "
                        "camera first",
                        camera=cam.camera,
                        frame_no=frame_no,
                        expected_frame=cam.last_frame + 1,
                    )
                plans.append(
                    {
                        "camera": cam.camera,
                        "frame_no": frame_no,
                        "features": arr,
                        "state_before": cam.state.copy(),
                        "version_before": cam.version,
                        "last_output_frame_before": cam.last_output_frame,
                    }
                )

        # ---- compute: no locks held; single-camera ingest/fill may run here
        computed: list[dict[str, Any]] = []
        try:
            for index, plan in enumerate(plans):
                if index > 0 and compute_delay_ms > 0:
                    # Demo-only window for a concurrent request to win the race.
                    time.sleep(compute_delay_ms / 1000.0)
                after = model.step(plan["state_before"], plan["features"])
                if not np.all(np.isfinite(after)):
                    raise ReplayError(
                        f"paired computation diverged on camera {plan['camera']} "
                        f"at frame {plan['frame_no']}"
                    )
                computed.append(
                    {
                        **plan,
                        "state_after": after,
                        "score": model.score(after),
                        "state_digest": model.state_digest(after),
                    }
                )
        except EngineError:
            raise
        except Exception as exc:  # defensive: never half-apply a pair
            raise ReplayError(
                f"paired computation failed: {exc}",
                retryable=True,
                state_preserved=True,
            ) from exc

        # ---- commit: re-check every version token, then emit both atomically
        with self._registry_lock:
            with self._camera_locks(cams):
                conflicts: list[dict[str, Any]] = []
                by_camera = {p["camera"]: p for p in computed}
                for cam in sorted(cams, key=lambda c: c.camera):
                    plan = by_camera[cam.camera]
                    if cam.version != plan["version_before"]:
                        conflicts.append(
                            {
                                "camera": cam.camera,
                                "reason": (
                                    f"camera {cam.camera} changed while the pair was being "
                                    f"scored (version {plan['version_before']} -> {cam.version}); "
                                    "a new score, replay commit or reset happened"
                                ),
                                "version_before": plan["version_before"],
                                "version_now": cam.version,
                            }
                        )
                    elif cam.last_frame != plan["frame_no"] - 1:
                        conflicts.append(
                            {
                                "camera": cam.camera,
                                "reason": (
                                    f"camera {cam.camera} moved past frame {plan['frame_no'] - 1}; "
                                    f"latest frame is now {cam.last_frame}"
                                ),
                                "expected_frame": plan["frame_no"],
                                "last_frame_now": cam.last_frame,
                            }
                        )
                if conflicts:
                    self._log_pair_conflict(cams, conflicts)
                    raise PairConflictError(
                        "paired submission rejected: at least one camera changed while the pair "
                        "was being scored; refresh the two streams and retry",
                        retryable=True,
                        refresh_required=True,
                        conflicts=conflicts,
                    )

                self._pair_seq += 1
                pair_id = f"pair-{self._pair_seq:06d}"
                published_at = _now()
                items_out: list[dict[str, Any]] = []
                for cam in sorted(cams, key=lambda c: c.camera):
                    plan = by_camera[cam.camera]
                    record = self._emit(
                        cam,
                        plan["frame_no"],
                        plan["features"].tolist(),
                        SOURCE_SUBMITTED,
                        plan["state_after"],
                        pair_id=pair_id,
                    )
                    gap_payload = self._gap_payload(cam)
                    self._log(
                        cam,
                        "pair_ingest",
                        pair_id=pair_id,
                        frame_no=plan["frame_no"],
                        score=record.score,
                        estimated=record.estimated,
                        state_digest=plan["state_digest"],
                    )
                    items_out.append(
                        {
                            "camera": cam.camera,
                            "frame_no": plan["frame_no"],
                            "score": record.score,
                            "estimated": record.estimated,
                            "feature_source": record.feature_source,
                            "state_digest": plan["state_digest"],
                            "gap": gap_payload,
                            "last_output_frame": cam.last_output_frame,
                            "version": cam.version,
                        }
                    )
                record_pair = PairRecord(
                    pair_id=pair_id,
                    submitted_at=published_at,
                    items=[
                        {
                            "camera": it["camera"],
                            "frame_no": it["frame_no"],
                            "score": it["score"],
                            "estimated": it["estimated"],
                            "version": it["version"],
                        }
                        for it in items_out
                    ],
                )
                self._pairs.append(record_pair)
                if len(self._pairs) > MAX_PAIR_HISTORY:
                    del self._pairs[: len(self._pairs) - MAX_PAIR_HISTORY]

        return {
            "pair_id": pair_id,
            "status": "paired",
            "submitted_at": published_at,
            "items": items_out,
            "published_scores_unchanged": True,
            "refresh_required": False,
        }

    def _log_pair_conflict(self, cams: list[CameraState], conflicts: list[dict[str, Any]]) -> None:
        touched = {c["camera"] for c in conflicts}
        for cam in cams:
            if cam.camera in touched:
                self._log(
                    cam,
                    "pair_conflict",
                    conflicts=conflicts,
                    version=cam.version,
                )

    def list_pairs(self, limit: int | None = None) -> dict[str, Any]:
        with self._registry_lock:
            pairs = [p.to_dict() for p in self._pairs]
        if limit is not None and limit >= 0:
            pairs = pairs[-limit:]
        return {"pairs": list(reversed(pairs)), "count": len(pairs)}

    # -------------------------------------------------------------------- fill
    def fill_gap(
        self,
        camera: int,
        frame_no: int,
        features: list[float],
        pace_ms: float | None = None,
        simulate_failure: bool = False,
    ) -> dict[str, Any]:
        cam = self._camera(camera)
        self._check_frame_no(frame_no)
        arr = self._validate_features(features)
        pace = self._check_pace(pace_ms)
        with cam.lock:
            if cam.gap is None:
                raise ConflictError(f"camera {camera} has no open gap to fill", frame_no=frame_no)
            if cam.gap.frame_no != frame_no:
                raise ConflictError(
                    f"open gap is at frame {cam.gap.frame_no}, not {frame_no}",
                    gap_frame=cam.gap.frame_no,
                )
            if cam.replay is not None:
                raise ConflictError(
                    f"a replay for frame {cam.replay.frame_no} is already running",
                    retryable=True,
                    gap_frame=cam.replay.frame_no,
                )
            job = ReplayJob(
                frame_no=frame_no,
                features=arr.tolist(),
                state_before=np.asarray(cam.gap.state_before, dtype=np.float64),
                state=np.asarray(cam.gap.state_before, dtype=np.float64).copy(),
                cursor=frame_no,
                target_at_start=cam.last_output_frame,
                simulate_failure=simulate_failure,
            )
            cam.replay = job
            gap_payload = self._gap_payload(cam)
            self._log(
                cam,
                "replay_started",
                frame_no=frame_no,
                target_at_start=job.target_at_start,
                pace_ms=pace,
            )
        return self._run_replay(cam, job, pace, gap_payload)

    def _run_replay(
        self, cam: CameraState, job: ReplayJob, pace: float, gap_payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        try:
            while True:
                with cam.lock:
                    if cam.replay is not job:
                        raise ReplayError("replay was cancelled")
                    if job.cursor > cam.last_output_frame:
                        return self._commit(cam, job)
                    frame_no = job.cursor
                    if frame_no == job.frame_no:
                        x = np.asarray(job.features, dtype=np.float64)
                    else:
                        frame = cam.frames.get(frame_no)
                        if frame is None:
                            raise ReplayError(f"frame {frame_no} is missing from the replay window")
                        if frame.feature_source != SOURCE_SUBMITTED:
                            raise ReplayError(
                                f"frame {frame_no} has no real features; cannot replay across "
                                "a second gap"
                            )
                        x = np.asarray(frame.features, dtype=np.float64)

                # Demo-only fault injection: fail once at least one frame has
                # been recomputed (or immediately if the window is just the gap
                # frame).  Nothing has been committed at this point, so the
                # provisional state and published scores stay untouched.
                if job.simulate_failure and (
                    job.cursor > job.frame_no or job.target_at_start == job.frame_no
                ):
                    raise ReplayError("simulated replay failure (demo fault injection)")

                if self.replay_step_hook is not None:
                    self.replay_step_hook(cam.camera, frame_no)
                if pace > 0:
                    time.sleep(pace / 1000.0)

                state = model.step(job.state, x)
                if not np.all(np.isfinite(state)):
                    raise ReplayError(f"replay diverged at frame {frame_no}")
                job.state = state
                job.states[frame_no] = state.copy()
                job.scores[frame_no] = model.score(state)
                job.cursor = frame_no + 1
        except ReplayError as exc:
            return self._fail_replay(cam, job, exc.message, gap_payload)
        except Exception as exc:  # defensive: never leave a half-applied replay
            return self._fail_replay(cam, job, f"replay failed: {exc}", gap_payload)

    def _fail_replay(
        self, cam: CameraState, job: ReplayJob, message: str, gap_payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        with cam.lock:
            if cam.replay is job:
                cam.replay = None
            if not job.cancelled:
                self._log(
                    cam,
                    "replay_failed",
                    frame_no=job.frame_no,
                    message=message,
                    cursor=job.cursor,
                )
            raise ReplayError(
                message,
                retryable=True,
                state_preserved=True,
                gap_frame=job.frame_no,
                gap=gap_payload,
                last_output_frame=cam.last_output_frame,
                replayed_before_failure=job.cursor - job.frame_no,
            )

    def _commit(self, cam: CameraState, job: ReplayJob) -> dict[str, Any]:
        """Apply a completed replay; caller holds ``cam.lock``."""
        target = job.cursor - 1
        if target < job.frame_no:
            raise ReplayError(f"replay did not reach frame {job.frame_no}")
        provisional_final = cam.state
        corrected_final = job.state
        revised_at = _now()
        revisions: list[dict[str, Any]] = []
        for f in range(job.frame_no, target + 1):
            record = cam.scores[f]
            new_score = job.scores[f]
            record.revision += 1
            record.revised_score = new_score
            record.revised_estimated = False
            record.revised_at = revised_at
            record.delta = new_score - record.score
            record.applied_to_stream = f == target
            revisions.append(
                {
                    "frame_no": f,
                    "published_score": record.score,
                    "revised_score": new_score,
                    "delta": record.delta,
                    "changed": abs(record.delta) > REVISION_EPS,
                    "estimated_at_publish": record.estimated,
                    "applied_to_stream": record.applied_to_stream,
                }
            )

        cam.frames[job.frame_no] = FrameRecord(
            frame_no=job.frame_no, features=list(job.features), feature_source=SOURCE_SUBMITTED
        )
        cam.gap = None
        cam.replay = None
        cam.state = corrected_final
        # A committed replay changes the stream state and every revised record:
        # any paired submission prepared before this point must be rejected.
        cam.version += 1
        state_shift = float(np.max(np.abs(corrected_final - provisional_final)))
        self._log(
            cam,
            "gap_filled",
            frame_no=job.frame_no,
            replay_from=job.frame_no,
            replay_to=target,
            replayed=len(revisions),
            target_at_start=job.target_at_start,
            target_at_commit=target,
            grew_during_replay=target > job.target_at_start,
            state_shift=state_shift,
            state_digest=model.state_digest(corrected_final),
        )
        return {
            "camera": cam.camera,
            "frame_no": job.frame_no,
            "status": "replayed",
            "replay": {
                "from": job.frame_no,
                "to": target,
                "frames": len(revisions),
                "target_at_start": job.target_at_start,
                "grew_during_replay": target > job.target_at_start,
                "caught_up_with_last_published_frame": True,
                "target_frame": target,
            },
            "gap_score": revisions[0],
            "revisions": revisions,
            "state_applied": True,
            "state_shift": state_shift,
            "state_digest_before": model.state_digest(provisional_final),
            "state_digest_after": model.state_digest(corrected_final),
            "published_scores_unchanged": True,
            "last_output_frame": cam.last_output_frame,
            "gap": None,
            "replay_in_progress": False,
            "version": cam.version,
        }

    # ---------------------------------------------------------------- snapshot
    @staticmethod
    def _gap_payload(cam: CameraState) -> dict[str, Any] | None:
        if cam.gap is None:
            return None
        return {"frame_no": cam.gap.frame_no, "opened_at": cam.gap.opened_at}

    def snapshot(self, camera: int) -> dict[str, Any]:
        cam = self._camera(camera)
        with cam.lock:
            replay = None
            if cam.replay is not None:
                replay = {
                    "frame_no": cam.replay.frame_no,
                    "cursor": cam.replay.cursor,
                    "started_at": cam.replay.started_at,
                    "target_at_start": cam.replay.target_at_start,
                    "last_output_frame": cam.last_output_frame,
                }
            return {
                "camera": camera,
                "last_frame": cam.last_frame,
                "last_output_frame": cam.last_output_frame,
                "state": [round(float(v), 9) for v in cam.state],
                "state_digest": model.state_digest(cam.state),
                "gap": self._gap_payload(cam),
                "replay": replay,
                "version": cam.version,
                "scores": [cam.scores[f].to_dict() for f in sorted(cam.scores)],
                "events": list(cam.events),
            }

    def scores(self, camera: int) -> dict[str, Any]:
        cam = self._camera(camera)
        with cam.lock:
            return {
                "camera": camera,
                "last_frame": cam.last_frame,
                "last_output_frame": cam.last_output_frame,
                "gap": self._gap_payload(cam),
                "replay_in_progress": cam.replay is not None,
                "version": cam.version,
                "scores": [cam.scores[f].to_dict() for f in sorted(cam.scores)],
            }

    def reset(self) -> dict[str, Any]:
        cancelled = []
        with self._registry_lock:
            states = list(self._states.values())
            # All camera locks together, in the fixed camera-id order, so a
            # paired submission can never observe a half-reset engine.
            with self._camera_locks(states):
                for cam in states:
                    if cam.replay is not None:
                        # Cancel: the replay worker notices the job is gone and
                        # aborts without committing anything.
                        cancelled.append(
                            {"camera": cam.camera, "frame_no": cam.replay.frame_no}
                        )
                        cam.replay.cancelled = True
                        cam.replay = None
                    cam.state = model.initial_state()
                    cam.last_frame = 0
                    cam.last_output_frame = 0
                    cam.gap = None
                    cam.replay = None
                    cam.scores.clear()
                    cam.frames.clear()
                    cam.events.clear()
                    # Bump rather than reset: stale pair tokens from before the
                    # reset must never match a fresh stream.
                    cam.version += 1
                self._pairs.clear()
        return {"status": "reset", "cameras": sorted(self._states), "cancelled_replays": cancelled}

    def summary(self) -> dict[str, Any]:
        with self._registry_lock:
            states = list(self._states.values())
        # Take every camera lock in camera-id order so the two-view summary is
        # one consistent snapshot: a paired commit cannot land between reading
        # the two cameras, and the summary can never show just one side of it.
        with self._camera_locks(states):
            out = []
            for cam in states:
                pair_scores = sum(1 for s in cam.scores.values() if s.pair_id is not None)
                out.append(
                    {
                        "camera": cam.camera,
                        "last_frame": cam.last_frame,
                        "last_output_frame": cam.last_output_frame,
                        "gap": self._gap_payload(cam),
                        "replay_in_progress": cam.replay is not None,
                        "published": len(cam.scores),
                        "paired_published": pair_scores,
                        "estimated_published": sum(
                            1 for s in cam.scores.values() if s.estimated
                        ),
                        "revised": sum(1 for s in cam.scores.values() if s.revision > 0),
                        "state_digest": model.state_digest(cam.state),
                        "version": cam.version,
                    }
                )
        # Pair history sits behind the registry lock only; read it separately
        # to keep one global lock order (registry lock -> camera locks) and
        # avoid inverting it against the commit path.
        with self._registry_lock:
            last_pair = self._pairs[-1].to_dict() if self._pairs else None
            pair_count = len(self._pairs)
        return {"cameras": out, "pairs": {"count": pair_count, "last": last_pair},
                "model": model.info()}
