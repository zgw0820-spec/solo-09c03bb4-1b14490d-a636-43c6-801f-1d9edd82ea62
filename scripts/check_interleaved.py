#!/usr/bin/env python3
"""Deliverable check: fill a gap while new frames keep arriving.

Starts the app on a free local port, runs the interleaved scenario over HTTP
and asserts the guarantees that matter, printing a readable trace:

1. each camera publishes scores immediately, even with an open gap;
2. a gap frame is scored from a zero vector and flagged ``estimated``;
3. replay starts from the state before the gap and walks frames in order;
4. frames published *during* replay are chased by the replay (target grows);
5. the corrected state is committed only after catching up, then new frames
   are scored as non-estimated and match a gap-free reference;
6. already-published scores are never overwritten;
7. a failed replay leaves state, gap and published scores intact and retries.

Run with:  python scripts/check_interleaved.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import model  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)


def wait_healthy(base: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=1) as resp:
                if json.load(resp)["status"] == "ok":
                    return
        except Exception:  # noqa: BLE001 - retry until healthy
            time.sleep(0.2)
    raise RuntimeError("server did not become healthy")


def reference(camera: int, samples: dict, limit: int) -> list[float]:
    frames = next(c for c in samples["cameras"] if c["camera"] == camera)["frames"]
    state = model.initial_state()
    out = []
    for frame in frames:
        if frame["frame_no"] > limit:
            break
        state = model.step(state, np.asarray(frame["features"]))
        out.append(model.score(state))
    return out


def check(condition: bool, message: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + message)
    if not condition:
        raise SystemExit(1)


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.api:app",
            "--host", "127.0.0.1", "--port", str(port), "--no-server-header",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_healthy(base)
        c = Client(base)
        samples = c.call("GET", "/api/samples")[1]
        cam0 = {f["frame_no"]: f["features"] for f in samples["cameras"][0]["frames"]}
        cam1 = {f["frame_no"]: f["features"] for f in samples["cameras"][1]["frames"]}
        c.call("POST", "/api/reset", {"confirm": True})

        print("[1] 相机 A: 送 1–3 帧，第 4 帧缺失，继续送 5–8 帧")
        for n in (1, 2, 3):
            status, r = c.call("POST", "/api/cameras/0/frames", {"frame_no": n, "features": cam0[n]})
            check(status == 200 and r["estimated"] is False, f"A#{n} 实测出分 {r['score']:.4f}")
        _, miss = c.call("POST", "/api/cameras/0/missing", {"frame_no": 4})
        check(miss["estimated"] is True, f"A#4 零向量代入，暂估 {miss['score']:.4f}")
        for n in (5, 6, 7, 8):
            _, r = c.call("POST", "/api/cameras/0/frames", {"frame_no": n, "features": cam0[n]})
            check(r["estimated"] is True, f"A#{n} 缺口未补，仍出分 {r['score']:.4f}（暂估）")

        print("[2] 相机 B: 1–7 帧，第 8 帧缺失，继续送 9–14 帧")
        for n in range(1, 8):
            c.call("POST", "/api/cameras/1/frames", {"frame_no": n, "features": cam1[n]})
        c.call("POST", "/api/cameras/1/missing", {"frame_no": 8})
        for n in range(9, 15):
            _, r = c.call("POST", "/api/cameras/1/frames", {"frame_no": n, "features": cam1[n]})
            check(r["estimated"] is True, f"B#{n} 缺口未补，仍出分 {r['score']:.4f}（暂估）")

        published_before = {
            s["frame_no"]: s["score"] for s in c.call("GET", "/api/cameras/0/scores")[1]["scores"]
        }

        print("[3] 相机 A 开始补算第 4 帧；补算途中相机 A 继续进第 9 帧，相机 B 继续进新帧")
        result: dict = {}

        def run_fill() -> None:
            result.update(
                c.call(
                    "POST",
                    "/api/cameras/0/fill",
                    {"frame_no": 4, "features": cam0[4], "pace_ms": 200},
                )[1]
            )

        worker = threading.Thread(target=run_fill)
        worker.start()
        time.sleep(0.12)
        _, b_new = c.call("POST", "/api/cameras/1/frames", {"frame_no": 15, "features": cam1[15]})
        check(b_new["estimated"] is True, f"补算期间 B#15 照常出分 {b_new['score']:.4f}")
        status, a9 = c.call("POST", "/api/cameras/0/frames", {"frame_no": 9, "features": cam0[9]})
        check(
            status == 200 and a9["estimated"] is True and a9["replay_in_progress"] is True,
            f"补算未追平时 A#9 先按暂估状态出分 {a9['score']:.4f}",
        )
        worker.join(timeout=30)
        check(not worker.is_alive(), "补算请求已返回")
        check(result.get("status") == "replayed", f"补算结果 {result.get('replay')}")
        replay = result["replay"]
        check(replay["from"] == 4 and replay["target_at_start"] == 8, "从缺口帧起回放，起始目标 #8")
        check(
            replay["to"] >= 9 and replay["grew_during_replay"] is True,
            f"追上补算期间新出分的帧 → #{replay['to']}",
        )
        check(replay["caught_up_with_last_published_frame"] is True, "追上本路最新已出分帧后才提交")

        print("[4] 校验：已发分数未改、修正值对齐无缺口参照、后续推理使用修正状态")
        snap = c.call("GET", "/api/cameras/0/state")[1]
        check(snap["gap"] is None, "缺口已清空")
        check(
            all(
                published_before[s["frame_no"]] == s["score"]
                for s in snap["scores"]
                if s["frame_no"] in published_before
            ),
            "已发分数逐帧保持不变（修正值并列，不覆盖）",
        )
        ref = reference(0, samples, replay["to"])
        revised_ok = all(
            abs(s["final_score"] - ref[s["frame_no"] - 1]) < 1e-12
            for s in snap["scores"]
            if s["frame_no"] >= 4
        )
        check(revised_ok, f"#4–#{replay['to']} 的修正分数与无缺口重算逐位一致")
        _, after = c.call("POST", "/api/cameras/0/frames", {"frame_no": 10, "features": cam0[10]})
        check(
            after["estimated"] is False
            and abs(after["score"] - reference(0, samples, 10)[-1]) < 1e-12,
            f"A#10 使用修正状态，实测出分 {after['score']:.4f}",
        )

        print("[5] 补算失败演练：原状态、缺口、已发分数保留，随后重试成功")
        b_fill = c.call("POST", "/api/cameras/1/fill", {"frame_no": 8, "features": cam1[8]})[1]
        check(b_fill["replay"]["to"] == 15, "先补完相机 B 的旧缺口 #8（回放到 #15）")
        c.call("POST", "/api/cameras/1/missing", {"frame_no": 16})
        c.call("POST", "/api/cameras/1/frames", {"frame_no": 17, "features": cam1[17]})
        before = c.call("GET", "/api/cameras/1/state")[1]
        status, err = c.call(
            "POST",
            "/api/cameras/1/fill",
            {"frame_no": 16, "features": cam1[16], "simulate_failure": True},
        )
        check(
            status == 422 and err["retryable"] is True and err["state_preserved"] is True,
            f"失败返回可重试且状态已保留（{status} {err.get('message', err)}）",
        )
        mid = c.call("GET", "/api/cameras/1/state")[1]
        check(mid["state_digest"] == before["state_digest"], "状态指纹未变")
        check(mid["gap"] == before["gap"], "缺口仍开放")
        check(
            [s["score"] for s in mid["scores"]] == [s["score"] for s in before["scores"]],
            "已发分数未变",
        )
        _, kept = c.call("POST", "/api/cameras/1/frames", {"frame_no": 18, "features": cam1[18]})
        check(kept["estimated"] is True, f"失败后仍按原状态出分 B#18 {kept['score']:.4f}（暂估）")
        status, retry = c.call("POST", "/api/cameras/1/fill", {"frame_no": 16, "features": cam1[16]})
        check(status == 200 and retry["replay"]["to"] == 18, "重试成功，回放追上 #18")
        print("\n交付核对通过。")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
