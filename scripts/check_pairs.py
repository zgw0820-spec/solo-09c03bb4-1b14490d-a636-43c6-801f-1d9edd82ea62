#!/usr/bin/env python3
"""Deliverable check for paired two-view scoring and concurrency conflicts.

Starts the app on a free local port and drives the full pairing contract over
HTTP, printing a readable PASS/FAIL trace:

1. a paired submission scores both cameras atomically and returns one pair id
   that is attached to the score rows on *both* cameras;
2. each camera recurses from its own state; a camera with an open gap still
   publishes estimated scores through the paired endpoint, while the other
   camera's scores stay measured;
3. single-camera ingest and gap fills keep working while a pair is computed;
4. if either camera publishes a score, commits a replay or is reset while the
   pair is being computed, the whole pair is rejected (409 pair_conflict,
   refresh_required=true) and neither camera gets a frame from it;
5. after a refresh the pair can be retried and commits cleanly;
6. filling the gap revises the estimated paired scores but keeps the original
   published scores and their pair id;
7. the two-camera summary and the pair history never show a half pair.

Run with:  python scripts/check_pairs.py
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

    def __init__(self, base: str) -> None:
        self.base = base


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


def check(condition: bool, message: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + message)
    if not condition:
        raise SystemExit(1)


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


def pair(c0_no: int, c1_no: int, cam0: dict, cam1: dict, delay: float | None = None) -> dict:
    body = {"items": [
        {"camera": 0, "frame_no": c0_no, "features": cam0[c0_no]},
        {"camera": 1, "frame_no": c1_no, "features": cam1[c1_no]},
    ]}
    if delay is not None:
        body["compute_delay_ms"] = delay
    return body


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

        print("[1] 成对提交 #1、#2：两路原子生效，共用 pair_id")
        status, r1 = c.call("POST", "/api/pairs", pair(1, 1, cam0, cam1))
        check(status == 200 and r1["status"] == "paired", f"首笔成对提交成功 {r1.get('pair_id')}")
        status, r2 = c.call("POST", "/api/pairs", pair(2, 2, cam0, cam1))
        check(status == 200 and r2["pair_id"] != r1["pair_id"], "pair_id 单调且唯一")
        snap0 = c.call("GET", "/api/cameras/0/state")[1]
        snap1 = c.call("GET", "/api/cameras/1/state")[1]
        check(
            [s["pair_id"] for s in snap0["scores"]]
            == [s["pair_id"] for s in snap1["scores"]]
            == [r1["pair_id"], r2["pair_id"]],
            "同一个 pair_id 同时挂在两路分数上",
        )
        ref0 = reference(0, samples, 2)
        ref1 = reference(1, samples, 2)
        check(
            abs(snap0["scores"][1]["score"] - ref0[1]) < 1e-12
            and abs(snap1["scores"][1]["score"] - ref1[1]) < 1e-12,
            "两路各按自己的状态递推，分数与无缺口参照一致",
        )

        print("[2] 相机 A 打开缺口 #3，之后成对提交：A 暂估、B 实测，仍然一笔生效")
        c.call("POST", "/api/cameras/0/missing", {"frame_no": 3})
        status, p4 = c.call("POST", "/api/pairs", pair(4, 3, cam0, cam1))
        a4 = next(i for i in p4["items"] if i["camera"] == 0)
        b3 = next(i for i in p4["items"] if i["camera"] == 1)
        check(
            status == 200 and a4["estimated"] is True and b3["estimated"] is False,
            f"缺口相机 A#4 暂估 {a4['score']:.4f}，B#3 实测 {b3['score']:.4f}",
        )
        status, p5 = c.call("POST", "/api/pairs", pair(5, 4, cam0, cam1))
        check(status == 200 and next(i for i in p5["items"] if i["camera"] == 0)["estimated"] is True,
              "缺口未补期间 A 的成对新帧继续暂估")

        print("[3] 并发冲突：成对计算期间单路送帧，整笔 409 退回且两路都不写入")
        outcome: dict = {}

        def slow_pair() -> None:
            outcome["r"] = c.call("POST", "/api/pairs", pair(6, 5, cam0, cam1, delay=500))

        worker = threading.Thread(target=slow_pair)
        worker.start()
        time.sleep(0.12)
        # Single-camera path on camera B must stay available during compute.
        t0 = time.perf_counter()
        status, single = c.call("POST", "/api/cameras/1/frames",
                                {"frame_no": 5, "features": cam1[5]})
        elapsed = time.perf_counter() - t0
        check(status == 200 and elapsed < 2.0,
              f"成对计算期间单路送帧照常出分 B#5 {single['score']:.4f}（{elapsed*1000:.0f} ms）")
        worker.join(timeout=30)
        check(not worker.is_alive(), "慢成对请求已返回")
        sc, body = outcome["r"]
        check(
            sc == 409 and body["code"] == "pair_conflict" and body["refresh_required"] is True,
            f"整笔退回 409 pair_conflict（{body.get('conflicts')}）",
        )
        snap0b = c.call("GET", "/api/cameras/0/state")[1]
        snap1b = c.call("GET", "/api/cameras/1/state")[1]
        check(snap0b["last_frame"] == 5, "相机 A 未被退回的半笔写入（仍停在 #5）")
        check(snap1b["last_frame"] == 5 and len(snap1b["scores"]) == 5,
              "相机 B 只有自己单路送的 #5，没有成对提交的另一笔")
        pairs_now = c.call("GET", "/api/pairs")[1]
        check(pairs_now["count"] == 4, "成对历史里没有被退回的半笔")

        print("[4] 刷新后重试：按两路各自的下一帧（A#6、B#6）重新成对提交")
        status, retry = c.call("POST", "/api/pairs", pair(6, 6, cam0, cam1))
        check(status == 200, f"刷新后重试成功 {retry.get('pair_id')}")
        check(
            c.call("GET", "/api/cameras/0/state")[1]["scores"][5]["pair_id"] == retry["pair_id"]
            and c.call("GET", "/api/cameras/1/state")[1]["scores"][5]["pair_id"] == retry["pair_id"],
            "重试笔的 pair_id 再次同时落到两路",
        )

        print("[5] 重置冲突：成对计算期间 reset，整笔退回且成对历史清空")
        reset_outcome: dict = {}

        def slow_pair_2() -> None:
            reset_outcome["r"] = c.call("POST", "/api/pairs", pair(7, 7, cam0, cam1, delay=500))

        worker = threading.Thread(target=slow_pair_2)
        worker.start()
        time.sleep(0.12)
        c.call("POST", "/api/reset", {"confirm": True})
        worker.join(timeout=30)
        sc, body = reset_outcome["r"]
        check(sc == 409 and body["refresh_required"] is True, "reset 触发成对提交整笔退回")
        check(c.call("GET", "/api/pairs")[1]["count"] == 0, "重置后成对历史清空")
        check(all(cam["version"] >= 1 for cam in c.call("GET", "/api/status")[1]["cameras"]),
              "重置后版本号不归零，旧令牌不可能复活")

        print("[6] 缺口补算与成对提交衔接：暂估分被修正、已发分与 pair_id 保留")
        c.call("POST", "/api/cameras/0/frames", {"frame_no": 1, "features": cam0[1]})
        c.call("POST", "/api/cameras/0/frames", {"frame_no": 2, "features": cam0[2]})
        c.call("POST", "/api/cameras/0/missing", {"frame_no": 3})
        c.call("POST", "/api/cameras/0/frames", {"frame_no": 4, "features": cam0[4]})
        c.call("POST", "/api/cameras/1/frames", {"frame_no": 1, "features": cam1[1]})
        status, pp = c.call("POST", "/api/pairs", pair(5, 2, cam0, cam1))
        check(status == 200, "缺口期间成对提交成功")
        published_before = {s["frame_no"]: s["score"]
                            for s in c.call("GET", "/api/cameras/0/state")[1]["scores"]}
        status, fill = c.call("POST", "/api/cameras/0/fill",
                              {"frame_no": 3, "features": cam0[3]})
        check(status == 200 and fill["replay"]["to"] == 5, f"补算追平到 #5（{fill['replay']}）")
        snap = c.call("GET", "/api/cameras/0/state")[1]
        check(
            all(s["score"] == published_before[s["frame_no"]] for s in snap["scores"]),
            "已发分数逐帧未改",
        )
        rec5 = next(s for s in snap["scores"] if s["frame_no"] == 5)
        check(rec5["pair_id"] == pp["pair_id"] and rec5["revised_score"] is not None,
              "成对暂估帧带上修正值且 pair_id 保留")
        ref = reference(0, samples, 5)
        check(abs(rec5["final_score"] - ref[4]) < 1e-12, "修正分数与无缺口重算逐位一致")
        status, after = c.call("POST", "/api/pairs", pair(6, 3, cam0, cam1))
        a6 = next(i for i in after["items"] if i["camera"] == 0)
        check(status == 200 and a6["estimated"] is False,
              f"补算后的成对帧恢复实测 A#6 {a6['score']:.4f}")

        print("[7] 双路汇总与成对历史一致性：永远不会只看到一笔提交的一路")
        summary = c.call("GET", "/api/status")[1]
        cams = {x["camera"]: x for x in summary["cameras"]}
        history = c.call("GET", "/api/pairs")[1]
        check(summary["pairs"]["count"] == history["count"] == 2, "汇总与历史的成对笔数一致")
        for p in history["pairs"]:
            ids = {i["camera"] for i in p["items"]}
            check(ids == {0, 1}, f"{p['pair_id']} 同时包含两路")
        check(cams[0]["paired_published"] >= 1 and cams[1]["paired_published"] >= 1,
              "两路汇总都能数到成对出分帧")

        print("\n成对出分与并发冲突交付核对通过。")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
