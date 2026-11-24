"use strict";

const state = {
  samples: null,
  busy: false,
};

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (typeof v === "boolean") {
      // Boolean attributes such as `disabled` must be absent, not "false".
      if (v) node.setAttribute(k, "");
    } else node.setAttribute(k, v);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
};
const fmt = (x, d = 4) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));

function toast(message) {
  const old = $(".toast");
  if (old) old.remove();
  const node = el("div", { class: "toast", text: message });
  document.body.append(node);
  setTimeout(() => node.remove(), 6000);
}

function log(kind, text) {
  const box = $("#log");
  box.prepend(el("div", { class: `line ${kind}`, text: `${new Date().toLocaleTimeString()} · ${text}` }));
  while (box.children.length > 200) box.lastChild.remove();
}

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail ? JSON.stringify(data.detail) : data.message || res.statusText;
    const err = new Error(`${res.status} ${detail}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function scoreColor(v) {
  if (v >= 0.8) return "var(--bad)";
  if (v >= 0.6) return "var(--warn)";
  return "var(--ok)";
}

function frameOf(camera, frameNo) {
  const cam = state.samples.cameras.find((c) => c.camera === camera);
  return cam ? cam.frames.find((f) => f.frame_no === frameNo) : undefined;
}
const camName = (camera) => {
  const cam = state.samples?.cameras.find((c) => c.camera === camera);
  return cam ? cam.name : `相机 ${camera}`;
};

function renderTable(cam, scores) {
  if (!scores.length) return el("div", { class: "empty", text: "尚未出分。可点「送入下一帧」开始。" });
  const table = el("table", {}, el("thead", {}, el("tr", {},
    el("th", { text: "帧号" }),
    el("th", { text: "已发分数" }),
    el("th", { text: "状态" }),
    el("th", { text: "修正分数" }),
    el("th", { text: "修正 Δ" }),
  )));
  const body = el("tbody");
  const latest = scores[scores.length - 1].frame_no;
  for (const s of scores) {
    const bar = el("div", { class: "scorebar" },
      el("span", { style: `width:${(s.score * 100).toFixed(1)}%;background:${s.estimated ? "var(--est)" : scoreColor(s.score)}` }));
    const tags = el("td", {},
      s.estimated ? el("span", { class: "tag est", text: "暂估" }) : el("span", { class: "tag live", text: "实测" }));
    if (s.pair_id) tags.append(" ", el("span", { class: "tag pair", title: `成对提交 ${s.pair_id}`, text: "成对" }));
    if (s.revision > 0) tags.append(" ", el("span", { class: "tag rev", text: `已补算 v${s.revision}` }));
    const revised = s.revised_score === null ? el("td", { class: "num", text: "—" })
      : el("td", { class: "num" }, el("div", { text: fmt(s.revised_score) }),
          el("div", { class: "scorebar" }, el("span", { style: `width:${(s.revised_score * 100).toFixed(1)}%;background:${scoreColor(s.revised_score)}` })));
    const delta = s.delta === null ? el("td", { class: "num", text: "—" })
      : el("td", { class: "num", style: `color:${Math.abs(s.delta) > 0.02 ? "var(--warn)" : "var(--muted)"}`, text: `${s.delta >= 0 ? "+" : ""}${fmt(s.delta)}` });
    body.append(el("tr", { class: s.frame_no === latest ? "latest" : "" },
      el("td", { class: "num", text: `#${s.frame_no}` }),
      el("td", {}, el("div", { class: "num", text: fmt(s.score) }), bar),
      tags, revised, delta));
  }
  table.append(body);
  return el("div", { class: "scroll" }, table);
}

function renderCamera(cam, snap) {
  const gap = snap?.gap;
  const replay = snap?.replay;
  const lastOut = snap?.last_output_frame ?? 0;
  const nextNo = (snap?.last_frame ?? 0) + 1;
  const badges = el("div", { class: "badges" },
    el("span", { class: "badge", text: `已发 ${snap?.scores.length ?? 0} 帧` }),
    gap ? el("span", { class: "badge gap", text: `缺口 #${gap.frame_no}` }) : null,
    replay ? el("span", { class: "badge replay", text: `补算中 → #${replay.cursor}` }) : null,
  );

  const controls = el("div", { class: "camera-controls" });
  const haveNext = !!frameOf(cam.camera, nextNo);
  controls.append(
    el("button", {
      "data-action": "send",
      text: `送入第 ${nextNo} 帧`,
      disabled: state.busy || !haveNext,
      onclick: () => sendFrame(cam.camera, nextNo),
    }),
    el("button", {
      "data-action": "missing",
      text: `标记第 ${nextNo} 帧缺失`,
      disabled: state.busy || !!gap || !haveNext,
      onclick: () => markMissing(cam.camera, nextNo),
    }),
  );
  if (gap) {
    const orig = frameOf(cam.camera, gap.frame_no);
    controls.append(
      el("button", {
        class: "accent",
        "data-action": "fill",
        text: `补交第 ${gap.frame_no} 帧原帧`,
        disabled: state.busy || !orig,
        onclick: () => fillGap(cam.camera, gap.frame_no),
      }),
      el("button", {
        class: "danger small",
        "data-action": "fill-fail",
        title: "演示补算失败：失败后原状态与已发分数保留，可直接点「补交」重试",
        text: "补算故障演练",
        disabled: state.busy,
        onclick: () => fillGap(cam.camera, gap.frame_no, undefined, true),
      }),
    );
  }
  controls.append(el("button", {
    class: "ghost grow",
    text: `跑完本路样例（到第 ${cam.frames.length} 帧）`,
    disabled: state.busy,
    onclick: () => runCameraToEnd(cam.camera),
  }));

  const meta = el("div", { class: "camera-meta" },
    `最新帧 #${snap?.last_frame ?? 0} · 最新已出分帧 #${lastOut} · `,
    el("span", { class: "statedigest", text: `状态 ${snap?.state_digest ?? "—"}` }),
  );

  return el("section", { class: "camera" },
    el("div", { class: "camera-head" },
      el("div", {}, el("h2", { class: "camera-title", text: cam.name }), meta),
      badges),
    el("div", { class: "camera-meta", text: cam.notes }),
    controls,
    renderTable(cam, snap?.scores ?? []));
}

async function refresh() {
  const snaps = await Promise.all(
    state.samples.cameras.map((cam) => api("GET", `/api/cameras/${cam.camera}/state`)),
  );
  const holder = $("#cameras");
  holder.replaceChildren(
    ...state.samples.cameras.map((cam, i) => {
      const node = renderCamera(cam, snaps[i]);
      node.dataset.camera = String(cam.camera);
      return node;
    }),
  );
  state.snaps = Object.fromEntries(snaps.map((s) => [s.camera, s]));
  renderPairPanel(snaps);
  await renderPairHistory();
}

function renderPairPanel(snaps) {
  const holder = $("#pair-fields");
  if (!holder) return;
  holder.replaceChildren(
    ...state.samples.cameras.map((cam) => {
      const snap = snaps.find((s) => s.camera === cam.camera);
      const nextNo = (snap?.last_frame ?? 0) + 1;
      const haveNext = !!frameOf(cam.camera, nextNo);
      return el("div", { class: "pair-cam" },
        el("div", { class: "pair-cam-head" },
          el("strong", { text: cam.name }),
          el("span", { class: "statedigest", text: `当前 #${snap?.last_frame ?? 0} · 版本 v${snap?.version ?? 0}${snap?.gap ? ` · 缺口 #${snap.gap.frame_no}` : ""}` }),
        ),
        el("div", { class: "pair-cam-body" },
          el("label", { class: "pair-label" }, "下一帧号",
            el("input", {
              class: "pair-frame", type: "number", min: "1", value: String(nextNo),
              "data-camera": String(cam.camera), disabled: state.busy,
            })),
          el("label", { class: "pair-label wide" }, "8 维特征",
            el("textarea", {
              class: "pair-features", rows: "2", spellcheck: "false",
              "data-camera": String(cam.camera), disabled: state.busy,
              placeholder: haveNext ? "点下方「用两路样例下一帧填充」" : "样例里没有该帧，请手动填写 8 个数字",
            })),
        ),
      );
    }),
  );
}

function pairInputs() {
  const rows = [...document.querySelectorAll("#pair-fields .pair-cam")].map((row) => {
    const camera = Number(row.querySelector(".pair-frame").dataset.camera);
    const frameNo = Number(row.querySelector(".pair-frame").value);
    const features = parseFeatures(row.querySelector(".pair-features").value);
    return { camera, frame_no: frameNo, features };
  });
  if (rows.length !== state.samples.cameras.length) {
    throw new Error("成对提交必须包含两路相机");
  }
  return rows;
}

async function fillPairFromSamples() {
  for (const cam of state.samples.cameras) {
    const snap = state.snaps?.[cam.camera];
    const nextNo = (snap?.last_frame ?? 0) + 1;
    const frame = frameOf(cam.camera, nextNo);
    const row = document.querySelector(`#pair-fields .pair-frame[data-camera="${cam.camera}"]`)?.closest(".pair-cam");
    if (!row) continue;
    row.querySelector(".pair-frame").value = String(nextNo);
    row.querySelector(".pair-features").value = frame ? frame.features.join(", ") : "";
  }
  toast("已按两路各自的下一帧填充样例");
}

async function submitPair() {
  await withBusy(async () => {
    const items = pairInputs();
    const delay = Number($("#pair-delay").value || 0);
    try {
      const r = await api("POST", "/api/pairs", { items, compute_delay_ms: delay });
      const desc = r.items.map((it) =>
        `${camName(it.camera)} #${it.frame_no} ${fmt(it.score)}${it.estimated ? "（暂估）" : ""}`).join("；");
      log("ok", `成对提交 ${r.pair_id} 两路同时生效：${desc}`);
    } catch (err) {
      if (err.status === 409 && err.data?.code === "pair_conflict") {
        const who = (err.data.conflicts || []).map((c) => `相机${c.camera}`).join("、") || "任一路";
        log("bad", `成对提交被整次退回：${who} 在计算期间有新分/补算/重置。请刷新后重试。`);
        toast(`并发冲突（409）：${who} 已变化，页面已刷新，请重试`);
      } else {
        throw err;
      }
    }
  });
}

async function renderPairHistory() {
  const holder = $("#pair-history");
  if (!holder) return;
  const data = await api("GET", "/api/pairs?limit=20");
  if (!data.pairs.length) {
    holder.replaceChildren(el("div", { class: "empty", text: "还没有成对提交。" }));
    return;
  }
  holder.replaceChildren(
    ...data.pairs.map((p) => el("div", { class: "pair-row" },
      el("span", { class: "tag pair", text: p.pair_id }),
      el("span", { class: "statedigest", text: p.submitted_at.replace("T", " ").replace("Z", " UTC") }),
      ...p.items.map((it) => el("span", { class: "pair-item" },
        `${camName(it.camera)} #${it.frame_no} `,
        el("strong", { text: fmt(it.score) }),
        it.estimated ? el("span", { class: "tag est", text: "暂估" }) : el("span", { class: "tag live", text: "实测" }),
      )),
    )),
  );
}

async function pairDemo() {
  if (state.busy) return;
  state.busy = true;
  $("#btn-pair-demo").disabled = true;
  try {
    await api("POST", "/api/reset", { confirm: true });
    await refresh();
    log("", "成对提交演示开始：每笔同时送两路下一帧，共送 4 笔（#1–#4）");
    for (let n = 1; n <= 4; n += 1) {
      const items = state.samples.cameras.map((cam) => ({
        camera: cam.camera,
        frame_no: n,
        features: frameOf(cam.camera, n).features,
      }));
      const r = await api("POST", "/api/pairs", { items });
      const desc = r.items.map((it) =>
        `${camName(it.camera)} #${it.frame_no} ${fmt(it.score)}${it.estimated ? "（暂估）" : ""}`).join("；");
      log("ok", `${r.pair_id} 两路同时生效：${desc}`);
      await refresh();
    }
    // One camera has a gap: its paired scores must be estimated, single flow
    // keeps working, then its fill revises the frames while the other camera
    // keeps publishing.
    log("", "在相机 A 打开缺口 #5（零向量暂估），随后两路继续成对提交");
    await api("POST", "/api/cameras/0/missing", { frame_no: 5 });
    await refresh();
    for (let n = 6; n <= 8; n += 1) {
      const items = state.samples.cameras.map((cam) => ({
        camera: cam.camera,
        frame_no: n,
        features: frameOf(cam.camera, n).features,
      }));
      const r = await api("POST", "/api/pairs", { items });
      const a = r.items.find((it) => it.camera === 0);
      const b = r.items.find((it) => it.camera === 1);
      log(a.estimated ? "warn" : "ok",
        `${r.pair_id}：相机 A #${a.frame_no} ${fmt(a.score)}（缺口未补，暂估）；相机 B #${b.frame_no} ${fmt(b.score)}（实测），一笔同时生效`);
      await refresh();
    }
    const fill = await api("POST", "/api/cameras/0/fill", {
      frame_no: 5, features: frameOf(0, 5).features, pace_ms: Number($("#pace").value || 0),
    });
    log("ok", `相机 A 补算完成：回放 #${fill.replay.from}–#${fill.replay.to}，已发分数未改、修正值并列；之后成对提交回到实测`);
    const next = 9;
    const r = await api("POST", "/api/pairs", {
      items: state.samples.cameras.map((cam) => ({
        camera: cam.camera, frame_no: next, features: frameOf(cam.camera, next).features,
      })),
    });
    const a9 = r.items.find((it) => it.camera === 0);
    log(a9.estimated ? "warn" : "ok",
      `${r.pair_id} 补算后首笔成对：相机 A #9 estimated=${a9.estimated}（应为 false，使用修正状态）`);
    await refresh();
    const history = await api("GET", "/api/pairs");
    log("", `成对记录共 ${history.count} 笔，每笔都含两路；双路汇总可查同一 pair_id。`);
  } catch (err) {
    toast(String(err.message || err));
    log("bad", `成对演示失败：${err.message || err}`);
  } finally {
    state.busy = false;
    $("#btn-pair-demo").disabled = false;
    await refresh().catch(() => {});
  }
}

async function conflictDemo() {
  if (state.busy) return;
  state.busy = true;
  $("#btn-conflict-demo").disabled = true;
  try {
    await api("POST", "/api/reset", { confirm: true });
    await refresh();
    // First pair succeeds so both streams sit at frame 1.
    const first = await api("POST", "/api/pairs", {
      items: state.samples.cameras.map((cam) => ({
        camera: cam.camera, frame_no: 1, features: frameOf(cam.camera, 1).features,
      })),
    });
    log("ok", `${first.pair_id} 成功，两路都到 #1`);
    // Slow pair: a single-camera ingest lands during its compute window.
    log("", "发起一笔带 800ms 延时的成对提交（#2 两路），期间从单路表单往相机 A 再送一帧 …");
    const slow = api("POST", "/api/pairs", {
      items: state.samples.cameras.map((cam) => ({
        camera: cam.camera, frame_no: 2, features: frameOf(cam.camera, 2).features,
      })),
      compute_delay_ms: 800,
    });
    await sleep(150);
    const single = await api("POST", "/api/cameras/0/frames", {
      frame_no: 2, features: frameOf(0, 2).features,
    });
    log("warn", `单路送帧照常生效：相机 A #2 出分 ${fmt(single.score)}（版本推进到 v${single.version}）`);
    let rejected = null;
    try {
      await slow;
    } catch (err) {
      rejected = err;
    }
    if (rejected?.status === 409 && rejected.data?.code === "pair_conflict") {
      const who = (rejected.data.conflicts || []).map((c) => `相机${c.camera}（${c.reason.split(";")[0]}）`).join("；");
      log("bad", `成对提交被整次退回 409：${who}。两路均未写入这笔，提示刷新后重试。`);
    } else {
      log("bad", "演练异常：慢成对提交没有按预期冲突");
    }
    await refresh();
    const snapA = state.snaps[0];
    const snapB = state.snaps[1];
    log("", `原子性核对：相机 A 最新 #${snapA.last_frame}（单路帧），相机 B 最新 #${snapB.last_frame}（仍是 #1，未被半笔写入）`);
    // Retry with refreshed next frames: A->3, B->2.
    const retry = await api("POST", "/api/pairs", {
      items: [
        { camera: 0, frame_no: 3, features: frameOf(0, 3).features },
        { camera: 1, frame_no: 2, features: frameOf(1, 2).features },
      ],
    });
    log("ok", `刷新后重试成功：${retry.pair_id} 两路同时生效（A #3、B #2）`);
  } catch (err) {
    toast(String(err.message || err));
    log("bad", `冲突演练失败：${err.message || err}`);
  } finally {
    state.busy = false;
    $("#btn-conflict-demo").disabled = false;
    await refresh().catch(() => {});
  }
}

async function withBusy(fn) {
  if (state.busy) return;
  state.busy = true;
  try { await fn(); }
  catch (err) { toast(String(err.message || err)); log("bad", `失败：${err.message || err}`); }
  finally {
    state.busy = false;
    await refresh().catch((e) => log("bad", `刷新失败：${e.message}`));
  }
}

function describeIngest(camera, r) {
  return `${camName(camera)} #${r.frame_no} 出分 ${fmt(r.score)}${r.estimated ? "（暂估）" : ""}`;
}

async function sendFrame(camera, frameNo) {
  const frame = frameOf(camera, frameNo);
  if (!frame) return toast(`样例里没有 ${camName(camera)} 第 ${frameNo} 帧`);
  await withBusy(async () => {
    const r = await api("POST", `/api/cameras/${camera}/frames`, { frame_no: frameNo, features: frame.features });
    log(r.estimated ? "warn" : "ok", describeIngest(camera, r) + (r.gap ? `（缺口 #${r.gap.frame_no} 仍待补）` : ""));
  });
}

async function markMissing(camera, frameNo) {
  await withBusy(async () => {
    const r = await api("POST", `/api/cameras/${camera}/missing`, { frame_no: frameNo });
    log("warn", `${camName(camera)} 第 ${frameNo} 帧缺失，已用零向量代入，出分 ${fmt(r.score)}（暂估）`);
  });
}

async function fillGap(camera, frameNo, paceMs, simulateFailure = false) {
  const frame = frameOf(camera, frameNo);
  if (!frame) return toast("样例里没有该帧原帧特征");
  await withBusy(async () => {
    const pace = paceMs ?? Number($("#pace").value || 0);
    log("", `${camName(camera)} 开始补算 #${frameNo}${simulateFailure ? "（故障演练）" : ""} …`);
    const r = await api("POST", `/api/cameras/${camera}/fill`, {
      frame_no: frameNo,
      features: frame.features,
      pace_ms: pace,
      simulate_failure: simulateFailure,
    });
    log("ok", `${camName(camera)} 补算完成：回放 #${r.replay.from}–#${r.replay.to}（${r.replay.frames} 帧，追上最新已出分帧）`);
    const changed = r.revisions.filter((x) => x.changed);
    log("", `${camName(camera)} 缺口帧 #${frameNo} ${fmt(r.gap_score.published_score)} → ${fmt(r.gap_score.revised_score)}；${changed.length} 帧修正值有变化，已发分数保持不变`);
  });
}

async function runCameraToEnd(camera) {
  const cam = state.samples.cameras.find((c) => c.camera === camera);
  const snap = await api("GET", `/api/cameras/${camera}/state`);
  let next = (snap.last_frame ?? 0) + 1;
  const last = cam.frames[cam.frames.length - 1].frame_no;
  if (next > last) return toast("本路样例已全部送入");
  await withBusy(async () => {
    while (next <= last) {
      const frame = frameOf(camera, next);
      const r = await api("POST", `/api/cameras/${camera}/frames`, { frame_no: next, features: frame.features });
      log(r.estimated ? "warn" : "ok", describeIngest(camera, r));
      next += 1;
    }
  });
}

async function resetAll() {
  await withBusy(async () => {
    await api("POST", "/api/reset", { confirm: true });
    log("", "已重置，重新载入两路样例");
  });
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

async function demo() {
  if (state.busy) return;
  state.busy = true;
  $("#btn-demo").disabled = true;
  try {
    await api("POST", "/api/reset", { confirm: true });
    await refresh();
    const pace = Number($("#pace").value || 0);
    log("", "演示开始：相机 A 暂缺第 4 帧，相机 B 暂缺第 8 帧");

    // Camera A: frames 1–3, skip 4, then 5–8.
    for (const n of [1, 2, 3]) {
      const r = await api("POST", "/api/cameras/0/frames", { frame_no: n, features: frameOf(0, n).features });
      log("ok", describeIngest(0, r));
    }
    let r = await api("POST", "/api/cameras/0/missing", { frame_no: 4 });
    log("warn", `相机 A #4 暂缺 → 零向量代入，出分 ${fmt(r.score)}（暂估）`);
    for (const n of [5, 6, 7, 8]) {
      const x = await api("POST", "/api/cameras/0/frames", { frame_no: n, features: frameOf(0, n).features });
      log("warn", describeIngest(0, x) + "（缺口未补，暂估）");
    }
    // Camera B: frames 1–7, skip 8, then 9–12.
    for (const n of [1, 2, 3, 4, 5, 6, 7]) {
      const x = await api("POST", "/api/cameras/1/frames", { frame_no: n, features: frameOf(1, n).features });
      log("ok", describeIngest(1, x));
    }
    r = await api("POST", "/api/cameras/1/missing", { frame_no: 8 });
    log("warn", `相机 B #8 暂缺 → 零向量代入，出分 ${fmt(r.score)}（暂估）`);
    for (const n of [9, 10, 11, 12]) {
      const x = await api("POST", "/api/cameras/1/frames", { frame_no: n, features: frameOf(1, n).features });
      log("warn", describeIngest(1, x) + "（缺口未补，暂估）");
    }
    await refresh();

    // Interleaved: start A's fill (paced), keep pushing B, and push A9 while
    // A is still replaying, so the replay target has to grow to catch up.
    log("", "相机 A 开始补算第 4 帧；补算期间相机 B 进第 13 帧，相机 A 自己也进第 9 帧");
    const fillA = api("POST", "/api/cameras/0/fill", {
      frame_no: 4, features: frameOf(0, 4).features, pace_ms: Math.max(pace, 150),
    });
    await sleep(90);
    const b13 = await api("POST", "/api/cameras/1/frames", { frame_no: 13, features: frameOf(1, 13).features });
    log("ok", describeIngest(1, b13) + "（补算 A 期间 B 照常出分，补算不受影响）");
    const a9 = await api("POST", "/api/cameras/0/frames", { frame_no: 9, features: frameOf(0, 9).features });
    log("warn", describeIngest(0, a9) + "（补算尚未追平，先按暂估状态出分）");
    const fillResult = await fillA;
    log("ok", `相机 A 补算完成：回放 #${fillResult.replay.from}–#${fillResult.replay.to}，追上最新已出分帧 #${fillResult.replay.target_frame}`);
    log("", `相机 A #4 ${fmt(fillResult.gap_score.published_score)} → ${fmt(fillResult.gap_score.revised_score)}；已发分数未改`);
    const a10 = await api("POST", "/api/cameras/0/frames", { frame_no: 10, features: frameOf(0, 10).features });
    log("ok", describeIngest(0, a10) + (a10.estimated ? "（暂估）" : "（已使用修正后的状态，非暂估）"));
    // Camera B's own gap is still open; fill it after the interleaved phase.
    const fillB = await api("POST", "/api/cameras/1/fill", {
      frame_no: 8, features: frameOf(1, 8).features, pace_ms: pace,
    });
    log("ok", `相机 B 补算完成：回放 #${fillB.replay.from}–#${fillB.replay.to}，#8 ${fmt(fillB.gap_score.published_score)} → ${fmt(fillB.gap_score.revised_score)}`);
    await refresh();
    log("", "演示结束。两路缺口均已补齐，已发分数保持不变。可继续手动点按，或重置后再试。");
  } catch (err) {
    toast(String(err.message || err));
    log("bad", `演示失败：${err.message || err}`);
  } finally {
    state.busy = false;
    $("#btn-demo").disabled = false;
    await refresh().catch(() => {});
  }
}

function parseFeatures(raw) {
  const parts = raw.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (parts.length !== 8 || parts.some((x) => !Number.isFinite(x))) {
    throw new Error(`需要正好 8 个数字，当前解析到 ${parts.length} 个`);
  }
  return parts;
}

function wireSubmitForm() {
  const camSel = $("#submit-camera");
  camSel.replaceChildren(...state.samples.cameras.map((c) => el("option", { value: String(c.camera), text: c.name })));
  const frameInput = $("#submit-frame");
  const featInput = $("#submit-features");

  const fillFromSample = () => {
    const camera = Number(camSel.value);
    const frame = frameOf(camera, Number(frameInput.value));
    if (!frame) return toast(`样例里没有相机 ${camera} 第 ${frameInput.value} 帧`);
    featInput.value = frame.features.join(", ");
  };
  $("#submit-samples").onclick = fillFromSample;

  const post = async () => {
    const camera = Number(camSel.value);
    const frameNo = Number(frameInput.value);
    await withBusy(async () => {
      const features = parseFeatures(featInput.value);
      const r = await api("POST", `/api/cameras/${camera}/frames`, { frame_no: frameNo, features });
      log(r.estimated ? "warn" : "ok", describeIngest(camera, r));
      frameInput.value = String(Number(r.frame_no) + 1);
    });
  };

  $("#submit-form").onsubmit = (ev) => { ev.preventDefault(); post(); };
  $("#submit-missing").onclick = async () => {
    const camera = Number(camSel.value);
    const frameNo = Number(frameInput.value);
    await withBusy(async () => {
      await api("POST", `/api/cameras/${camera}/missing`, { frame_no: frameNo });
      log("warn", `${camName(camera)} 第 ${frameNo} 帧标记为缺失（零向量代入，暂估）`);
      frameInput.value = String(Number(frameNo) + 1);
    });
  };
}

async function init() {
  try {
    const info = await api("GET", "/api/model");
    $("#modelbox").textContent = [
      `模型 ${info.name}`,
      `${info.features} 维输入 → ${info.hidden} 维隐状态 → 缺陷分数`,
      `权重指纹 ${info.weight_digest}`,
      "权重硬编码，运行时不训练",
    ].join("\n");
    state.samples = await api("GET", "/api/samples");
    await refresh();
    wireSubmitForm();
    $("#btn-reset").onclick = resetAll;
    $("#btn-clear-log").onclick = () => { $("#log").innerHTML = ""; };
    $("#btn-demo").onclick = demo;
    $("#btn-play-a").onclick = () => runCameraToEnd(0);
    $("#btn-pair-demo").onclick = pairDemo;
    $("#btn-conflict-demo").onclick = conflictDemo;
    $("#pair-submit").onclick = submitPair;
    $("#pair-fill-samples").onclick = fillPairFromSamples;
  } catch (err) {
    toast(`初始化失败：${err.message || err}`);
  }
}

init();
