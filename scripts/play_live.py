"""Local HTTP server: play live vs a trained Mushroom Wars v2 agent.

Holds one game session in memory: a `MushroomEnv` with the user as P1 and
a neural agent (loaded from a champion checkpoint or local weights) as P2.
The browser polls /api/state and POSTs /api/action to drive the user side.

This is a *local-only* dev tool — no auth, no concurrency. Run:

    python scripts/play_live.py
    python scripts/play_live.py --champion 1757a025  # pick by run_id prefix
    python scripts/play_live.py --level random_close_4_5 --port 8765

then open http://localhost:8765 in a browser.

Why a local server (and not the queue-based dashboard play.html): live play
needs ms-latency action handling, which the queue path doesn't deliver. The
existing play.html queues a *batch* match for the worker; this script runs
one game in this process so clicks get an immediate sim response.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force CPU before any torch / JAX import — the user is playing live, not
# training; CPU is plenty and avoids GPU contention with PaulLinux training.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ---------------------------------------------------------------------------
# Shared mutable state. Single-game, single-user — no need for any locking
# beyond the GIL. The handler reads/writes it on every request.
# ---------------------------------------------------------------------------

class _Session:
    # Real-time pacing. The sim advances on its own clock; the user's clicks
    # queue up and apply on the next decision tick. The agent (P2) picks its
    # action every decision tick too — so this is fully real-time, not turn-
    # based. 1 Hz gives time to actually look + react; the trained net was
    # trained against an env stepping at this same decision-interval cadence
    # (so cadence-wise the net is happy at any TICK_HZ — only wall-time
    # changes between training and live play).
    TICK_HZ = 1

    def __init__(self, level_name: str, champion_run_id: str | None):
        self.level_name = level_name
        self.champion_run_id = champion_run_id
        self.env = None
        self.opponent_label = "(unloaded)"
        self.lock = threading.Lock()
        # Thread-safe inbox of latest user actions. Each tick consumes up to
        # one. maxlen=4 buffers small bursts (rapid clicks) without going
        # unbounded.
        self._action_queue: collections.deque[int] = collections.deque(maxlen=4)
        self._ticker_thread: threading.Thread | None = None
        # Per-thread stop signal. The previous "single bool on the session"
        # design leaked ticker threads on reset: if the old ticker was mid-
        # sleep when reset() set running=False then immediately set it back
        # to True for the new ticker, the OLD thread woke up, saw True, and
        # kept ticking alongside the new one → double-rate sim. A per-thread
        # Event sidesteps that — the old thread's Event stays set, the new
        # thread has its own fresh Event.
        self._ticker_stop: threading.Event | None = None

    def reset(self, level_name: str | None = None, champion_run_id: str | None = None) -> None:
        # Stop the old ticker before re-initializing the env. The old thread
        # might still be holding self.lock — _stop_ticker waits for it.
        self._stop_ticker()

        if level_name is not None:
            self.level_name = level_name
        if champion_run_id is not None:
            self.champion_run_id = champion_run_id

        from sim.envs.mushroom_env import MushroomEnv
        from sim.envs.opponents import random_legal_opponent, make_neural_opponent

        if self.champion_run_id is None or self.champion_run_id == "random_legal":
            opponent = random_legal_opponent
            self.opponent_label = "random_legal"
        else:
            w_path, n_path, label = _resolve_champion(self.champion_run_id)
            opponent = make_neural_opponent(
                weights_path=w_path,
                obs_norm_path=n_path,
                device="cpu",
            )
            self.opponent_label = label

        self.env = MushroomEnv(level_name=self.level_name, opponent=opponent, seed=int(time.time()) & 0xFFFF)
        self.env.reset()

        # Drain any stale clicks queued before the reset.
        self._action_queue.clear()
        self._start_ticker()

    # ------------------------------------------------------------------
    # Real-time ticker
    # ------------------------------------------------------------------

    def _start_ticker(self) -> None:
        stop = threading.Event()
        self._ticker_stop = stop
        t = threading.Thread(target=self._tick_loop, args=(stop,),
                             daemon=True, name="mw-ticker")
        self._ticker_thread = t
        t.start()

    def _stop_ticker(self) -> None:
        stop = self._ticker_stop
        if stop is not None:
            stop.set()
        t = self._ticker_thread
        if t and t.is_alive():
            # Wait up to one full tick — long enough for a sleeping ticker
            # to wake up and notice its stop Event.
            t.join(timeout=1.5 / self.TICK_HZ)
        self._ticker_thread = None
        self._ticker_stop = None

    def _tick_loop(self, stop: threading.Event) -> None:
        """Background loop: advance the sim once per TICK_HZ regardless of
        whether the user has clicked. Each tick consumes one queued user
        action (or noop) and lets the agent take its turn via env.step().

        `stop` is this thread's own Event — _stop_ticker() sets it; we
        check before AND after sleep so a reset can interrupt us mid-wait
        rather than producing one ghost step on the way out."""
        from sim.actions import NOOP_INDEX
        from sim import config as C

        dt = 1.0 / self.TICK_HZ
        next_t = time.time()
        while not stop.is_set():
            # Sleep until our next tick boundary. Event.wait returns True if
            # the stop was signalled mid-sleep — bail immediately.
            wait = next_t - time.time()
            if wait > 0 and stop.wait(timeout=wait):
                return
            next_t += dt

            with self.lock:
                if stop.is_set():
                    return
                if self.env is None or self.env.state.phase != C.PHASE_PLAYING:
                    return
                a_idx = self._action_queue.popleft() if self._action_queue else NOOP_INDEX
                try:
                    self.env.step(int(a_idx))
                except Exception:
                    traceback.print_exc()
                    return

    def state_json(self) -> dict:
        if self.env is None:
            return {"phase": -1, "ready": False}
        st = self.env.state
        # Buildings: alive, owner, garrison/capacity, x/y for layout, type for icon.
        buildings = []
        for i in range(len(st.buildings_alive)):
            if not st.buildings_alive[i]:
                continue
            buildings.append({
                "slot":     int(i),
                "owner":    int(st.buildings_owner[i]),
                "garrison": int(st.buildings_garrison[i]),
                "capacity": int(st.buildings_capacity[i]),
                "x":        float(st.buildings_x[i]),
                "y":        float(st.buildings_y[i]),
                "type_id":  int(st.buildings_type[i]),
            })
        # Groups: in-flight squads.
        groups = []
        for i in range(len(st.groups_alive)):
            if not st.groups_alive[i]:
                continue
            groups.append({
                "owner":    int(st.groups_owner[i]),
                "src":      int(st.groups_src[i]),
                "tgt":      int(st.groups_tgt[i]),
                "count":    int(st.groups_count[i]),
                "progress": float(st.groups_progress[i]),
                "travel":   float(st.groups_travel[i]),
            })
        # Action mask — the browser uses this to disable illegal source clicks.
        from sim.actions import compute_mask
        from sim import config as C
        mask = compute_mask(st, C.OWNER_P1)
        return {
            "ready":       True,
            "tick":        int(st.tick),
            "phase":       int(st.phase),
            "level":       self.level_name,
            "opponent":    self.opponent_label,
            "champion_run_id": self.champion_run_id,
            "buildings":   buildings,
            "groups":      groups,
            # Action mask is huge (4097 bools) — only ship the legal indices.
            "legal_actions": [int(i) for i in range(mask.shape[0]) if mask[i]],
        }

    def enqueue_action(self, action_idx: int) -> dict:
        """Buffer the user's click for the ticker to consume on the next
        decision tick. Returns immediately — the actual sim step happens
        in the background thread within ~200 ms."""
        from sim import config as C
        if self.env is None:
            return {"error": "not initialized"}
        if self.env.state.phase != C.PHASE_PLAYING:
            return {"error": f"game already ended (phase={int(self.env.state.phase)})"}
        with self.lock:
            self._action_queue.append(int(action_idx))
        return {"ok": True, "queued": True, "queue_size": len(self._action_queue)}


def _resolve_champion(run_id_prefix: str) -> tuple[str, str | None, str]:
    """Pick a champion + return (weights_local_path, obs_norm_local_path, label).

    Accepts a run_id prefix (looks up in Supabase + downloads to /tmp), or a
    local path to weights.pt (sibling obs_norm.pt auto-detected).
    """
    p = Path(run_id_prefix)
    if p.exists() and p.is_file():
        sib = p.parent / "obs_norm.pt"
        return str(p), (str(sib) if sib.exists() else None), p.stem

    import tempfile
    import urllib.request
    from cli.db import connect
    from workers.worker import _public_url

    with connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, label, weights_url, obs_norm_url
              FROM runs
             WHERE id::text LIKE %s
             ORDER BY queued_at DESC LIMIT 1
            """,
            (run_id_prefix + "%",),
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"No run matches prefix {run_id_prefix!r}")
    rid, label, w_url, n_url = row

    work = Path(tempfile.mkdtemp(prefix="mw2-play-"))
    w_local = work / "weights.pt"
    _download(w_url, w_local)
    n_local: Path | None = None
    if n_url:
        n_local = work / "obs_norm.pt"
        _download(n_url, n_local)
    return str(w_local), (str(n_local) if n_local else None), f"{label} ({rid[:8]})"


def _list_champions() -> list[dict]:
    """Return the same list of finished, weights-bearing runs that play.html shows.

    Mirror's the dashboard query so the in-browser picker matches what's on
    the deployed page. Synthesises a "random_legal" entry at the top so the
    user can switch back to the baseline opponent without restarting.
    """
    from cli.db import connect

    with connect() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, label, result
              FROM runs
             WHERE weights_url IS NOT NULL
               AND id::text != '00000000-0000-0000-0000-000000000001'
               AND status = 'done'
             ORDER BY finished_at DESC NULLS LAST
             LIMIT 100
            """
        )
        rows = cur.fetchall()
    out: list[dict] = [{"run_id": "random_legal", "label": "random_legal (baseline)"}]
    for rid, label, result in rows:
        rate = result.get("rate") if isinstance(result, dict) else None
        suffix = f" ({rate * 100:.0f}%)" if isinstance(rate, (int, float)) else ""
        out.append({"run_id": rid, "label": f"{label}{suffix}"})
    return out


def _download(url_relpath: str, dst: Path) -> None:
    import urllib.request
    from workers.worker import _public_url
    url = _public_url(url_relpath)
    if url is None:
        raise SystemExit(f"Could not resolve URL for {url_relpath!r}")
    with urllib.request.urlopen(url, timeout=60) as r:
        dst.write_bytes(r.read())


# ---------------------------------------------------------------------------
# Embedded HTML — single file, no external deps, talks to /api/* on same host.
# ---------------------------------------------------------------------------

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mushroom Wars — play live</title>
<style>
  /* Full-viewport game shell. Canvas takes the screen; everything else
     floats over it. */
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
               font: 13px/1.4 ui-sans-serif, system-ui, sans-serif;
               background: #050610; color: #e6e8ef; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  button { background: rgba(31, 36, 51, 0.85); color: #e6e8ef;
           border: 1px solid #2c3242; padding: 6px 12px; border-radius: 5px;
           cursor: pointer; font: inherit; }
  button:hover { background: rgba(48, 56, 78, 0.95); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  select { background: #1f2330; color: #e6e8ef; border: 1px solid #2c3242;
           padding: 6px 10px; border-radius: 5px; font: inherit; }
  canvas { display: block; position: fixed; top: 0; left: 0;
           width: 100vw; height: 100vh; background: #050610;
           cursor: crosshair; }
  /* Faction emblems pinned to canvas corners. */
  .emblem-corner { position: fixed; width: 52px; height: 52px; opacity: 0.85;
                   pointer-events: none; filter: drop-shadow(0 0 10px rgba(0,0,0,0.7));
                   z-index: 5; }
  .emblem-corner.tl { top: 16px;    left: 16px;  }
  .emblem-corner.tr { top: 16px;    right: 16px; }
  /* Top HUD strip — minimal: opponent label + new-game button. */
  .top-hud { position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
             display: flex; gap: 16px; align-items: center; z-index: 6;
             padding: 8px 16px;
             background: rgba(14, 17, 23, 0.55);
             border: 1px solid rgba(96, 165, 250, 0.12);
             border-radius: 999px;
             backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
  .top-hud .opp-label { color: #93a3b8; font-size: 12px; }
  .top-hud .opp-label .mono { color: #e6e8ef; }
  /* Compact floating Send-size widget at top-right. The full sidebar was
     too much real estate; this keeps just the gameplay-critical control. */
  .send-size { position: fixed; top: 12px; right: 16px; z-index: 6;
               display: flex; align-items: center; gap: 8px;
               padding: 8px 12px;
               background: rgba(14, 17, 23, 0.55);
               border: 1px solid rgba(96, 165, 250, 0.12);
               border-radius: 999px;
               backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
  .send-size .send-label { color: #93a3b8; font-size: 11px;
                           text-transform: uppercase; letter-spacing: 0.06em; }
  .send-size .pct-row { display: flex; gap: 4px; margin: 0; }
  .pct-row button { padding: 6px 12px; font-size: 12px; font-weight: 500; min-width: 48px; }
  .pct-row button.active { background: #f59e0b; border-color: #f59e0b; color: #11141b; }
  /* Bottom unit-balance bar. Two flex divs, widths proportional to unit %. */
  .balance-bar { position: fixed; bottom: 16px; left: 16px; right: 16px;
                 height: 28px; display: flex; z-index: 4;
                 border-radius: 999px; overflow: hidden;
                 background: rgba(14, 17, 23, 0.55);
                 border: 1px solid rgba(96, 165, 250, 0.12);
                 backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
                 font-size: 12px; font-weight: 600;
                 box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
  .balance-fill { display: flex; align-items: center; justify-content: center;
                  white-space: nowrap; transition: width 0.3s ease;
                  text-shadow: 0 1px 2px rgba(0,0,0,0.7); }
  .balance-fill.p1 { background: linear-gradient(90deg, #ef4444, #b91c1c); color: #fff;
                     padding-left: 12px; justify-content: flex-start; }
  .balance-fill.p2 { background: linear-gradient(90deg, #1e40af, #3b82f6); color: #fff;
                     padding-right: 12px; justify-content: flex-end; }
  .balance-fill.neutral { background: rgba(107, 114, 128, 0.3); color: #93a3b8; }
  /* Toast (transient notification). */
  .toast { position: fixed; top: 80px; left: 50%; transform: translateX(-50%);
           background: rgba(31, 36, 51, 0.95); padding: 8px 16px; border-radius: 5px;
           border: 1px solid #2c3242; opacity: 0; transition: opacity 0.2s;
           pointer-events: none; z-index: 10; }
  .toast.show { opacity: 1; }
  /* Start / game-over overlay — modal-like, covers the canvas. */
  .overlay { position: fixed; inset: 0; z-index: 20;
             display: flex; align-items: center; justify-content: center;
             background: rgba(5, 6, 16, 0.78);
             backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
  .overlay-card { background: rgba(17, 20, 27, 0.95); border: 1px solid rgba(96, 165, 250, 0.2);
                  border-radius: 12px; padding: 32px 36px; min-width: 360px; max-width: 480px;
                  box-shadow: 0 12px 60px rgba(0,0,0,0.6); }
  .overlay-card h1 { margin: 0 0 4px; font-size: 24px; color: #e6e8ef; font-weight: 600; }
  .overlay-card .subtitle { color: #93a3b8; font-size: 13px; margin-bottom: 24px; }
  .overlay-card .result { font-size: 28px; font-weight: 700; margin-bottom: 12px; }
  .overlay-card .result.win  { color: #34d399; }
  .overlay-card .result.lose { color: #f87171; }
  .overlay-card .result.draw { color: #93a3b8; }
  .overlay-card label { display: block; color: #93a3b8; font-size: 11px;
                        text-transform: uppercase; letter-spacing: 0.06em; margin: 14px 0 6px; }
  .overlay-card select { width: 100%; }
  .overlay-card .play-btn { width: 100%; padding: 12px; font-size: 15px; font-weight: 600;
                            background: #f59e0b; color: #11141b; border-color: #f59e0b;
                            margin-top: 24px; }
  .overlay-card .play-btn:hover { background: #fbbf24; }
  .overlay-card .loading { color: #93a3b8; text-align: center; padding: 20px; font-size: 13px; }
  .overlay-card .loading .spinner { display: inline-block; width: 14px; height: 14px;
                                    border: 2px solid #2c3242; border-top-color: #60a5fa;
                                    border-radius: 50%; animation: spin 1s linear infinite;
                                    vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <!-- Game canvas fills the viewport; everything else floats on top. -->
  <canvas id="board"></canvas>

  <!-- Minimal top HUD: shows current opponent + map. New-game returns to overlay. -->
  <div class="top-hud">
    <span class="opp-label">vs <span class="mono" id="opp-display">—</span></span>
    <span class="opp-label">on <span class="mono" id="level-label">—</span></span>
    <button id="btn-reset">New game</button>
  </div>

  <!-- Floating Send-size widget. The rest of the sidebar (Session / State
       counters) was removed per feedback — they were noise during play.
       Session win/loss tracking still happens in localStorage; we just
       don't surface it on the game screen. -->
  <div class="send-size">
    <span class="send-label">Send</span>
    <div class="pct-row" id="pct-row">
      <!-- buttons inserted at runtime from SEND_PERCENTAGES (25/50/75/100) -->
    </div>
  </div>

  <!-- Bottom unit-balance bar — proportional to YOUR vs ENEMY units only
       (neutral excluded; the bar represents the war between you two). -->
  <div class="balance-bar" id="balance-bar">
    <div class="balance-fill p1" id="balance-p1" style="width: 50%">—</div>
    <div class="balance-fill p2" id="balance-p2" style="width: 50%">—</div>
  </div>

  <!-- Toast for transient messages. -->
  <div id="toast" class="toast"></div>

  <!-- Start / game-over overlay — covers the canvas; user picks opponent + clicks Play. -->
  <div class="overlay" id="overlay">
    <div class="overlay-card">
      <h1>🍄 Mushroom Wars</h1>
      <div class="subtitle">Real-time space conquest vs a trained agent</div>

      <div id="result-line" style="display:none"></div>

      <label for="overlay-opp">Opponent</label>
      <select id="overlay-opp" class="mono">
        <option value="">loading…</option>
      </select>

      <label for="overlay-level">Map</label>
      <select id="overlay-level" class="mono">
        <option value="random_close_4_5" selected>random_close_4_5  (small, 4–5 worlds)</option>
        <option value="random_close_4_6">random_close_4_6  (small, 4–6 worlds)</option>
        <option value="random_close_4_8">random_close_4_8  (small, 4–8 worlds)</option>
        <option value="random_4_5">random_4_5  (large, 4–5 worlds)</option>
        <option value="random_4_8">random_4_8  (large, 4–8 worlds)</option>
        <option value="crossroads_6">crossroads_6  (fixed map)</option>
      </select>

      <button class="play-btn" id="play-btn" disabled>
        <span class="loading"><span class="spinner"></span>Loading…</span>
      </button>
    </div>
  </div>

<script>
// Canvas-only state. The server is the source of truth; we just paint.
const CV  = document.getElementById('board');
const CTX = CV.getContext('2d');
const $opp = document.getElementById('overlay-opp');     // pre-game opponent picker
const $level = document.getElementById('overlay-level'); // pre-game map picker
const $oppDisplay = document.getElementById('opp-display');  // current opponent label in HUD
const $lvl = document.getElementById('level-label');
const $toast = document.getElementById('toast');
const $overlay = document.getElementById('overlay');
const $resultLine = document.getElementById('result-line');
const $playBtn = document.getElementById('play-btn');

// Match canvas resolution to viewport (hi-DPI safe). Re-runs on resize.
function fitCanvas() {
  const dpr = window.devicePixelRatio || 1;
  CV.width  = Math.floor(window.innerWidth  * dpr);
  CV.height = Math.floor(window.innerHeight * dpr);
  CTX.setTransform(dpr, 0, 0, dpr, 0, 0);
}
fitCanvas();
window.addEventListener('resize', () => { fitCanvas(); if (lastState) render(lastState); });

// Selected source building slot (null = nothing picked).
let selectedSrc = null;
let lastState = null;
let pollHandle = null;

// Sim coordinate → canvas pixel. Auto-fits the bounding box of all live
// buildings into the viewport so the play area always fills the screen
// — fixed-extent rendering left small maps clustered in a corner.
// The bbox is frozen at game start (resetBbox()) to avoid planets
// jumping around as buildings get captured / destroyed.
const CANVAS_PAD = 120;
let bbox = null;  // {minX, maxX, minY, maxY}
function resetBbox() { bbox = null; }
function ensureBbox() {
  if (bbox) return;
  const bs = lastState?.buildings ?? [];
  if (!bs.length) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const b of bs) {
    minX = Math.min(minX, b.x); maxX = Math.max(maxX, b.x);
    minY = Math.min(minY, b.y); maxY = Math.max(maxY, b.y);
  }
  bbox = { minX, maxX, minY, maxY };
}
function simToCanvas(x, y) {
  ensureBbox();
  const w = window.innerWidth, h = window.innerHeight;
  if (!bbox) return [w / 2, h / 2];
  const bbW = (bbox.maxX - bbox.minX) || 1;
  const bbH = (bbox.maxY - bbox.minY) || 1;
  const scale = Math.min((w - 2 * CANVAS_PAD) / bbW, (h - 2 * CANVAS_PAD) / bbH);
  const offsetX = (w - bbW * scale) / 2;
  const offsetY = (h - bbH * scale) / 2;
  return [offsetX + (x - bbox.minX) * scale, offsetY + (y - bbox.minY) * scale];
}

function ownerColor(owner) {
  // Saturated red P1 / saturated blue P2 / dim gray neutral.
  return ['#6b7280', '#ef4444', '#3b82f6'][owner] || '#6b7280';
}

function radiusFor(building) {
  // Bigger capacity → larger sprite. Scaled to viewport so planets stay
  // proportional on big and small screens; with the auto-fit bbox the
  // base radius can be larger because each planet has room to breathe.
  const scale = Math.min(window.innerWidth, window.innerHeight) / 720;
  return Math.min(72, 28 + building.capacity * 0.4) * scale;
}

// ── Asset preloader ──────────────────────────────────────────────────────
// Sprite images are served by play_live.py at /assets/<name>.png. Drawn
// onto the canvas in render() once loaded. Missing assets fall back to
// solid colored circles so the page still works during a slow first-load.
const ASSETS = {};
const ASSET_NAMES = [
  'background',
  'planet-human', 'planet-alien', 'planet-neutral',
  'fighter-human', 'fighter-alien',
  'capture-fx',
];
function loadAssets() {
  return Promise.all(ASSET_NAMES.map(n => new Promise(res => {
    const img = new Image();
    img.onload  = () => { ASSETS[n] = img; res(); };
    img.onerror = () => { console.warn('asset missing:', n); res(); };
    img.src = `/assets/${n}.png`;
  })));
}

// (Capture FX removed in favour of cleaner visuals.)

function showToast(msg) {
  $toast.textContent = msg;
  $toast.classList.add('show');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => $toast.classList.remove('show'), 1500);
}

// (Old in-canvas phase banner replaced by the start/game-over overlay below.)
function showOverlay(opts) {
  // opts: { result: 'win'|'lose'|'draw'|null, playLabel: string }
  $resultLine.style.display = 'none';
  if (opts?.result) {
    const txt = { win: 'YOU WIN', lose: 'AGENT WINS', draw: 'DRAW' }[opts.result] || '';
    $resultLine.textContent = txt;
    $resultLine.className = `result ${opts.result}`;
    $resultLine.style.display = 'block';
  }
  $playBtn.textContent = opts?.playLabel ?? 'Play';
  // Re-enable the button. Previous click handler leaves it disabled on
  // success (intentionally — we don't want a double-click queueing a
  // second reset); resetting state here means the next time the overlay
  // appears, Play actually works.
  $playBtn.disabled = false;
  $overlay.style.display = 'flex';
}
function hideOverlay() { $overlay.style.display = 'none'; }

// Single planet sprite for all owners — ownership is conveyed by the
// colour tint, not by swapping the planet icon. Keeps the world coherent
// (the planet doesn't morph when captured, only changes flag).
const PLANET_SPRITE = 'planet-neutral';
const FIGHTER_BY_OWNER = ['fighter-human', 'fighter-human', 'fighter-alien'];  // owner 0 never moves

// Strong color tint so P1/P2 read at a glance even at small sizes. Applied
// with source-atop blend so it stays clipped to the sprite pixels (the
// circular planet shape) rather than bleeding into the surrounding canvas.
const TINT_BY_OWNER = [
  null,                       // neutral — no tint
  'rgba(239, 68, 68, 0.55)',  // P1 red
  'rgba(59, 130, 246, 0.55)', // P2 blue
];

function render(state) {
  // Background fills the entire viewport.
  const W = window.innerWidth, H = window.innerHeight;
  if (ASSETS['background']) {
    CTX.drawImage(ASSETS['background'], 0, 0, W, H);
  } else {
    CTX.fillStyle = '#050610';
    CTX.fillRect(0, 0, W, H);
  }
  if (!state || !state.ready) return;

  // Buildings — single planet sprite, tinted in owner colour, with selected
  // halo. We draw inside a circular clip so the sprite's black square
  // background is masked away (the PNGs aren't actually transparent).
  for (const b of state.buildings) {
    const [cx, cy] = simToCanvas(b.x, b.y);
    const r = radiusFor(b);
    const img = ASSETS[PLANET_SPRITE];
    if (img) {
      CTX.save();
      CTX.beginPath();
      CTX.arc(cx, cy, r, 0, Math.PI * 2);
      CTX.clip();
      CTX.drawImage(img, cx - r, cy - r, r * 2, r * 2);
      const tint = TINT_BY_OWNER[b.owner];
      if (tint) {
        CTX.fillStyle = tint;
        CTX.fillRect(cx - r, cy - r, r * 2, r * 2);
      }
      CTX.restore();
    } else {
      // Fallback: colored disc.
      CTX.beginPath();
      CTX.arc(cx, cy, r, 0, Math.PI * 2);
      CTX.fillStyle = ownerColor(b.owner);
      CTX.fill();
    }
    // Owner ring — colored outline so ownership reads clearly even when
    // the 40-55%-alpha tint washes out against the dark planet sprite at
    // small sizes. Neutrals get no ring — keeps the "yours / theirs /
    // neutral" trichotomy visually unambiguous.
    if (b.owner !== 0) {
      CTX.beginPath();
      CTX.arc(cx, cy, r + 2, 0, Math.PI * 2);
      CTX.strokeStyle = ownerColor(b.owner);
      CTX.lineWidth = 3;
      CTX.stroke();
    }
    // Selected-source halo — yellow glow ring outside the planet.
    if (b.slot === selectedSrc) {
      CTX.save();
      CTX.shadowColor = '#facc15';
      CTX.shadowBlur = 18;
      CTX.beginPath();
      CTX.arc(cx, cy, r + 5, 0, Math.PI * 2);
      CTX.strokeStyle = '#facc15';
      CTX.lineWidth = 3;
      CTX.stroke();
      CTX.restore();
    }
    // Garrison count — divided by 10 (Mushroom-Wars-style condensed unit
    // count; the underlying sim still uses 0..300 internally).
    // Optimistic decrement: subtract any of our own pending sends from
    // this slot that the server hasn't confirmed yet. The server ticks
    // at 1 Hz, so click→server-apply is up to ~1 s; without this the
    // source planet's count stays at full while the fighter visually
    // flies away, then snaps down after arrival. Stops once the
    // tracker entry flips serverConfirmed (server reflected the drop).
    let displayedInternal = b.garrison;
    for (const tr of fighterTracker.values()) {
      if (tr.srcSlot === b.slot && !tr.serverConfirmed) {
        displayedInternal -= tr.pendingAmount || 0;
      }
    }
    if (displayedInternal < 0) displayedInternal = 0;
    const garrisonDisplay = Math.round(displayedInternal / 10);
    CTX.font = `bold ${Math.max(14, r * 0.55)}px ui-monospace, monospace`;
    CTX.textAlign = 'center';
    CTX.textBaseline = 'middle';
    CTX.fillStyle = 'rgba(0,0,0,0.85)';
    CTX.fillText(String(garrisonDisplay), cx + 1, cy + 1);
    CTX.fillStyle = '#fff';
    CTX.fillText(String(garrisonDisplay), cx, cy);
  }

  // Fighters — driven entirely by fighterTracker, NOT state.groups. The
  // tracker holds one entry per live (or recently-landed) ship. Each
  // frame we compute (frac = elapsedSec / durationSec) and lerp position
  // between cached src/tgt. Once frac >= 1 the entry is reaped + an
  // impact flash spawned.
  const nowMs = performance.now();
  for (const [, tr] of fighterTracker) {
    const elapsedSec = (nowMs - tr.startTime) / 1000;
    const frac = Math.min(1, elapsedSec / tr.durationSec);
    const cx0_sim = tr.srcX + (tr.tgtX - tr.srcX) * frac;
    const cy0_sim = tr.srcY + (tr.tgtY - tr.srcY) * frac;
    const [cx0, cy0] = simToCanvas(cx0_sim, cy0_sim);
    const [tcx, tcy] = simToCanvas(tr.tgtX, tr.tgtY);
    const angle = Math.atan2(tcy - cy0, tcx - cx0);
    const spriteName = FIGHTER_BY_OWNER[tr.owner];
    const img = ASSETS[spriteName];
    const ownerCol = ownerColor(tr.owner);
    // One small ship per "displayed" unit (count÷10), capped at 30 to
    // match the per-planet capacity. Earlier cap of 8 made a 30-unit
    // send look identical to an 8-unit one. Lay out in rows of up to
    // SHIPS_PER_ROW perpendicular to travel, stacking additional rows
    // BEHIND the lead row so big squadrons read as a deep formation
    // rather than overflowing horizontally.
    const numShips = Math.min(30, Math.max(1, Math.round(tr.count / 10)));
    const baseSize = Math.max(16, radiusFor({capacity: 0}) * 0.5);
    const SHIPS_PER_ROW = 6;
    const rowCount = Math.ceil(numShips / SHIPS_PER_ROW);
    const perpSpacing = baseSize * 0.7;
    const longSpacing = baseSize * 0.85;
    const perpX = -Math.sin(angle), perpY = Math.cos(angle);
    const longX =  Math.cos(angle), longY = Math.sin(angle);
    for (let i = 0; i < numShips; i++) {
      // The last row may be partial — centre it on its own width so it
      // sits neatly behind the full rows rather than offset to one side.
      const rowIdx = Math.floor(i / SHIPS_PER_ROW);
      const colIdx = i % SHIPS_PER_ROW;
      const inThisRow = (rowIdx === rowCount - 1)
        ? (numShips - rowIdx * SHIPS_PER_ROW) : SHIPS_PER_ROW;
      const perpOffset = (colIdx - (inThisRow - 1) / 2) * perpSpacing;
      const longOffset = -rowIdx * longSpacing;   // negative = behind lead row
      const cx = cx0 + perpX * perpOffset + longX * longOffset;
      const cy = cy0 + perpY * perpOffset + longY * longOffset;
      if (img) {
        CTX.save();
        CTX.translate(cx, cy);
        CTX.rotate(angle + Math.PI / 2);
        CTX.beginPath();
        CTX.arc(0, 0, baseSize / 2, 0, Math.PI * 2);
        CTX.clip();
        CTX.drawImage(img, -baseSize / 2, -baseSize / 2, baseSize, baseSize);
        // Owner-color tint on top of the sprite — same source-atop trick
        // we use on planets. Without this the human ship reads as
        // metallic-gray at small sizes; users can't tell P1 from P2.
        CTX.fillStyle = ownerCol;
        CTX.globalAlpha = 0.55;
        CTX.fillRect(-baseSize / 2, -baseSize / 2, baseSize, baseSize);
        CTX.globalAlpha = 1;
        CTX.restore();
      } else {
        CTX.beginPath();
        CTX.arc(cx, cy, baseSize / 3, 0, Math.PI * 2);
        CTX.fillStyle = ownerCol;
        CTX.fill();
      }
    }
  }

  // Impact flashes — short-lived expanding ring at the target.
  for (const fx of impactEffects) {
    const t = (nowMs - fx.startedAt) / IMPACT_LIFE_MS;
    if (t > 1) continue;
    const [cx, cy] = simToCanvas(fx.x, fx.y);
    const baseR = radiusFor({capacity: 100});
    const r = baseR * (1 + 0.6 * t);
    CTX.save();
    CTX.globalAlpha = 1 - t;
    CTX.beginPath();
    CTX.arc(cx, cy, r, 0, Math.PI * 2);
    CTX.strokeStyle = ownerColor(fx.owner);
    CTX.lineWidth = 3;
    CTX.stroke();
    CTX.restore();
  }
}
let lastStateTime = 0;   // performance.now() at last poll

// ── Pure-UI fighter animation layer ────────────────────────────────────
// The sim is the source of truth for *game logic*. The UI animation here
// is decoupled — its only job is to draw fighter motion in a way that
// reads naturally to a human (full journey from source to target,
// regardless of how short the sim makes it).
//
// Two important UX choices:
//   1) Animation starts at the SOURCE planet, not at the sim's reported
//      current progress. The server ticks every ~1s; by the time we poll
//      a fighter has often already moved ~7% in sim — if we started
//      rendering at that progress the user would see the ship "teleport"
//      partway. Starting fresh at src is correct visually.
//   2) Animation duration is clamped to a minimum so very short journeys
//      still show motion. Otherwise short hops (travel=2 ticks = 1s sim)
//      would be over too fast to see.
//
// Cost: the UI fighter may visually arrive slightly later than the sim
// actually applies impact. We let the UI animation finish + then play a
// brief impact flash at the target — the user perceives the journey end
// as the moment of landing, not the sim's silent damage application.
const TICKS_PER_SEC_WALL = 2;
const MIN_ANIM_SEC       = 0.9;   // shortest visible fighter journey
const IMPACT_LIFE_MS     = 500;   // arrival-flash duration
const fighterTracker = new Map();
const impactEffects  = [];        // [{x, y, owner, startedAt}]
function resetFighterTracker() {
  fighterTracker.clear();
  impactEffects.length = 0;
}
function fighterKey(g) { return `${g.owner}-${g.src}-${g.tgt}`; }
function trackFighters(state) {
  if (!state) return;
  const now = performance.now();
  const bById = {};
  for (const b of state.buildings) bById[b.slot] = b;

  // 1. Ingest new groups — snapshot src/tgt positions + start animation.
  const seenKeys = new Set();
  for (const g of state.groups || []) {
    const k = fighterKey(g);
    seenKeys.add(k);
    if (fighterTracker.has(k)) {
      // We already client-spawned this on click; server has now confirmed
      // it. Flip serverConfirmed so the optimistic source-garrison
      // decrement stops being applied — b.garrison from the server now
      // reflects the drop on its own, applying both would double-count.
      fighterTracker.get(k).serverConfirmed = true;
      continue;
    }
    const src = bById[g.src], tgt = bById[g.tgt];
    if (!src || !tgt) continue;
    const durationSec = Math.max(g.travel / TICKS_PER_SEC_WALL, MIN_ANIM_SEC);
    fighterTracker.set(k, {
      owner: g.owner, count: g.count,
      srcSlot: g.src, tgtSlot: g.tgt,
      srcX: src.x, srcY: src.y, tgtX: tgt.x, tgtY: tgt.y,
      startTime: now,
      durationSec,
      landed: false,
      pendingAmount: 0,           // server-originated → no optimistic deduction
      serverConfirmed: true,
    });
  }

  // 2. Mark groups that vanished as landed (sim already applied impact).
  //    They keep animating in the UI until they reach frac=1, then we
  //    spawn an impact effect at the target and delete the entry.
  for (const [k, tr] of fighterTracker) {
    if (!seenKeys.has(k) && !tr.landed) {
      tr.landed = true;
    }
  }
}
function reapFinishedFighters() {
  const now = performance.now();
  for (const [k, tr] of [...fighterTracker]) {
    const elapsedSec = (now - tr.startTime) / 1000;
    const frac = elapsedSec / tr.durationSec;
    if (frac >= 1.0) {
      // Spawn impact flash + drop the tracker entry.
      impactEffects.push({ x: tr.tgtX, y: tr.tgtY, owner: tr.owner, startedAt: now });
      fighterTracker.delete(k);
    }
  }
  // Cull expired impact flashes.
  for (let i = impactEffects.length - 1; i >= 0; i--) {
    if (now - impactEffects[i].startedAt > IMPACT_LIFE_MS) {
      impactEffects.splice(i, 1);
    }
  }
}

function updateSidebar(state) {
  if (!state || !state.ready) return;
  let mineU = 0, enemyU = 0;
  for (const b of state.buildings) {
    if (b.owner === 1) mineU  += b.garrison;
    if (b.owner === 2) enemyU += b.garrison;
  }
  let mineF = 0, enemyF = 0;
  for (const g of state.groups) {
    if (g.owner === 1) mineF  += g.count;
    if (g.owner === 2) enemyF += g.count;
  }
  $oppDisplay.textContent = state.opponent ?? '—';
  $lvl.textContent = state.level ?? '—';

  // Bottom balance bar: width is proportional (P1 vs P2 only, neutral
  // excluded) but the LABEL shows the raw count ÷10 so the user can read
  // absolute strength, not just relative share.
  const totMine  = mineU + mineF;
  const totEnemy = enemyU + enemyF;
  const total = totMine + totEnemy || 1;
  const pctP1 = Math.round(100 * totMine / total);
  const pctP2 = 100 - pctP1;
  const $p1 = document.getElementById('balance-p1');
  const $p2 = document.getElementById('balance-p2');
  $p1.style.width = pctP1 + '%';
  $p2.style.width = pctP2 + '%';
  const p1Display = Math.round(totMine  / 10);
  const p2Display = Math.round(totEnemy / 10);
  $p1.textContent = pctP1 > 6 ? `${p1Display}` : '';
  $p2.textContent = pctP2 > 6 ? `${p2Display}` : '';

  // Game-over → show the overlay on the RISING EDGE (phase transitioning
  // from 0 → non-0). Using rising-edge avoids the race where a stale poll
  // response (still phase=non-0 because it was generated before the
  // /api/reset hit the server) re-opens the overlay right after the user
  // clicked Play Again.
  const prevPhase = updateSidebar._prevPhase ?? 0;
  if (prevPhase === 0 && state.phase !== 0) {
    const resultKind = state.phase === 1 ? 'win' : state.phase === 2 ? 'lose' : 'draw';
    showOverlay({ result: resultKind, playLabel: 'Play again' });
  }
  updateSidebar._prevPhase = state.phase;
}

// Session-stats: how many games this browser session, how many won. Lives in
// localStorage so reloading the page or restarting the server doesn't reset
// your streak. Increments exactly once per game-end (tracks last seen phase
// per session+game so a polled state at phase=1 doesn't bump twice).
// Session win/loss tracking — still persists to localStorage so the
// count survives reloads, but with the sidebar removed there's no DOM
// surface for it. Kept in case we want to surface a 'streak' somewhere
// later. To reset, use the browser console:
//     localStorage.removeItem('mw2_play_live_session_v1'); location.reload();
const SESSION_KEY = 'mw2_play_live_session_v1';
let session = JSON.parse(localStorage.getItem(SESSION_KEY) || '{"games":0,"wins":0,"losses":0,"draws":0,"counted_for":null}');
function persistSession() {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}
function maybeCountGameEnd(state) {
  if (!state || state.phase === 0 || state.phase === undefined) return;
  const gid = `${state.level}::${state.opponent}::${state.tick}::${state.phase}`;
  if (session.counted_for === gid) return;
  session.counted_for = gid;
  session.games += 1;
  if (state.phase === 1) session.wins   += 1;
  if (state.phase === 2) session.losses += 1;
  if (state.phase === 3) session.draws  += 1;
  persistSession();
}

// ----- Click handling -------------------------------------------------------
// Action encoding mirrors sim/actions.py:
//   NOOP_INDEX = 4096
//   send action: idx = type * SLOTS_SQ + src * MAX_BUILDING_SLOTS + tgt
// SLOTS_SQ and MAX_BUILDING_SLOTS are constants pulled from the API on first
// state to stay decoupled from the JS source.
let SLOTS = null, SLOTS_SQ = null, NOOP_IDX = null, ACTION_DIM = null;
let SEND_PERCENTAGES = [25, 50, 75, 100];
let selectedTypeIdx = 3;  // default 100% — most natural human play
async function fetchConsts() {
  const r = await fetch('/api/constants');
  const j = await r.json();
  SLOTS      = j.MAX_BUILDING_SLOTS;
  SLOTS_SQ   = j.SLOTS_SQ;
  NOOP_IDX   = j.NOOP_INDEX;
  ACTION_DIM = j.ACTION_SPACE_SIZE;
  if (Array.isArray(j.SEND_PERCENTAGES) && j.SEND_PERCENTAGES.length > 0) {
    SEND_PERCENTAGES = j.SEND_PERCENTAGES;
    selectedTypeIdx = SEND_PERCENTAGES.length - 1;  // last = 100%
  }
  buildPctButtons();
}
function buildPctButtons() {
  const $row = document.getElementById('pct-row');
  $row.innerHTML = '';
  SEND_PERCENTAGES.forEach((pct, idx) => {
    const btn = document.createElement('button');
    btn.textContent = `${pct}%`;
    btn.dataset.idx = idx;
    if (idx === selectedTypeIdx) btn.classList.add('active');
    btn.addEventListener('click', () => {
      selectedTypeIdx = idx;
      // Refresh active state on all buttons.
      [...$row.children].forEach(b => b.classList.toggle('active',
        Number(b.dataset.idx) === selectedTypeIdx));
    });
    $row.appendChild(btn);
  });
}
function buildSendAction(srcSlot, tgtSlot, typeIdx) {
  // typeIdx indexes SEND_PERCENTAGES (0..len-1).
  return typeIdx * SLOTS_SQ + srcSlot * SLOTS + tgtSlot;
}
// Number keys 1..N pick a percentage (1=25%, 2=50%, 3=75%, 4=100%).
// Space passes the current turn (NOOP) — useful when you want to defend / let
// units accumulate without sending anything.
window.addEventListener('keydown', async (e) => {
  if (e.key >= '1' && e.key <= String(SEND_PERCENTAGES.length)) {
    selectedTypeIdx = Number(e.key) - 1;
    buildPctButtons();
    return;
  }
  if (e.key === ' ' && lastState && lastState.phase === 0) {
    e.preventDefault();
    selectedSrc = null;
    render(lastState);
    try {
      await fetch('/api/action', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({idx: NOOP_IDX}),
      });
    } catch (err) { showToast('net error: ' + err.message); }
  }
});

function buildingAt(canvasX, canvasY) {
  if (!lastState) return null;
  for (const b of lastState.buildings) {
    const [cx, cy] = simToCanvas(b.x, b.y);
    const r = radiusFor(b);
    const dx = canvasX - cx, dy = canvasY - cy;
    if (dx*dx + dy*dy <= (r + 4) * (r + 4)) return b;
  }
  return null;
}

CV.addEventListener('click', async (e) => {
  if (!lastState || lastState.phase !== 0) return;  // only during play
  // CSS pixels — fitCanvas() uses ctx.setTransform(dpr,...) so drawing
  // coordinates are CSS pixels, not device pixels. The old `* CV.width
  // / rect.width` math gave device pixels and broke clicks on Retina
  // displays where rect.width != CV.width.
  const rect = CV.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const b = buildingAt(cx, cy);
  if (!b) { selectedSrc = null; render(lastState); return; }

  if (selectedSrc === null) {
    // First click — pick source. Must be one of YOUR buildings (owner 1).
    if (b.owner !== 1) {
      showToast("Pick one of your (red) buildings as source");
      return;
    }
    selectedSrc = b.slot;
    render(lastState);
    return;
  }

  // Second click — target. Build send action.
  const tgtSlot = b.slot;
  if (tgtSlot === selectedSrc) {
    // Click same building = deselect.
    selectedSrc = null;
    render(lastState);
    return;
  }
  const actionIdx = buildSendAction(selectedSrc, tgtSlot, selectedTypeIdx);
  // CLIENT-SIDE SPAWN — record the fighter in the tracker BEFORE the server
  // confirms it. For very short trips the sim can resolve the whole launch
  // → land in a single tick that we never poll, so the fighter would
  // never appear in state.groups and never be tracked. Spawning client-
  // side guarantees the user always sees the journey. Key uses the same
  // owner-src-tgt scheme as trackFighters() so when the server-side
  // confirmation arrives, trackFighters' `if (fighterTracker.has(k)) continue`
  // de-dupes naturally.
  const srcBuilding = lastState.buildings.find(b => b.slot === selectedSrc);
  if (srcBuilding) {
    const pct = SEND_PERCENTAGES[selectedTypeIdx] / 100;
    const count = Math.floor(srcBuilding.garrison * pct);
    if (count > 0) {
      const dx = b.x - srcBuilding.x, dy = b.y - srcBuilding.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // Match the sim's apparent travel speed; 200 sim-units/sec is roughly
      // calibrated against observed sim arrivals on random_close_4_5.
      const durationSec = Math.max(MIN_ANIM_SEC, dist / 200);
      const k = `1-${selectedSrc}-${tgtSlot}`;
      if (!fighterTracker.has(k)) {
        fighterTracker.set(k, {
          owner: 1, count,
          srcSlot: selectedSrc, tgtSlot,
          srcX: srcBuilding.x, srcY: srcBuilding.y,
          tgtX: b.x, tgtY: b.y,
          startTime: performance.now(),
          durationSec,
          landed: false,
          pendingAmount: count,    // optimistic source decrement until server confirms
          serverConfirmed: false,
        });
      }
    }
  }
  selectedSrc = null;
  try {
    const r = await fetch('/api/action', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({idx: actionIdx}),
    });
    const j = await r.json();
    if (j.error) showToast("⚠ " + j.error);
  } catch (err) {
    showToast("net error: " + err.message);
  }
});

CV.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  selectedSrc = null;
  render(lastState);
});
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { selectedSrc = null; render(lastState); }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  // Don't reset immediately — open the overlay so the user can switch
  // opponents if they want, then Play kicks off the actual reset.
  showOverlay({ result: null, playLabel: 'Play' });
});

// ----- Opponent picker ------------------------------------------------------
// Loads the same champion list the deployed play.html shows, plus a synthetic
// "random_legal" baseline. Selecting an option starts a new game vs that
// opponent — first switch may take a few seconds while the worker downloads
// the weights from Supabase storage.
async function loadChampions() {
  try {
    const r = await fetch('/api/champions');
    const j = await r.json();
    if (j.error) { showToast('couldn\'t load models: ' + j.error); return; }
    const champs = j.champions || [];
    const opts = champs.map(c =>
      `<option value="${c.run_id}">${c.label.replace(/</g, '&lt;')}</option>`
    ).join('');
    $opp.innerHTML = opts;
    // Sync to whatever the server currently has loaded.
    const target = lastState?.champion_run_id ?? 'random_legal';
    if ([...$opp.options].some(o => o.value === target)) $opp.value = target;
  } catch (err) {
    showToast('net error loading models: ' + err.message);
  }
}
// Play button on the start/game-over overlay — applies the picked opponent
// (if changed) then resets the sim and hides the overlay.
$playBtn.addEventListener('click', async () => {
  $playBtn.disabled = true;
  $playBtn.textContent = 'Loading…';
  const champ = $opp.value || null;
  const level = $level.value || null;
  try {
    const body = {};
    if (champ) body.champion = champ;
    if (level) body.level_name = level;
    const r = await fetch('/api/reset', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (j.error) {
      showToast('⚠ ' + j.error);
      $playBtn.textContent = 'Try again';
      $playBtn.disabled = false;
      return;
    }
    selectedSrc = null;
    // Reset rising-edge state so the next game's end can re-open the
    // overlay. Also clear lastState so a stale render doesn't show old
    // ownership/balance for one frame. Reset the play-area bbox so the
    // camera re-fits to the new game's building layout.
    updateSidebar._prevPhase = 0;
    lastState = null;
    resetBbox();
    resetFighterTracker();
    hideOverlay();
  } catch (err) {
    showToast('net error: ' + err.message);
    $playBtn.textContent = 'Try again';
    $playBtn.disabled = false;
  }
});

// ----- Polling + animation loop ---------------------------------------------
// Polling pulls state; rendering runs on requestAnimationFrame so fighter
// motion stays smooth between polls (state lives at 2 ticks/sec wall clock;
// rAF runs at the display refresh rate, typically 60-120 Hz).
async function poll() {
  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    lastState = j;
    lastStateTime = performance.now();
    trackFighters(j);   // first-seen bookkeeping for smooth client-side fighter animation
    updateSidebar(j);
    maybeCountGameEnd(j);
  } catch (err) { /* server might be restarting */ }
}

function rafLoop() {
  // Reaping fighters on the rAF tick (not on poll) ensures a fighter that
  // lands BETWEEN polls still gets its arrival flash on the very next
  // frame, not delayed until the next state poll.
  reapFinishedFighters();
  if (lastState) render(lastState);
  requestAnimationFrame(rafLoop);
}

(async () => {
  // Show start overlay immediately with a loading state — Play button is
  // disabled until assets + champion list are loaded.
  showOverlay({ result: null, playLabel: 'Play' });
  $playBtn.disabled = true;
  $playBtn.innerHTML = '<span class="loading"><span class="spinner"></span>Loading…</span>';

  // Load assets + API setup in parallel.
  await Promise.all([
    loadAssets(),
    fetchConsts(),
    loadChampions(),
  ]);

  $playBtn.innerHTML = '';
  $playBtn.textContent = 'Play';
  $playBtn.disabled = false;
  pollHandle = setInterval(poll, 100);   // state poll every 100ms; rAF renders every frame in between for smooth motion
  poll();
  requestAnimationFrame(rafLoop);
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler.
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    session: _Session  # set by class on construction

    def log_message(self, fmt, *args):
        # Quieter than the default — only log errors.
        if not args or "200" in str(args[0]):
            return
        super().log_message(fmt, *args)

    # CORS: needed when the GitHub-Pages-hosted dashboard fetches /api/* on
    # http://localhost:8765 from an https:// origin. Browsers allow the
    # HTTPS→localhost loopback exception, but the CORS handshake still has
    # to succeed.
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        # Chrome's Private Network Access — HTTPS pages need this to fetch
        # http://localhost without being blocked by PNA preflight rules.
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        # Preflight for cross-origin POST /api/action and /api/reset.
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._html(_HTML)
            return
        if self.path.startswith("/assets/"):
            name = self.path[len("/assets/"):]
            # Whitelist: only [a-zA-Z0-9._-] allowed, prevents directory escape.
            if not all(ch.isalnum() or ch in "._-" for ch in name) or "/" in name or ".." in name:
                self.send_error(400, "bad asset name")
                return
            repo = Path(__file__).resolve().parent.parent
            path = repo / "dashboard" / "lib" / "assets" / name
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self._cors()
            ctype = "image/png" if name.endswith(".png") else "application/octet-stream"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/state":
            with self.session.lock:
                self._json(self.session.state_json())
            return
        if self.path == "/api/champions":
            try:
                champs = _list_champions()
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc)}, status=500)
                return
            self._json({"champions": champs})
            return
        if self.path == "/api/constants":
            from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX, SLOTS_SQ
            from sim import config as C
            self._json({
                "MAX_BUILDING_SLOTS": int(C.MAX_BUILDING_SLOTS),
                "SLOTS_SQ":           int(SLOTS_SQ),
                "NOOP_INDEX":         int(NOOP_INDEX),
                "ACTION_SPACE_SIZE":  int(ACTION_SPACE_SIZE),
                # SEND_PERCENTAGES = (25, 50, 75, 100) — type 0..3 indexes the
                # tuple. UI surfaces this so the user can pick how much to send.
                "SEND_PERCENTAGES":   list(C.SEND_PERCENTAGES),
            })
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/action":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                idx = int(body.get("idx", 0))
            except Exception as exc:
                self._json({"error": f"bad request: {exc}"}, status=400)
                return
            # No lock here — enqueue_action manages its own under the hood.
            # Returning immediately lets the click feel snappy; the actual
            # sim step happens on the next tick (≤200 ms).
            self._json(self.session.enqueue_action(idx))
            return
        if self.path == "/api/reset":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length > 0 else {}
                # `champion` / `level_name` keys absent → preserve current;
                # explicit string → swap. Both passed through to session.reset.
                champion   = body.get("champion")
                level_name = body.get("level_name")
                with self.session.lock:
                    self.session.reset(champion_run_id=champion, level_name=level_name)
                    label = self.session.opponent_label
                self._json({"ok": True, "opponent": label, "level": self.session.level_name})
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc)}, status=500)
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="random_close_4_5")
    ap.add_argument("--champion", default=None,
                    help="Champion run_id prefix (Supabase) or local weights.pt path. "
                         "If omitted or 'random_legal', plays vs random legal.")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    session = _Session(args.level, args.champion)
    print(f"[play_live] initializing session: level={args.level} champion={args.champion}")
    session.reset()
    print(f"[play_live] opponent: {session.opponent_label}")

    # Bind the session to the handler class.
    handler_cls = type("_BoundHandler", (_Handler,), {"session": session})

    addr = ("127.0.0.1", args.port)
    httpd = ThreadingHTTPServer(addr, handler_cls)
    print(f"[play_live] serving on http://{addr[0]}:{addr[1]}  —  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[play_live] shutting down")
        httpd.server_close()


if __name__ == "__main__":
    main()
