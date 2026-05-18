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
    # based. 5 Hz feels active without spamming; the trained net was trained
    # against an env stepping at this same decision-interval cadence.
    TICK_HZ = 5

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
        self._ticker_running = False

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
        self._ticker_running = True
        t = threading.Thread(target=self._tick_loop, daemon=True, name="mw-ticker")
        self._ticker_thread = t
        t.start()

    def _stop_ticker(self) -> None:
        self._ticker_running = False
        t = self._ticker_thread
        if t and t.is_alive():
            # Brief wait — the ticker's max iteration time is ~1/TICK_HZ.
            t.join(timeout=2.0 / self.TICK_HZ)
        self._ticker_thread = None

    def _tick_loop(self) -> None:
        """Background loop: advance the sim once per TICK_HZ regardless of
        whether the user has clicked. Each tick consumes one queued user
        action (or noop) and lets the agent take its turn via env.step()."""
        from sim.actions import NOOP_INDEX
        from sim import config as C

        dt = 1.0 / self.TICK_HZ
        next_t = time.time()
        while self._ticker_running:
            now = time.time()
            wait = next_t - now
            if wait > 0:
                time.sleep(wait)
            next_t += dt

            with self.lock:
                if self.env is None or self.env.state.phase != C.PHASE_PLAYING:
                    # Game ended (or env was torn down) — stop ticking. A
                    # /api/reset will restart us with a fresh game.
                    self._ticker_running = False
                    break
                a_idx = self._action_queue.popleft() if self._action_queue else NOOP_INDEX
                try:
                    self.env.step(int(a_idx))
                except Exception:
                    traceback.print_exc()
                    self._ticker_running = False
                    break

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
  body { margin: 0; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif;
         background: #0a0c10; color: #e6e8ef; }
  header { padding: 8px 16px; display: flex; gap: 16px; align-items: center;
           background: #11141b; border-bottom: 1px solid #1f2330; }
  header h1 { margin: 0; font-size: 14px; color: #93a3b8; }
  header .opp { color: #93a3b8; }
  header .opp .mono { color: #e6e8ef; }
  header .actions { margin-left: auto; display: flex; gap: 8px; }
  button { background: #1f2330; color: #e6e8ef; border: 1px solid #2c3242;
           padding: 4px 10px; border-radius: 4px; cursor: pointer; font: inherit; }
  button:hover { background: #2a3042; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .layout { display: grid; grid-template-columns: 1fr 280px; gap: 0; }
  .canvas-wrap { position: relative; padding: 20px; }
  canvas { background: #050610; border-radius: 6px; cursor: crosshair; display: block;
           width: 100%; max-width: 720px; aspect-ratio: 1;
           box-shadow: 0 0 40px rgba(96, 165, 250, 0.06) inset; }
  /* Glassmorphism sidebar — translucent dark with backdrop blur. */
  aside { padding: 16px; background: rgba(14, 17, 23, 0.72); border-left: 1px solid rgba(96, 165, 250, 0.12);
          min-height: 100vh; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); }
  /* Emblem corners on the canvas. */
  .emblem-corner { position: absolute; width: 56px; height: 56px; opacity: 0.75;
                   pointer-events: none; filter: drop-shadow(0 0 8px rgba(0,0,0,0.6)); }
  .emblem-corner.tl { top: 32px;    left: 32px; }
  .emblem-corner.tr { top: 32px;    right: 32px; }
  aside h3 { margin: 0 0 8px; font-size: 12px; color: #93a3b8; text-transform: uppercase;
             letter-spacing: 0.05em; }
  .stat { display: flex; justify-content: space-between; padding: 4px 0;
          border-bottom: 1px solid #1f2330; }
  .stat .k { color: #93a3b8; }
  .stat .v { font-family: ui-monospace, monospace; }
  .legend { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
  .help { color: #93a3b8; font-size: 12px; margin-top: 12px; line-height: 1.5; }
  .help kbd { background: #1f2330; border: 1px solid #2c3242; border-radius: 3px;
              padding: 1px 5px; font-family: ui-monospace, monospace; font-size: 11px; }
  .pct-row { display: flex; gap: 6px; margin-top: 8px; }
  .pct-row button { flex: 1; padding: 8px 4px; font-size: 13px; font-weight: 500; }
  .pct-row button.active { background: #f59e0b; border-color: #f59e0b; color: #11141b; }
  .toast { position: absolute; top: 24px; left: 50%; transform: translateX(-50%);
           background: #1f2330; padding: 6px 14px; border-radius: 4px;
           border: 1px solid #2c3242; opacity: 0; transition: opacity 0.2s;
           pointer-events: none; }
  .toast.show { opacity: 1; }
  .phase-banner { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
                  font-size: 36px; font-weight: 600; padding: 16px 32px;
                  background: rgba(17,20,27,0.92); border: 2px solid #2c3242;
                  border-radius: 8px; pointer-events: none; }
  .phase-banner.win { color: #34d399; border-color: #34d399; }
  .phase-banner.lose { color: #f87171; border-color: #f87171; }
  .phase-banner.draw { color: #93a3b8; }
  select { background: #1f2330; color: #e6e8ef; border: 1px solid #2c3242;
           padding: 4px 8px; border-radius: 4px; font: inherit; }
</style>
</head>
<body>
<header>
  <h1>🍄 mushroom wars — live</h1>
  <span class="opp">vs
    <select id="opp-picker" class="mono" style="margin-left:4px; max-width:360px;">
      <option value="">loading...</option>
    </select>
  </span>
  <span class="opp">on <span class="mono" id="level-label">—</span></span>
  <div class="actions">
    <button id="btn-reset">New game</button>
  </div>
</header>
<div class="layout">
  <div class="canvas-wrap">
    <canvas id="board" width="720" height="720"></canvas>
    <img class="emblem-corner tl" id="emblem-tl" src="/assets/emblem-human.png" alt="">
    <img class="emblem-corner tr" id="emblem-tr" src="/assets/emblem-alien.png" alt="">
    <div id="toast" class="toast"></div>
    <div id="phase-banner" class="phase-banner" style="display:none"></div>
  </div>
  <aside>
    <h3>Send size</h3>
    <div class="pct-row" id="pct-row">
      <!-- buttons inserted at runtime from SEND_PERCENTAGES (25/50/75/100) -->
    </div>
    <div class="muted small" style="margin-top:6px;">How much of the source garrison to send. Press <kbd>space</kbd> to pass this turn.</div>

    <h3 style="margin-top:18px">Session</h3>
    <div class="stat"><span class="k">games</span><span class="v" id="s-sess-games">0</span></div>
    <div class="stat"><span class="k">wins</span><span class="v" id="s-sess-wins" style="color:#34d399">0</span></div>
    <div class="stat"><span class="k">losses</span><span class="v" id="s-sess-losses" style="color:#f87171">0</span></div>
    <div class="stat"><span class="k">draws</span><span class="v" id="s-sess-draws">0</span></div>
    <div class="stat"><span class="k">win rate</span><span class="v" id="s-sess-rate">—</span></div>

    <h3 style="margin-top:18px">State</h3>
    <div class="legend">
      <span><span class="swatch" style="background:#f87171"></span>You (P1)</span>
      <span><span class="swatch" style="background:#60a5fa"></span>Agent (P2)</span>
      <span><span class="swatch" style="background:#6b7280"></span>Neutral</span>
    </div>
    <div class="stat"><span class="k">tick</span><span class="v" id="s-tick">—</span></div>
    <div class="stat"><span class="k">phase</span><span class="v" id="s-phase">—</span></div>
    <div class="stat"><span class="k">your buildings</span><span class="v" id="s-mine">—</span></div>
    <div class="stat"><span class="k">enemy buildings</span><span class="v" id="s-enemy">—</span></div>
    <div class="stat"><span class="k">your units</span><span class="v" id="s-mine-units">—</span></div>
    <div class="stat"><span class="k">enemy units</span><span class="v" id="s-enemy-units">—</span></div>
    <div class="stat"><span class="k">in-flight (yours)</span><span class="v" id="s-mine-flight">—</span></div>
    <div class="stat"><span class="k">in-flight (theirs)</span><span class="v" id="s-enemy-flight">—</span></div>

    <h3 style="margin-top:24px">How to play</h3>
    <div class="help">
      <p><strong>1.</strong> Pick a send size above (or press <kbd>1</kbd>–<kbd>4</kbd>).</p>
      <p><strong>2.</strong> Click one of your (red) buildings to select a source.</p>
      <p><strong>3.</strong> Click any building to send that fraction of the garrison there.</p>
      <p><strong>Esc</strong> or <kbd>right-click</kbd> to deselect.</p>
      <p><strong>Real-time:</strong> the agent acts on its own clock — don't dawdle.
         Garrisons regenerate automatically; the game runs until one side has zero
         buildings or you both run out of units.</p>
    </div>
  </aside>
</div>

<script>
// Canvas-only state. The server is the source of truth; we just paint.
const CV  = document.getElementById('board');
const CTX = CV.getContext('2d');
const $opp = document.getElementById('opp-picker');
const $lvl = document.getElementById('level-label');
const $toast = document.getElementById('toast');
const $banner = document.getElementById('phase-banner');

// Selected source building slot (null = nothing picked).
let selectedSrc = null;
let lastState = null;
let pollHandle = null;

// Sim coordinate → canvas pixel. Maps the ~700×700 sim space to the canvas
// minus a margin so building circles aren't clipped at the edges.
const CANVAS_PAD = 40;
function simToCanvas(x, y) {
  const w = CV.clientWidth;
  const h = CV.clientHeight;
  // sim coords run 0..700 (full) or 0..350 (close); auto-fit by inferring max
  // observed extent from current state.
  const ext = (lastState?.buildings ?? []).reduce(
    (m, b) => Math.max(m, b.x, b.y), 700);
  const sx = CANVAS_PAD + (x / ext) * (w - 2*CANVAS_PAD);
  const sy = CANVAS_PAD + (y / ext) * (h - 2*CANVAS_PAD);
  return [sx, sy];
}

function ownerColor(owner) {
  return ['#6b7280', '#f87171', '#60a5fa'][owner] || '#6b7280';
}

function radiusFor(building) {
  // Bigger capacity → larger sprite. Cap at 38 px so big planets don't blow up.
  return Math.min(38, 16 + building.capacity * 0.3);
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

// ── Capture-FX state ────────────────────────────────────────────────────
// When a building's owner flips between renders, spawn a short-lived burst
// that gets composited on top of that building for ~600 ms. Each fx entry:
// {x, y, r, startedAt}.
const captureFx = [];
let lastOwners = {};   // slot → owner, used to detect flips

function spawnCaptureFx(state) {
  if (!state || !state.buildings) return;
  for (const b of state.buildings) {
    const prev = lastOwners[b.slot];
    if (prev !== undefined && prev !== b.owner) {
      captureFx.push({ x: b.x, y: b.y, r: radiusFor(b), startedAt: performance.now() });
    }
    lastOwners[b.slot] = b.owner;
  }
}
function drawCaptureFx() {
  const now = performance.now();
  const lifeMs = 600;
  for (let i = captureFx.length - 1; i >= 0; i--) {
    const fx = captureFx[i];
    const t = (now - fx.startedAt) / lifeMs;
    if (t > 1) { captureFx.splice(i, 1); continue; }
    const [cx, cy] = simToCanvas(fx.x, fx.y);
    // Burst grows + fades over the lifetime.
    const r = fx.r * (1.2 + 1.6 * t);
    CTX.save();
    CTX.globalAlpha = 1 - t;
    if (ASSETS['capture-fx']) {
      CTX.drawImage(ASSETS['capture-fx'], cx - r, cy - r, r * 2, r * 2);
    } else {
      CTX.beginPath();
      CTX.arc(cx, cy, r, 0, Math.PI * 2);
      CTX.strokeStyle = '#fde047';
      CTX.lineWidth = 3;
      CTX.stroke();
    }
    CTX.restore();
  }
}

function showToast(msg) {
  $toast.textContent = msg;
  $toast.classList.add('show');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => $toast.classList.remove('show'), 1500);
}

function showBanner(text, kind) {
  $banner.textContent = text;
  $banner.className = `phase-banner ${kind}`;
  $banner.style.display = 'block';
}
function hideBanner() { $banner.style.display = 'none'; }

// Map owner code → planet sprite name. 0=neutral, 1=P1 (human), 2=P2 (alien).
const PLANET_BY_OWNER = ['planet-neutral', 'planet-human', 'planet-alien'];
const FIGHTER_BY_OWNER = ['fighter-human', 'fighter-human', 'fighter-alien'];  // owner 0 never moves

function render(state) {
  // Background — full-canvas sprite, falls back to solid black.
  if (ASSETS['background']) {
    CTX.drawImage(ASSETS['background'], 0, 0, CV.width, CV.height);
  } else {
    CTX.fillStyle = '#050610';
    CTX.fillRect(0, 0, CV.width, CV.height);
  }
  if (!state || !state.ready) return;

  // Buildings — planet sprites by owner, with selected-source highlight ring.
  for (const b of state.buildings) {
    const [cx, cy] = simToCanvas(b.x, b.y);
    const r = radiusFor(b);
    const spriteName = PLANET_BY_OWNER[b.owner] || 'planet-neutral';
    const img = ASSETS[spriteName];
    if (img) {
      CTX.drawImage(img, cx - r, cy - r, r * 2, r * 2);
    } else {
      // Fallback: colored circle until sprite loads.
      CTX.beginPath();
      CTX.arc(cx, cy, r, 0, Math.PI * 2);
      CTX.fillStyle = ownerColor(b.owner);
      CTX.globalAlpha = 0.85;
      CTX.fill();
      CTX.globalAlpha = 1;
    }
    // Owner ring — thin colored stroke so red/blue/gray reads at a glance
    // even with the realistic planet textures.
    CTX.beginPath();
    CTX.arc(cx, cy, r + 1, 0, Math.PI * 2);
    CTX.strokeStyle = ownerColor(b.owner);
    CTX.lineWidth = 2.5;
    CTX.globalAlpha = 0.7;
    CTX.stroke();
    CTX.globalAlpha = 1;
    // Selected-source highlight.
    if (b.slot === selectedSrc) {
      CTX.beginPath();
      CTX.arc(cx, cy, r + 5, 0, Math.PI * 2);
      CTX.strokeStyle = '#facc15';
      CTX.lineWidth = 3;
      CTX.stroke();
    }
    // Garrison number — bold + glow so it pops against the planet.
    CTX.font = 'bold 14px ui-monospace, monospace';
    CTX.textAlign = 'center';
    CTX.textBaseline = 'middle';
    CTX.fillStyle = 'rgba(0,0,0,0.85)';
    CTX.fillText(String(b.garrison), cx + 1, cy + 1);
    CTX.fillStyle = '#fff';
    CTX.fillText(String(b.garrison), cx, cy);
  }

  // Groups (in-flight squads) — ship sprite rotated along travel direction.
  const buildingByIdx = {};
  for (const b of state.buildings) buildingByIdx[b.slot] = b;
  for (const g of state.groups) {
    const src = buildingByIdx[g.src];
    const tgt = buildingByIdx[g.tgt];
    if (!src || !tgt) continue;
    const frac = g.travel > 0 ? Math.min(1, g.progress / g.travel) : 0;
    const x = src.x + (tgt.x - src.x) * frac;
    const y = src.y + (tgt.y - src.y) * frac;
    const [cx, cy] = simToCanvas(x, y);
    const [tcx, tcy] = simToCanvas(tgt.x, tgt.y);
    const angle = Math.atan2(tcy - cy, tcx - cx);  // 0 rad = +x (right)
    const spriteName = FIGHTER_BY_OWNER[g.owner];
    const img = ASSETS[spriteName];
    const size = 18;
    if (img) {
      CTX.save();
      CTX.translate(cx, cy);
      // Sprite is drawn nose-up (-y); rotate so nose points along travel.
      CTX.rotate(angle + Math.PI / 2);
      CTX.drawImage(img, -size / 2, -size / 2, size, size);
      CTX.restore();
    } else {
      CTX.beginPath();
      CTX.arc(cx, cy, 6, 0, Math.PI * 2);
      CTX.fillStyle = ownerColor(g.owner);
      CTX.fill();
    }
    // Unit count just behind the ship.
    CTX.font = 'bold 11px ui-monospace, monospace';
    CTX.textAlign = 'center';
    CTX.textBaseline = 'middle';
    CTX.fillStyle = 'rgba(0,0,0,0.85)';
    CTX.fillText(String(g.count), cx + 1, cy + size / 2 + 9);
    CTX.fillStyle = ownerColor(g.owner);
    CTX.fillText(String(g.count), cx, cy + size / 2 + 8);
  }

  // Capture FX last — overlays sit on top of buildings.
  drawCaptureFx();
}

function updateSidebar(state) {
  if (!state || !state.ready) return;
  document.getElementById('s-tick').textContent  = state.tick;
  const phaseStr = ['playing', 'P1 wins', 'P2 wins', 'draw'][state.phase] || `phase ${state.phase}`;
  document.getElementById('s-phase').textContent = phaseStr;
  let mine = 0, enemy = 0, mineU = 0, enemyU = 0;
  for (const b of state.buildings) {
    if (b.owner === 1) { mine++;  mineU  += b.garrison; }
    if (b.owner === 2) { enemy++; enemyU += b.garrison; }
  }
  let mineF = 0, enemyF = 0;
  for (const g of state.groups) {
    if (g.owner === 1) mineF  += g.count;
    if (g.owner === 2) enemyF += g.count;
  }
  document.getElementById('s-mine').textContent          = mine;
  document.getElementById('s-enemy').textContent         = enemy;
  document.getElementById('s-mine-units').textContent    = mineU;
  document.getElementById('s-enemy-units').textContent   = enemyU;
  document.getElementById('s-mine-flight').textContent   = mineF;
  document.getElementById('s-enemy-flight').textContent  = enemyF;
  // Sync dropdown to backend without clobbering the user's mid-pick state:
  // only update if the user isn't actively focused on the select.
  const targetVal = state.champion_run_id ?? 'random_legal';
  if (document.activeElement !== $opp && $opp.value !== targetVal) {
    // Only set if the option actually exists (champions list might still be loading).
    if ([...$opp.options].some(o => o.value === targetVal)) {
      $opp.value = targetVal;
    }
  }
  $lvl.textContent = state.level ?? '—';

  if (state.phase === 1)      showBanner("YOU WIN",   'win');
  else if (state.phase === 2) showBanner("AGENT WINS",'lose');
  else if (state.phase === 3) showBanner("DRAW",      'draw');
  else                        hideBanner();
}

// Session-stats: how many games this browser session, how many won. Lives in
// localStorage so reloading the page or restarting the server doesn't reset
// your streak. Increments exactly once per game-end (tracks last seen phase
// per session+game so a polled state at phase=1 doesn't bump twice).
const SESSION_KEY = 'mw2_play_live_session_v1';
let session = JSON.parse(localStorage.getItem(SESSION_KEY) || '{"games":0,"wins":0,"losses":0,"draws":0,"counted_for":null}');
function persistSession() {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  document.getElementById('s-sess-games').textContent  = session.games;
  document.getElementById('s-sess-wins').textContent   = session.wins;
  document.getElementById('s-sess-losses').textContent = session.losses;
  document.getElementById('s-sess-draws').textContent  = session.draws;
  // Decimal precision: 99.99% mastery target needs 2 decimals to read.
  const decided = session.wins + session.losses + session.draws;
  if (decided > 0) {
    const rate = (session.wins + 0.5 * session.draws) / decided;
    document.getElementById('s-sess-rate').textContent = `${(rate * 100).toFixed(2)}%`;
  } else {
    document.getElementById('s-sess-rate').textContent = '—';
  }
}
function maybeCountGameEnd(state) {
  if (!state || state.phase === 0 || state.phase === undefined) return;
  // game_id = (level + opponent + first-seen-tick of this terminal). Identifies
  // a single game-end so polling repeatedly doesn't double-count it. Reset
  // clears it because session.counted_for is set per-game.
  const gid = `${state.level}::${state.opponent}::${state.tick}::${state.phase}`;
  if (session.counted_for === gid) return;
  session.counted_for = gid;
  session.games += 1;
  if (state.phase === 1) session.wins   += 1;
  if (state.phase === 2) session.losses += 1;
  if (state.phase === 3) session.draws  += 1;
  persistSession();
}
document.getElementById('s-sess-games').addEventListener('dblclick', () => {
  // Hidden reset: double-click "games" stat to wipe session counters.
  if (confirm('Reset session counters?')) {
    session = {games:0, wins:0, losses:0, draws:0, counted_for:null};
    persistSession();
  }
});
persistSession();  // initial render from localStorage

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
  const rect = CV.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (CV.width / rect.width);
  const cy = (e.clientY - rect.top)  * (CV.height / rect.height);
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
  selectedSrc = null;
  // Fire the action; render will refresh on next poll.
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

document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  selectedSrc = null;
  hideBanner();
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
$opp.addEventListener('change', async () => {
  const champ = $opp.value;
  if (!champ) return;
  const shortLabel = $opp.options[$opp.selectedIndex]?.text ?? champ;
  showToast('Loading ' + shortLabel + '...');
  $opp.disabled = true;
  try {
    const r = await fetch('/api/reset', {
      method:  'POST',
      headers: {'content-type': 'application/json'},
      body:    JSON.stringify({champion: champ}),
    });
    const j = await r.json();
    if (j.error) { showToast('⚠ ' + j.error); return; }
    selectedSrc = null;
    hideBanner();
    showToast('Now playing vs ' + (j.opponent || shortLabel));
  } catch (err) {
    showToast('net error: ' + err.message);
  } finally {
    $opp.disabled = false;
  }
});

// ----- Polling --------------------------------------------------------------
async function poll() {
  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    lastState = j;
    spawnCaptureFx(j);   // before render, so the burst paints on this frame
    render(j);
    updateSidebar(j);
    maybeCountGameEnd(j);
  } catch (err) { /* server might be restarting */ }
}

(async () => {
  // Kick off asset loading in parallel with the API setup — render falls
  // back to colored circles until the sprites land, so don't block.
  loadAssets();
  await fetchConsts();
  await loadChampions();
  pollHandle = setInterval(poll, 100);  // 100ms = 10 Hz — server ticks at 5 Hz, this gives ~2× margin
  poll();
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
                # `champion` key absent → preserve current; explicit string → swap.
                champion = body.get("champion")
                with self.session.lock:
                    self.session.reset(champion_run_id=champion)
                    label = self.session.opponent_label
                self._json({"ok": True, "opponent": label})
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
