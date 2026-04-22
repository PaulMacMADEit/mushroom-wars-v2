/**
 * Replay playback for Mushroom Wars v2.
 *
 * The event log is produced by `sim/envs/replay.py`. No sim logic runs here —
 * buildings tick up with a constant production rate between events, squads
 * linearly interpolate between src and dst buildings from their `send` event
 * to the matching `arrive` event. All numbers in the log are sim-authoritative,
 * so the canvas only has to paint.
 */

const OWNER_COLOR = {
  0: "#6b7280",   // neutral
  1: "#f87171",   // P1 red
  2: "#60a5fa",   // P2 blue
};
const OWNER_GLOW = {
  0: "rgba(107,114,128,0.25)",
  1: "rgba(248,113,113,0.35)",
  2: "rgba(96,165,250,0.35)",
};
const OWNER_NAME = { 0: "Neutral", 1: "P1", 2: "P2" };

/** Build a per-building timeline of `{t, owner, garrison}` change-points. */
function buildBuildingTimelines(data) {
  const timelines = {};
  for (const b of data.map.buildings) {
    timelines[b.slot] = [{ t: 0, owner: b.init.owner, garrison: b.init.garrison }];
  }
  for (const e of data.events) {
    if (e.kind === "send") {
      const tl = timelines[e.src];
      if (tl) tl.push({ t: e.t, owner: tl[tl.length - 1].owner, garrison: e.src_garrison_after });
    } else if (e.kind === "arrive") {
      const tl = timelines[e.dst];
      if (tl) tl.push({ t: e.t, owner: e.dst_owner_after, garrison: e.dst_garrison_after });
    }
  }
  return timelines;
}

/** Find the latest change-point at or before t (binary search). */
function findChangePoint(timeline, t) {
  let lo = 0, hi = timeline.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (timeline[mid].t <= t) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return timeline[ans];
}

/** Pair sends with their matching arrivals by group id. */
function buildSquadFlights(data) {
  const sendByGroup = {};
  for (const e of data.events) if (e.kind === "send") sendByGroup[e.group] = e;
  const flights = [];
  for (const e of data.events) {
    if (e.kind === "arrive") {
      const s = sendByGroup[e.group];
      if (s) flights.push({ group: e.group, send: s, arrive: e });
    }
  }
  // Sort by send.t for quick linear scans during playback.
  flights.sort((a, b) => a.send.t - b.send.t);
  return flights;
}

export class ReplayPlayer {
  constructor(data, canvas) {
    this.data = data;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    this.timelines = buildBuildingTimelines(data);
    this.flights   = buildSquadFlights(data);
    this.buildingById = Object.fromEntries(data.map.buildings.map(b => [b.slot, b]));
    this.prodPerTick = data.prod_per_tick ?? 10;
    this.scale       = data.scale ?? 10;
    this.duration    = data.duration_ticks;
    this.endEvent    = data.events.find(e => e.kind === "end") ?? null;

    this.t = 0;
    this.speed = 1;
    this.playing = false;

    this._lastFrame = null;
    this._loop = this._loop.bind(this);
    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    this._resize();
    this.render();
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width  = Math.floor(rect.width  * dpr);
    this.canvas.height = Math.floor(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._cssW = rect.width;
    this._cssH = rect.height;
    this.render();
  }

  // --- Public controls ----------------------------------------------------

  play() {
    if (this.playing) return;
    if (this.t >= this.duration) this.t = 0;
    this.playing = true;
    this._lastFrame = performance.now();
    requestAnimationFrame(this._loop);
    this._emit("stateChange");
  }

  pause() {
    this.playing = false;
    this._lastFrame = null;
    this._emit("stateChange");
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  seek(t) {
    this.t = Math.max(0, Math.min(this.duration, t));
    this.render();
    this._emit("tick");
  }

  setSpeed(s) { this.speed = s; }

  onTick(cb)        { this._onTick = cb; }
  onStateChange(cb) { this._onStateChange = cb; }
  _emit(name) {
    if (name === "tick" && this._onTick) this._onTick(this);
    if (name === "stateChange" && this._onStateChange) this._onStateChange(this);
  }

  // --- Main loop ----------------------------------------------------------

  _loop(now) {
    if (!this.playing) return;
    const dt = (now - this._lastFrame) / 1000;   // seconds
    this._lastFrame = now;
    this.t += dt * this.speed;                    // 1 tick = 1 second
    if (this.t >= this.duration) {
      this.t = this.duration;
      this.playing = false;
      this._emit("stateChange");
    }
    this.render();
    this._emit("tick");
    if (this.playing) requestAnimationFrame(this._loop);
  }

  // --- State derivation ---------------------------------------------------

  _buildingStateAt(slot, t) {
    const cp = findChangePoint(this.timelines[slot], t);
    const b = this.buildingById[slot];
    let garrison = cp.garrison;
    if (cp.owner === 1 || cp.owner === 2) {
      const elapsed = Math.max(0, t - cp.t);
      garrison = Math.min(b.capacity, garrison + this.prodPerTick * elapsed);
    }
    return { owner: cp.owner, garrison };
  }

  _activeFlightsAt(t) {
    const out = [];
    for (const f of this.flights) {
      if (f.send.t > t) break;
      if (t < f.send.t || t > f.arrive.t) continue;
      const span = Math.max(1e-6, f.arrive.t - f.send.t);
      const p = (t - f.send.t) / span;
      out.push({ flight: f, p });
    }
    return out;
  }

  // --- Rendering ----------------------------------------------------------

  _worldToScreen(x, y) {
    const pad = 40;
    const w = this._cssW, h = this._cssH;
    const size = Math.min(w, h) - pad * 2;
    const originX = (w - size) / 2;
    const originY = (h - size) / 2;
    return [
      originX + (x / this.data.map.width)  * size,
      originY + (y / this.data.map.height) * size,
    ];
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this._cssW, this._cssH);

    // Subtle playfield backdrop.
    const pad = 40;
    const size = Math.min(this._cssW, this._cssH) - pad * 2;
    const ox = (this._cssW - size) / 2;
    const oy = (this._cssH - size) / 2;
    ctx.save();
    ctx.strokeStyle = "#262a36";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, size, size);
    ctx.restore();

    const t = this.t;

    // Squads first so they pass under building labels.
    for (const { flight, p } of this._activeFlightsAt(t)) {
      const src = this.buildingById[flight.send.src];
      const dst = this.buildingById[flight.send.dst];
      if (!src || !dst) continue;
      const x = src.x + (dst.x - src.x) * p;
      const y = src.y + (dst.y - src.y) * p;
      const [sx, sy] = this._worldToScreen(x, y);
      const count = flight.send.count / this.scale;
      const color = OWNER_COLOR[flight.send.owner];
      const r = 6 + Math.min(10, Math.sqrt(count));

      // Glow halo.
      const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, r * 2.2);
      grad.addColorStop(0, OWNER_GLOW[flight.send.owner]);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(sx, sy, r * 2.2, 0, Math.PI * 2); ctx.fill();

      // Core dot.
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fill();

      // Count label.
      ctx.fillStyle = "#fff";
      ctx.font = "600 11px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(Math.round(count), sx, sy);
    }

    // Buildings on top.
    for (const b of this.data.map.buildings) {
      const { owner, garrison } = this._buildingStateAt(b.slot, t);
      const [sx, sy] = this._worldToScreen(b.x, b.y);
      const real = garrison / this.scale;
      const cap  = b.capacity / this.scale;
      const r = 18 + (real / cap) * 10;     // grow a bit with fullness

      // Capacity ring.
      ctx.strokeStyle = "#2a2f3d";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 28, 0, Math.PI * 2); ctx.stroke();

      // Fill disc.
      const color = OWNER_COLOR[owner];
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fill();

      // Outline.
      ctx.strokeStyle = "rgba(0,0,0,0.4)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Garrison label.
      ctx.fillStyle = "#fff";
      ctx.font = "700 14px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(Math.floor(real), sx, sy);
    }
  }

  destroy() {
    window.removeEventListener("resize", this._resize);
    this.playing = false;
  }
}

export { OWNER_NAME };
