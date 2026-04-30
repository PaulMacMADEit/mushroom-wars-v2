/**
 * JS port of training/encoder.py — recomputes the 41 named features the
 * model actually sees, from replay state at any tick. Pure derivation: no
 * sim logic. Matches encoder.py block-by-block (same names, same order,
 * same normalizers); see encoder.py for the canonical definitions.
 *
 *   GLOBALS  (10)         — tick, building/unit shares, margins
 *   BUILDING (22 × 32)    — per-slot ownership, garrison, threat aggregates
 *   GROUP    (9  × 32)    — per-flight ownership, progress, endpoints
 *
 * Output of computeFeatures() is a plain dict keyed by feature name. The
 * panel in game.html renders rows from these names directly.
 */

export const GLOBAL_NAMES = [
  "tick",
  "time_remaining",
  "p1_buildings_share",
  "p2_buildings_share",
  "neutral_buildings_share",
  "p1_total_units",
  "p2_total_units",
  "p1_share_of_units",
  "building_margin",
  "unit_margin",
];

export const BUILDING_NAMES = [
  "alive",
  "is_p1",
  "is_p2",
  "is_neutral",
  "garrison_norm",
  "garrison_ratio",
  "capacity_norm",
  "over_cap",
  "x_norm",
  "y_norm",
  "type_oh_0",
  "type_oh_1",
  "type_oh_2",
  "type_oh_3",
  "type_oh_4",
  "incoming_p1",
  "incoming_p2",
  "incoming_friendly",
  "incoming_hostile",
  "threat_capped",
  "will_fall",
  "near_cap",
];

export const GROUP_NAMES = [
  "alive",
  "is_p1",
  "is_p2",
  "progress_frac",
  "count_norm",
  "src_x",
  "src_y",
  "tgt_x",
  "tgt_y",
];

export const N_BUILDINGS = 32;
export const N_GROUPS    = 32;

/** Sim constants pulled or fallback. CAP_NORM tracks DEFAULT_CAPACITY = 30 * SCALE. */
export function constantsFromData(data) {
  const scale = data.scale ?? 10;
  return {
    SCALE:               scale,
    CAP_NORM:            scale * 30,            // DEFAULT_CAPACITY
    POS_NORM:            700,
    TIMEOUT_NORM:        data.game_timeout_ticks ?? 200,
    TRAVEL_NORM:         8,
    COUNT_SUM_NORM:      scale * 30 * 4,
    BUILDING_COUNT_NORM: 32,
    PROD_PER_TICK:       data.prod_per_tick ?? 10,
  };
}

/** Walk the event log up to tick t, return per-slot building state + the
 * list of active flights. Cheap enough to call per render (~few hundred ops
 * per game). */
export function deriveStateAt(data, t) {
  const C = constantsFromData(data);
  const owner    = new Array(N_BUILDINGS).fill(0);
  const garrison = new Array(N_BUILDINGS).fill(0);
  const capacity = new Array(N_BUILDINGS).fill(0);
  const type     = new Array(N_BUILDINGS).fill(0);
  const x        = new Array(N_BUILDINGS).fill(0);
  const y        = new Array(N_BUILDINGS).fill(0);
  const alive    = new Array(N_BUILDINGS).fill(0);

  for (const b of data.map.buildings) {
    const s = b.slot;
    owner[s]    = b.init.owner;
    garrison[s] = b.init.garrison;
    capacity[s] = b.capacity;
    type[s]     = b.type;
    x[s]        = b.x;
    y[s]        = b.y;
    alive[s]    = 1;
  }

  // Bucket events ≤ t.
  const tFloor = Math.floor(t);
  const sends  = new Map();    // group id -> {owner, src, tgt, count, send_t, arrive_t}
  const flights = [];          // currently active (sent ≤ t, not arrived ≤ t)

  // Apply production for ticks 1..tFloor + events at each tick.
  // Mirrors computeForceSeries() in game.html, but keeps per-slot state and
  // active flight list rather than aggregating.
  const byT = new Map();
  for (const e of data.events) {
    if (!byT.has(e.t)) byT.set(e.t, []);
    byT.get(e.t).push(e);
  }

  for (let tick = 0; tick <= tFloor; tick++) {
    if (tick > 0) {
      for (let s = 0; s < N_BUILDINGS; s++) {
        if (alive[s] && (owner[s] === 1 || owner[s] === 2) && garrison[s] < capacity[s]) {
          garrison[s] = Math.min(capacity[s], garrison[s] + C.PROD_PER_TICK);
        }
      }
    }
    const evs = byT.get(tick);
    if (evs) {
      for (const e of evs) {
        if (e.kind === "send") {
          sends.set(e.group, { owner: e.owner, src: e.src, tgt: e.dst, count: e.count, send_t: e.t, arrive_t: e.arrive_t });
          garrison[e.src] = e.src_garrison_after;
        } else if (e.kind === "arrive") {
          owner[e.dst]    = e.dst_owner_after;
          garrison[e.dst] = e.dst_garrison_after;
          sends.delete(e.group);
        }
      }
    }
  }

  // Active flights at the (fractional) display tick t — anything in `sends`
  // whose arrive_t > t. Sim-side: arrive_t = send_t + travel_ticks - 1, and
  // the spawn-tick itself burns one progress (replay.py timing comment), so
  // travel = arrive_t - send_t + 1 and progress(t) = (t - send_t) + 1.
  for (const [, s] of sends) {
    if (s.arrive_t > t) {
      const travel = Math.max(1, s.arrive_t - s.send_t + 1);
      const progress = Math.max(0, Math.min(travel, (t - s.send_t) + 1));
      flights.push({
        owner: s.owner, src: s.src, tgt: s.tgt, count: s.count,
        progress, travel,
      });
    }
  }

  return { owner, garrison, capacity, type, x, y, alive, flights, C };
}

/** Encoder-equivalent feature dict at tick t.
 * Returns: { globals: { name: number }, building: { name: [N] }, group: { name: [N] } } */
export function computeFeatures(data, t) {
  const st = deriveStateAt(data, t);
  const { owner, garrison, capacity, type, x, y, alive, flights, C } = st;

  // Per-building derived
  const is_p1 = new Array(N_BUILDINGS);
  const is_p2 = new Array(N_BUILDINGS);
  const is_n  = new Array(N_BUILDINGS);
  const garr_ratio = new Array(N_BUILDINGS);
  for (let i = 0; i < N_BUILDINGS; i++) {
    is_p1[i] = (owner[i] === 1 && alive[i]) ? 1 : 0;
    is_p2[i] = (owner[i] === 2 && alive[i]) ? 1 : 0;
    is_n[i]  = (owner[i] === 0 && alive[i]) ? 1 : 0;
    const cap = capacity[i] > 0 ? capacity[i] : 1;
    garr_ratio[i] = garrison[i] / cap;
  }

  // Incoming flight aggregates per target slot
  const incoming_p1 = new Array(N_BUILDINGS).fill(0);
  const incoming_p2 = new Array(N_BUILDINGS).fill(0);
  for (const f of flights) {
    if (f.tgt < 0 || f.tgt >= N_BUILDINGS) continue;
    if (f.owner === 1) incoming_p1[f.tgt] += f.count;
    else if (f.owner === 2) incoming_p2[f.tgt] += f.count;
  }
  const incoming_friendly = new Array(N_BUILDINGS).fill(0);
  const incoming_hostile  = new Array(N_BUILDINGS).fill(0);
  for (let i = 0; i < N_BUILDINGS; i++) {
    if (is_p1[i]) { incoming_friendly[i] = incoming_p1[i]; incoming_hostile[i] = incoming_p2[i]; }
    else if (is_p2[i]) { incoming_friendly[i] = incoming_p2[i]; incoming_hostile[i] = incoming_p1[i]; }
    else if (is_n[i])  { incoming_hostile[i]  = incoming_p1[i] + incoming_p2[i]; }
  }

  // GLOBALS
  let p1_b = 0, p2_b = 0, n_b = 0, p1_g = 0, p2_g = 0, p1_f = 0, p2_f = 0;
  for (let i = 0; i < N_BUILDINGS; i++) {
    if (is_p1[i]) { p1_b++; p1_g += garrison[i]; }
    if (is_p2[i]) { p2_b++; p2_g += garrison[i]; }
    if (is_n[i])  { n_b++; }
  }
  for (const f of flights) {
    if (f.owner === 1) p1_f += f.count;
    else if (f.owner === 2) p2_f += f.count;
  }
  const p1_total = p1_g + p1_f;
  const p2_total = p2_g + p2_f;
  const tot      = p1_total + p2_total + 1e-6;
  const tickF    = Math.max(0, t);

  const globals = {
    tick:                    tickF / C.TIMEOUT_NORM,
    time_remaining:          1 - tickF / C.TIMEOUT_NORM,
    p1_buildings_share:      p1_b / C.BUILDING_COUNT_NORM,
    p2_buildings_share:      p2_b / C.BUILDING_COUNT_NORM,
    neutral_buildings_share: n_b  / C.BUILDING_COUNT_NORM,
    p1_total_units:          p1_total / C.COUNT_SUM_NORM,
    p2_total_units:          p2_total / C.COUNT_SUM_NORM,
    p1_share_of_units:       p1_total / tot,
    building_margin:         (p1_b - p2_b) / C.BUILDING_COUNT_NORM,
    unit_margin:             (p1_total - p2_total) / C.COUNT_SUM_NORM,
  };

  // BUILDING block
  const blank = () => new Array(N_BUILDINGS).fill(0);
  const building = {};
  for (const name of BUILDING_NAMES) building[name] = blank();
  for (let i = 0; i < N_BUILDINGS; i++) {
    building.alive[i]          = alive[i];
    building.is_p1[i]          = is_p1[i];
    building.is_p2[i]          = is_p2[i];
    building.is_neutral[i]     = is_n[i];
    building.garrison_norm[i]  = garrison[i] / C.CAP_NORM;
    building.garrison_ratio[i] = garr_ratio[i];
    building.capacity_norm[i]  = capacity[i] / C.CAP_NORM;
    building.over_cap[i]       = (alive[i] && garrison[i] > capacity[i]) ? 1 : 0;
    building.x_norm[i]         = x[i] / C.POS_NORM;
    building.y_norm[i]         = y[i] / C.POS_NORM;
    const ti = Math.max(0, Math.min(4, type[i] | 0));
    building[`type_oh_${ti}`][i] = alive[i];
    building.incoming_p1[i]       = incoming_p1[i] / C.CAP_NORM;
    building.incoming_p2[i]       = incoming_p2[i] / C.CAP_NORM;
    building.incoming_friendly[i] = incoming_friendly[i] / C.CAP_NORM;
    building.incoming_hostile[i]  = incoming_hostile[i] / C.CAP_NORM;
    building.threat_capped[i]     = Math.min(incoming_hostile[i], garrison[i]) / C.CAP_NORM;
    building.will_fall[i]         = (alive[i] && incoming_hostile[i] > garrison[i]) ? 1 : 0;
    building.near_cap[i]          = (alive[i] && garr_ratio[i] > 0.95) ? 1 : 0;
  }

  // GROUP block — slot index here is just the order flights appear in the
  // active list; encoder uses sim slot ids, but for "what's in flight right
  // now" purposes the order doesn't carry meaning. Pad to N_GROUPS.
  const group = {};
  for (const name of GROUP_NAMES) group[name] = new Array(N_GROUPS).fill(0);
  const fa = flights.slice(0, N_GROUPS);
  for (let i = 0; i < fa.length; i++) {
    const f = fa[i];
    const travelSafe = f.travel > 0 ? f.travel : 1;
    const frac = Math.max(0, Math.min(1, f.progress / travelSafe));
    group.alive[i]         = 1;
    group.is_p1[i]         = (f.owner === 1) ? 1 : 0;
    group.is_p2[i]         = (f.owner === 2) ? 1 : 0;
    group.progress_frac[i] = frac;
    group.count_norm[i]    = f.count / C.CAP_NORM;
    group.src_x[i]         = (x[f.src] ?? 0) / C.POS_NORM;
    group.src_y[i]         = (y[f.src] ?? 0) / C.POS_NORM;
    group.tgt_x[i]         = (x[f.tgt] ?? 0) / C.POS_NORM;
    group.tgt_y[i]         = (y[f.tgt] ?? 0) / C.POS_NORM;
  }

  return { globals, building, group };
}

/** Pre-compute the 10 globals for every integer tick in a single forward
 * walk — used to draw sparklines without recomputing history per frame. */
export function precomputeGlobalSeries(data) {
  const C = constantsFromData(data);
  const maxT = data.duration_ticks;

  const owner    = new Array(N_BUILDINGS).fill(0);
  const garrison = new Array(N_BUILDINGS).fill(0);
  const capacity = new Array(N_BUILDINGS).fill(0);
  const alive    = new Array(N_BUILDINGS).fill(0);
  for (const b of data.map.buildings) {
    owner[b.slot]    = b.init.owner;
    garrison[b.slot] = b.init.garrison;
    capacity[b.slot] = b.capacity;
    alive[b.slot]    = 1;
  }
  const sends = new Map();
  const byT = new Map();
  for (const e of data.events) {
    if (!byT.has(e.t)) byT.set(e.t, []);
    byT.get(e.t).push(e);
  }

  const series = {};
  for (const name of GLOBAL_NAMES) series[name] = new Array(maxT + 1);

  for (let t = 0; t <= maxT; t++) {
    if (t > 0) {
      for (let s = 0; s < N_BUILDINGS; s++) {
        if (alive[s] && (owner[s] === 1 || owner[s] === 2) && garrison[s] < capacity[s]) {
          garrison[s] = Math.min(capacity[s], garrison[s] + C.PROD_PER_TICK);
        }
      }
    }
    const evs = byT.get(t);
    if (evs) {
      for (const e of evs) {
        if (e.kind === "send") {
          sends.set(e.group, { owner: e.owner, src: e.src, tgt: e.dst, count: e.count, send_t: e.t, arrive_t: e.arrive_t });
          garrison[e.src] = e.src_garrison_after;
        } else if (e.kind === "arrive") {
          owner[e.dst]    = e.dst_owner_after;
          garrison[e.dst] = e.dst_garrison_after;
          sends.delete(e.group);
        }
      }
    }
    let p1_b = 0, p2_b = 0, n_b = 0, p1_g = 0, p2_g = 0;
    for (let s = 0; s < N_BUILDINGS; s++) {
      if (!alive[s]) continue;
      if (owner[s] === 1) { p1_b++; p1_g += garrison[s]; }
      else if (owner[s] === 2) { p2_b++; p2_g += garrison[s]; }
      else if (owner[s] === 0) { n_b++; }
    }
    let p1_f = 0, p2_f = 0;
    for (const [, s] of sends) {
      if (s.arrive_t > t) {
        if (s.owner === 1) p1_f += s.count;
        else if (s.owner === 2) p2_f += s.count;
      }
    }
    const p1_total = p1_g + p1_f;
    const p2_total = p2_g + p2_f;
    const tot      = p1_total + p2_total + 1e-6;
    series.tick[t]                    = t / C.TIMEOUT_NORM;
    series.time_remaining[t]          = 1 - t / C.TIMEOUT_NORM;
    series.p1_buildings_share[t]      = p1_b / C.BUILDING_COUNT_NORM;
    series.p2_buildings_share[t]      = p2_b / C.BUILDING_COUNT_NORM;
    series.neutral_buildings_share[t] = n_b  / C.BUILDING_COUNT_NORM;
    series.p1_total_units[t]          = p1_total / C.COUNT_SUM_NORM;
    series.p2_total_units[t]          = p2_total / C.COUNT_SUM_NORM;
    series.p1_share_of_units[t]       = p1_total / tot;
    series.building_margin[t]         = (p1_b - p2_b) / C.BUILDING_COUNT_NORM;
    series.unit_margin[t]             = (p1_total - p2_total) / C.COUNT_SUM_NORM;
  }
  return { series, maxT };
}
