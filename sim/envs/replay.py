"""Replay recorder — produces a compact event log for browser playback.

The recorder consumes low-level engine events (`spawn`, `arrive`, `end`) that
`step_tick(..., events=buf)` appends into a buffer, and turns them into a
public event schema keyed by stable per-game group ids.

Design: the browser never runs sim logic. It only interpolates squad
positions linearly between spawn/arrive events, and ticks garrison counters
up by a constant production rate until the next event resets them. So every
event carries sim-authoritative end-of-tick values (`src_garrison_after`,
`dst_garrison_after`, `dst_owner_after`).

Usage:

    recorder = Recorder(game_id=..., level_name=..., seed=...)
    recorder.capture_map(state)                  # after levels.reset()
    buf = recorder.get_tick_events_buffer()
    while not done:
        _, _, done = step_tick(state, a1, a2, events=buf)
        recorder.absorb_tick(state)
    recorder.write_json("out.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sim import config as C
from sim.state import State


MAP_SIZE = 700   # matches sim/levels.py _MAP_SIZE; renderer scales to pixels.


def phase_to_winner(phase: int) -> Optional[int]:
    """Translate engine phase code → winner for the event log."""
    if phase == C.PHASE_P1_WINS:
        return 1
    if phase == C.PHASE_P2_WINS:
        return 2
    if phase == C.PHASE_DRAW:
        return 0
    return None


@dataclass
class Recorder:
    game_id: str
    sim_version: str = "v9.1"
    level_name: str = ""
    seed: Optional[int] = None

    # Opponent metadata so the dashboard can show "vs <full label>" in the
    # replay viewer. opponent_run_id is the short id; opponent_label is the
    # full label (e.g. v13.1.01-Continue-rollout_steps-mid). Default "" for
    # legacy captures (P2=random_legal); trainer sets these when capturing
    # against a rotation pool member.
    opponent_run_id: str = ""
    opponent_label:  str = ""

    _map: Optional[dict] = field(default=None, init=False)
    _events: list[dict] = field(default_factory=list, init=False)
    _decisions: list[dict] = field(default_factory=list, init=False)
    _next_gid: int = field(default=0, init=False)
    _slot_to_gid: dict[int, int] = field(default_factory=dict, init=False)
    _tick_buf: list[dict] = field(default_factory=list, init=False)
    _winner: Optional[int] = field(default=None, init=False)

    def capture_map(self, state: State, map_size: int = MAP_SIZE) -> None:
        """Snapshot initial building layout. Call after levels.reset()."""
        b = state.buildings
        buildings = []
        for slot in range(len(b)):
            if not b["alive"][slot]:
                continue
            buildings.append({
                "slot":     int(slot),
                "x":        int(b["x"][slot]),
                "y":        int(b["y"][slot]),
                "type":     int(b["type_id"][slot]),
                "capacity": int(b["capacity"][slot]),
                "init": {
                    "owner":    int(b["owner"][slot]),
                    "garrison": int(b["garrison"][slot]),
                },
            })
        self._map = {
            "width":     map_size,
            "height":    map_size,
            "buildings": buildings,
        }

    def get_tick_events_buffer(self) -> list:
        """Returns the buffer step_tick writes into. Cleared per tick."""
        self._tick_buf.clear()
        return self._tick_buf

    def absorb_tick(self, state: State) -> None:
        """Fold raw engine events from the buffer into the public event log.

        Must be called AFTER step_tick. Reads end-of-tick state for
        `src_garrison_after` / `dst_garrison_after` / `dst_owner_after`.
        """
        t = int(state.tick)
        b = state.buildings

        for e in self._tick_buf:
            kind = e["kind"]
            if kind == "spawn":
                gid = self._next_gid
                self._next_gid += 1
                self._slot_to_gid[int(e["slot"])] = gid
                src = int(e["src"])
                travel_ticks = int(e["travel_ticks"])
                # Spawn runs in phase 1; the same step_tick advances movement
                # in phase 3, so the group consumes one tick of travel on its
                # spawn tick. Actual arrival lands at t + travel_ticks - 1.
                arrive_t = t + max(0, travel_ticks - 1)
                self._events.append({
                    "t":                  t,
                    "kind":               "send",
                    "group":              gid,
                    "owner":              int(e["owner"]),
                    "src":                src,
                    "dst":                int(e["tgt"]),
                    "count":              int(e["count"]),
                    "arrive_t":           arrive_t,
                    "src_garrison_after": int(b["garrison"][src]),
                })
            elif kind == "arrive":
                slot = int(e["slot"])
                gid = self._slot_to_gid.pop(slot, None)
                if gid is None:
                    # Shouldn't happen — every arrive follows a recorded spawn.
                    continue
                dst = int(e["tgt"])
                self._events.append({
                    "t":                   t,
                    "kind":                "arrive",
                    "group":               gid,
                    "dst":                 dst,
                    "dst_owner_after":     int(b["owner"][dst]),
                    "dst_garrison_after":  int(b["garrison"][dst]),
                })
            elif kind == "capture":
                self._events.append({
                    "t":              t,
                    "kind":           "capture",
                    "tgt":            int(e["tgt"]),
                    "owner_before":   int(e["owner_before"]),
                    "owner_after":    int(e["owner_after"]),
                    "garrison_after": int(e["garrison_after"]),
                })
            elif kind == "end":
                w = phase_to_winner(int(e["phase"]))
                self._winner = w
                self._events.append({
                    "t":      t,
                    "kind":   "end",
                    "winner": w,
                })

    def record_decision(
        self,
        tick: int,
        player: int,
        diag: dict,
        top_k: int = 5,
    ) -> None:
        """Attach a per-decision policy snapshot (value, top-k probs, entropy).

        `diag` is the dict returned by `PPOAgent.act_one_with_diag`. We only
        store top-k per head to keep the replay JSON compact — the UI only
        renders the top handful anyway.
        """
        def _top_k(probs, mask, k: int):
            import numpy as np
            valid_idx = list(range(len(probs)))
            if mask is not None:
                valid_idx = [i for i in valid_idx if bool(mask[i])]
            ranked = sorted(valid_idx, key=lambda i: -float(probs[i]))[:k]
            return [[int(i), float(probs[i])] for i in ranked]

        self._decisions.append({
            "t":       int(tick),
            "player":  int(player),
            "value":   float(diag["value"]),
            "entropy": float(diag["entropy"]),
            "picked": {
                "src":  int(diag["src_picked"]),
                "type": int(diag["type_picked"]),
                "tgt":  int(diag["tgt_picked"]),
            },
            "src_top":  _top_k(diag["src_probs"],  diag.get("src_mask"),  top_k),
            "type_top": _top_k(diag["type_probs"], diag.get("type_mask"), top_k),
            "tgt_top":  _top_k(diag["tgt_probs"],  diag.get("tgt_mask"),  top_k),
        })

    def to_dict(self) -> dict[str, Any]:
        duration = self._events[-1]["t"] if self._events else 0
        return {
            "game_id":            self.game_id,
            "sim_version":        self.sim_version,
            "level_name":         self.level_name,
            "seed":               self.seed,
            "tick_hz":            C.TICK_HZ,
            "scale":              C.SCALE,
            "prod_per_tick":      C.PRODUCTION_PER_TICK,
            "game_timeout_ticks": C.GAME_TIMEOUT_TICKS,
            "duration_ticks":     duration,
            "winner":             self._winner,
            "opponent_run_id":    self.opponent_run_id,
            "opponent_label":     self.opponent_label,
            "map":                self._map,
            "events":             self._events,
            "decisions":          self._decisions,
        }

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        return p
