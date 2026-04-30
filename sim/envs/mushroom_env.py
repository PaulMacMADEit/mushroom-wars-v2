"""Single-agent gymnasium env from P1's perspective.

The opponent for P2 is a swappable callable. The env surface matches
`gymnasium.Env`, so it slots into `AsyncVectorEnv` for rollout parallelism
without custom plumbing.

Observation is a dict of the sim's canonical numpy arrays plus the action
mask. Encoding to a flat float tensor is the trainer's job — keeps `sim/`
free of training-code dependencies.

Decision cadence: per `config.DECISION_INTERVAL_TICKS`, the agent decides
every N sim ticks. One env.step() advances exactly N sim ticks; the supplied
action and the opponent's action apply on the first of those ticks, the
remainder run with no new actions. Reward is the sum over the interval.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, compute_mask, decode
from sim.engine import step_tick
from sim.envs.opponents import Opponent, random_legal_opponent
from sim.levels import reset as level_reset
from sim.state import State, count_owned_units


# v10 helper: encode an Action dataclass into the (4,) int8 history slot.
# kind: 0=noop, 1=send. type_idx/src/tgt are zero for noop.
_ACTION_KIND_NOOP = 0
_ACTION_KIND_SEND = 1


def _action_to_history_row(action) -> np.ndarray:
    if action.kind == "send":
        return np.array(
            [_ACTION_KIND_SEND, int(action.type_idx), int(action.src), int(action.tgt)],
            dtype=np.int8,
        )
    return np.zeros(4, dtype=np.int8)


class MushroomEnv(gym.Env):
    """Gymnasium env: one Mushroom Wars game from P1's perspective."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_name: str = "crossroads_6",
        opponent: Optional[Opponent] = None,
        decision_interval: Optional[int] = None,
        seed: Optional[int] = None,
        recorder: Optional[Any] = None,
        reward_version: int = C.REWARD_VERSION_V12,
    ):
        super().__init__()
        self._level_name = level_name
        self._opponent: Opponent = opponent if opponent is not None else random_legal_opponent
        self._decision_interval = (
            decision_interval if decision_interval is not None else C.DECISION_INTERVAL_TICKS
        )
        self._rng = np.random.default_rng(seed)
        self._reward_version = int(reward_version)
        # Optional replay recorder. When set, the env feeds engine events into
        # its buffer per step_tick and calls absorb_tick so post-tick state
        # reads are correct. See sim/envs/replay.py.
        self._recorder = recorder

        # State is created on first reset(); declared for type-checkers.
        self.state: State = level_reset(self._level_name, reward_version=self._reward_version)

        N = C.MAX_BUILDING_SLOTS
        M = C.MAX_UNIT_GROUP_SLOTS
        K = C.HISTORY_K
        i16_hi = np.iinfo(np.int16).max
        i32_hi = np.iinfo(np.int32).max
        self.observation_space = spaces.Dict(
            {
                "buildings_alive":    spaces.Box(0, 1, shape=(N,), dtype=np.int8),
                "buildings_owner":    spaces.Box(0, 2, shape=(N,), dtype=np.int8),
                "buildings_type":     spaces.Box(0, 127, shape=(N,), dtype=np.int8),
                "buildings_garrison": spaces.Box(0, i16_hi, shape=(N,), dtype=np.int16),
                "buildings_capacity": spaces.Box(0, i16_hi, shape=(N,), dtype=np.int16),
                "buildings_x":        spaces.Box(0, i16_hi, shape=(N,), dtype=np.int16),
                "buildings_y":        spaces.Box(0, i16_hi, shape=(N,), dtype=np.int16),
                "groups_alive":       spaces.Box(0, 1, shape=(M,), dtype=np.int8),
                "groups_owner":       spaces.Box(0, 2, shape=(M,), dtype=np.int8),
                "groups_src":         spaces.Box(0, N - 1, shape=(M,), dtype=np.int8),
                "groups_tgt":         spaces.Box(0, N - 1, shape=(M,), dtype=np.int8),
                "groups_count":       spaces.Box(0, i16_hi, shape=(M,), dtype=np.int16),
                "groups_progress":    spaces.Box(0, i16_hi, shape=(M,), dtype=np.int16),
                "groups_travel":      spaces.Box(0, i16_hi, shape=(M,), dtype=np.int16),
                "travel_matrix":      spaces.Box(0, i16_hi, shape=(N, N), dtype=np.int16),
                "tick":               spaces.Box(0, i32_hi, shape=(), dtype=np.int32),
                "action_mask":        spaces.Box(0, 1, shape=(ACTION_SPACE_SIZE,), dtype=bool),
                # v10 decision-interval bookkeeping.
                "arrivals_p1":          spaces.Box(0, i16_hi, shape=(N,), dtype=np.int16),
                "arrivals_p2":          spaces.Box(0, i16_hi, shape=(N,), dtype=np.int16),
                "prev_buildings_owner": spaces.Box(0, 2, shape=(N,), dtype=np.int8),
                "prev_p1_units_total":  spaces.Box(0, i32_hi, shape=(), dtype=np.int32),
                "prev_p2_units_total":  spaces.Box(0, i32_hi, shape=(), dtype=np.int32),
                # action history: (HISTORY_K, 4) — [kind, type_idx, src, tgt]
                # int8 stored, but spaces.Box requires nonneg-bounded ints; use a
                # generous upper bound (kind/type/idx all small).
                "last_actions_p1":      spaces.Box(0, 127, shape=(K, 4), dtype=np.int8),
                "last_actions_p2":      spaces.Box(0, 127, shape=(K, 4), dtype=np.int8),
            }
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    # ------------------------------------------------------------------
    # gymnasium.Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        level = (options or {}).get("level_name", self._level_name)
        # Per-reset level seed — matters only for dynamic level names like
        # `random_8_32`. Static names ignore it. Pulled from the env's own
        # rng so determinism-under-seed is preserved.
        level_seed = int(self._rng.integers(0, 2**31))
        self.state = level_reset(level, seed=level_seed, reward_version=self._reward_version)
        if self._recorder is not None:
            self._recorder.capture_map(self.state)
        return self._make_obs(C.OWNER_P1), self._make_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self.state.phase != C.PHASE_PLAYING:
            # Already terminal — caller should have reset. Return a zero step.
            return self._make_obs(C.OWNER_P1), 0.0, True, False, self._make_info()

        action_p1 = decode(int(action))
        action_p2_idx = int(self._opponent(self.state, self._rng))
        action_p2 = decode(action_p2_idx)

        # v10: snapshot prev-state at the start of the decision interval.
        # The encoder reads (current vs prev) to surface "tower flipped" /
        # "you took/lost units" — signals that vanish from the per-tick state.
        # arrivals_* are accumulated by the engine; reset them here so the
        # encoder sees only landings WITHIN this interval.
        self._snapshot_decision_boundary(action_p1, action_p2)

        # First tick of the decision interval carries both players' actions;
        # subsequent ticks run with nothing submitted (implicit idle).
        r1_total = 0.0
        done = False
        for i in range(self._decision_interval):
            a1 = action_p1 if i == 0 else None
            a2 = action_p2 if i == 0 else None
            buf = self._recorder.get_tick_events_buffer() if self._recorder is not None else None
            r1, _r2, done = step_tick(self.state, a1, a2, events=buf)
            if self._recorder is not None:
                self._recorder.absorb_tick(self.state)
            r1_total += r1
            if done:
                break

        terminated = done
        truncated = False
        return (
            self._make_obs(C.OWNER_P1),
            float(r1_total),
            terminated,
            truncated,
            self._make_info(),
        )

    def render(self):
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_decision_boundary(self, action_p1, action_p2) -> None:
        """v10 boundary work — runs at the START of every step():

        1. Snapshot current owner array → prev_buildings_owner (so the encoder
           can flag towers that flip during this interval).
        2. Snapshot total internal units per side → prev_p1/p2_units_total
           (encoder uses this for reward_delta).
        3. Reset arrivals_p1/p2 to 0 — engine accumulates landings during the
           interval; encoder reads them at the END (i.e. on the obs returned
           from this step).
        4. Push the new actions to the history ring buffers (idx 0 = newest).
        """
        st = self.state
        st.prev_buildings_owner[:] = st.buildings_owner
        st.prev_p1_units_total = count_owned_units(st, C.OWNER_P1)
        st.prev_p2_units_total = count_owned_units(st, C.OWNER_P2)
        st.arrivals_p1[:] = 0
        st.arrivals_p2[:] = 0
        # Ring-buffer push: shift down, write newest at index 0.
        st.last_actions_p1[1:] = st.last_actions_p1[:-1]
        st.last_actions_p1[0]  = _action_to_history_row(action_p1)
        st.last_actions_p2[1:] = st.last_actions_p2[:-1]
        st.last_actions_p2[0]  = _action_to_history_row(action_p2)

    def _make_obs(self, player: int) -> dict:
        b = self.state.buildings
        g = self.state.unit_groups
        st = self.state
        return {
            "buildings_alive":    b["alive"].copy(),
            "buildings_owner":    b["owner"].copy(),
            "buildings_type":     b["type_id"].copy(),
            "buildings_garrison": b["garrison"].copy(),
            "buildings_capacity": b["capacity"].copy(),
            "buildings_x":        b["x"].copy(),
            "buildings_y":        b["y"].copy(),
            "groups_alive":       g["alive"].copy(),
            "groups_owner":       g["owner"].copy(),
            "groups_src":         g["src_slot"].copy(),
            "groups_tgt":         g["tgt_slot"].copy(),
            "groups_count":       g["count"].copy(),
            "groups_progress":    g["progress"].copy(),
            "groups_travel":      g["travel_ticks"].copy(),
            "travel_matrix":      st.travel_matrix.copy(),
            "tick":               np.int32(st.tick),
            "action_mask":        compute_mask(st, player),
            # v10 decision-interval features. Mirrored P1↔P2 by the
            # opponent-perspective wrapper in opponents._mirror_ownership.
            "arrivals_p1":          st.arrivals_p1.copy(),
            "arrivals_p2":          st.arrivals_p2.copy(),
            "prev_buildings_owner": st.prev_buildings_owner.copy(),
            "prev_p1_units_total":  np.int32(st.prev_p1_units_total),
            "prev_p2_units_total":  np.int32(st.prev_p2_units_total),
            "last_actions_p1":      st.last_actions_p1.copy(),
            "last_actions_p2":      st.last_actions_p2.copy(),
        }

    def _make_info(self) -> dict[str, Any]:
        return {
            "phase": int(self.state.phase),
            "tick":  int(self.state.tick),
        }
