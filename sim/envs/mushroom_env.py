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
from sim.state import State


class MushroomEnv(gym.Env):
    """Gymnasium env: one Mushroom Wars game from P1's perspective."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_name: str = "crossroads_6",
        opponent: Optional[Opponent] = None,
        decision_interval: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._level_name = level_name
        self._opponent: Opponent = opponent if opponent is not None else random_legal_opponent
        self._decision_interval = (
            decision_interval if decision_interval is not None else C.DECISION_INTERVAL_TICKS
        )
        self._rng = np.random.default_rng(seed)

        # State is created on first reset(); declared for type-checkers.
        self.state: State = level_reset(self._level_name)

        N = C.MAX_BUILDING_SLOTS
        M = C.MAX_UNIT_GROUP_SLOTS
        i16_hi = np.iinfo(np.int16).max
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
                "tick":               spaces.Box(0, np.iinfo(np.int32).max, shape=(), dtype=np.int32),
                "action_mask":        spaces.Box(0, 1, shape=(ACTION_SPACE_SIZE,), dtype=bool),
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
        self.state = level_reset(level)
        return self._make_obs(C.OWNER_P1), self._make_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        if self.state.phase != C.PHASE_PLAYING:
            # Already terminal — caller should have reset. Return a zero step.
            return self._make_obs(C.OWNER_P1), 0.0, True, False, self._make_info()

        action_p1 = decode(int(action))
        action_p2_idx = int(self._opponent(self.state, self._rng))
        action_p2 = decode(action_p2_idx)

        # First tick of the decision interval carries both players' actions;
        # subsequent ticks run with nothing submitted (implicit idle).
        r1_total = 0.0
        done = False
        for i in range(self._decision_interval):
            a1 = action_p1 if i == 0 else None
            a2 = action_p2 if i == 0 else None
            r1, _r2, done = step_tick(self.state, a1, a2)
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

    def _make_obs(self, player: int) -> dict:
        b = self.state.buildings
        g = self.state.unit_groups
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
            "travel_matrix":      self.state.travel_matrix.copy(),
            "tick":               np.int32(self.state.tick),
            "action_mask":        compute_mask(self.state, player),
        }

    def _make_info(self) -> dict[str, Any]:
        return {
            "phase": int(self.state.phase),
            "tick":  int(self.state.tick),
        }
