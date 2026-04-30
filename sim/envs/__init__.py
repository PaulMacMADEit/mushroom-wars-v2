"""Gymnasium env wrappers around `sim/`. `training/` depends on these; `sim/`
core never depends on training code."""

from typing import Callable, Optional

from sim.envs.mushroom_env import MushroomEnv
from sim.envs.opponents import (
    Opponent,
    greedy_capacity_aware_opponent,
    make_neural_opponent,
    noop_opponent,
    random_legal_opponent,
)


_SIMPLE_OPPONENTS = {
    "random_legal":          random_legal_opponent,
    "noop":                  noop_opponent,
    "greedy_capacity_aware": greedy_capacity_aware_opponent,
}


def make_env(
    seed: int = 0,
    level_name: str = "crossroads_6",
    opponent_name: str = "random_legal",
    opponent_kwargs: Optional[dict] = None,
    reward_version: int = 0,
) -> Callable[[], MushroomEnv]:
    """Factory suitable for gymnasium.vector.{Sync,Async}VectorEnv.

    `opponent_name`:
      - "random_legal" / "noop"  — stateless opponents
      - "neural"                 — self-play; loads weights from disk in the
                                    subprocess. Requires `opponent_kwargs`:
                                      {weights_path, obs_norm_path?, device?}

    `AsyncVectorEnv` runs each factory in a separate process, so the returned
    closure must be picklable — opponent selection happens INSIDE _thunk so
    only strings + dicts cross the pickle boundary.
    """
    opp_kwargs = dict(opponent_kwargs or {})
    # Strip dashboard-only labelling keys so make_neural_opponent doesn't
    # see them. See workers/worker.py:_resolve_opponent_kwargs.
    _label_kwargs = {k: v for k, v in opp_kwargs.items() if k.startswith("_label_")}
    opp_kwargs = {k: v for k, v in opp_kwargs.items() if not k.startswith("_label_")}

    def _thunk() -> MushroomEnv:
        if opponent_name == "neural":
            opponent = make_neural_opponent(**opp_kwargs)
        else:
            if opponent_name not in _SIMPLE_OPPONENTS:
                raise ValueError(f"unknown opponent_name: {opponent_name!r}")
            opponent = _SIMPLE_OPPONENTS[opponent_name]
        return MushroomEnv(
            level_name=level_name, opponent=opponent, seed=seed,
            reward_version=reward_version,
        )

    return _thunk


__all__ = [
    "MushroomEnv",
    "Opponent",
    "noop_opponent",
    "random_legal_opponent",
    "greedy_capacity_aware_opponent",
    "make_neural_opponent",
    "make_env",
]
