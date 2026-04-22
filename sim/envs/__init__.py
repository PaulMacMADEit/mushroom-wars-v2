"""Gymnasium env wrappers around `sim/`. `training/` depends on these; `sim/`
core never depends on training code."""

from typing import Callable, Optional

from sim.envs.mushroom_env import MushroomEnv
from sim.envs.opponents import (
    Opponent,
    noop_opponent,
    random_legal_opponent,
)


def make_env(
    seed: int = 0,
    level_name: str = "crossroads_6",
    opponent_name: str = "random_legal",
) -> Callable[[], MushroomEnv]:
    """Factory suitable for gymnasium.vector.{Sync,Async}VectorEnv.

    `AsyncVectorEnv` runs each factory in a separate process, so the returned
    closure must be picklable — that's why we take an opponent *name* instead
    of a callable reference.
    """
    opponents: dict[str, Opponent] = {
        "random_legal": random_legal_opponent,
        "noop": noop_opponent,
    }
    opponent = opponents[opponent_name]

    def _thunk() -> MushroomEnv:
        return MushroomEnv(level_name=level_name, opponent=opponent, seed=seed)

    return _thunk


__all__ = [
    "MushroomEnv",
    "Opponent",
    "noop_opponent",
    "random_legal_opponent",
    "make_env",
]
