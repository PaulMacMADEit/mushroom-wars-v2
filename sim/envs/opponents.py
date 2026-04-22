"""Opponent policies used inside MushroomEnv.

Opponents receive full sim State (same process, not a network call) and return
an action index in [0, ACTION_SPACE_SIZE). Self-play plugs a frozen learner in
here later; for the Phase-2 smoke train, random-legal is enough to keep the
signal non-trivial.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX, compute_mask
from sim.state import State


Opponent = Callable[[State, np.random.Generator], int]


def noop_opponent(state: State, rng: np.random.Generator) -> int:
    """Always idle. Weakest possible baseline — useful for isolating learner
    signal from opponent signal during debugging."""
    del state, rng
    return NOOP_INDEX


def random_legal_opponent(state: State, rng: np.random.Generator) -> int:
    """Uniform-random over legal actions (including noop).

    Falls back to noop if the mask somehow has no set bits — shouldn't happen
    since noop is always legal, but defensive.
    """
    mask = compute_mask(state, C.OWNER_P2)
    legal = np.where(mask)[0]
    if legal.size == 0:
        return NOOP_INDEX
    return int(rng.choice(legal))
