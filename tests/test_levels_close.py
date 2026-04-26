"""Tests for `random_close_*` close-map level generators.

Close maps shrink the playable square (350×350 default) so games end
faster, which is the phase-1 lever in CURRICULUM_PLAN.md §3.2.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from sim import config as C
from sim.levels import (
    _CLOSE_MAP_SIZE,
    generate_random_close_level,
    generate_random_level,
    reset,
)


@pytest.mark.parametrize("seed", list(range(10)))
def test_random_close_level_within_close_map(seed):
    """Every building on a close-map level must fit inside the close-map square."""
    rng = np.random.default_rng(seed)
    level = generate_random_close_level(6, rng)
    for owner, x, y, garrison, type_id in level:
        assert 0 <= x <= _CLOSE_MAP_SIZE, f"x={x} out of bounds (seed={seed})"
        assert 0 <= y <= _CLOSE_MAP_SIZE, f"y={y} out of bounds (seed={seed})"


@pytest.mark.parametrize("level_name", ["random_close_4_6", "random_close_6_10"])
def test_random_close_level_via_reset(level_name):
    """`reset(level_name)` accepts the new close-map names and produces a valid State."""
    s = reset(level_name=level_name, seed=42)
    alive_count = int((s.buildings_alive == 1).sum())
    m = re.match(r"^random_close_(\d+)_(\d+)$", level_name)
    n_min, n_max = int(m.group(1)), int(m.group(2))
    # placement might give up below n_max but never above; should be in [2, n_max].
    assert 2 <= alive_count <= n_max
    # All alive buildings should be inside the close-map box.
    xs = s.buildings_x[s.buildings_alive == 1]
    ys = s.buildings_y[s.buildings_alive == 1]
    assert (xs <= _CLOSE_MAP_SIZE).all()
    assert (ys <= _CLOSE_MAP_SIZE).all()


def test_random_close_180_degree_symmetry():
    """Close maps preserve the 180° rotational symmetry that fairness requires."""
    rng = np.random.default_rng(7)
    level = generate_random_close_level(8, rng, map_size=_CLOSE_MAP_SIZE)
    half = _CLOSE_MAP_SIZE
    # Slot 0 = P1 base, slot 1 = P2 mirror.
    p1 = level[0]
    p2 = level[1]
    assert p1[0] == C.OWNER_P1 and p2[0] == C.OWNER_P2
    assert p1[1] + p2[1] == half
    assert p1[2] + p2[2] == half


def test_close_map_distance_smaller_than_full():
    """Close maps should produce smaller average distances than full maps for
    the same building count — that's the whole point."""
    rng_close = np.random.default_rng(123)
    rng_full  = np.random.default_rng(123)
    close = generate_random_close_level(6, rng_close)
    full  = generate_random_level(6, rng_full)
    # Compute mean pairwise distance.
    def _mean_dist(level):
        coords = np.array([(x, y) for _o, x, y, _g, _t in level])
        d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        return d[d > 0].mean()
    assert _mean_dist(close) < _mean_dist(full), (
        "close-map mean distance should be smaller than full-map"
    )


def test_close_map_uses_v13_reward_when_requested():
    """`reset(..., reward_version=1)` works for close maps (just sanity)."""
    s = reset(level_name="random_close_4_6", seed=0, reward_version=1)
    assert s.reward_version == 1
