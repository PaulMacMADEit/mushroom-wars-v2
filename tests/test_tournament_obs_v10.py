"""Regression test for the bug where `scripts.tournament._state_to_obs_dict_for_player`
was missing the v10 fields. The tournament path is what `_auto_rate_run` and
`bench_eval` call to evaluate a freshly-trained run vs random_legal — if the
obs dict is missing v10 keys, `encode_obs` raises KeyError and the eval is
silently lost (no champion gets admitted to the archive).

Caught 2026-04-29: every karpv2 v10 cell's bench_eval was failing with
`'arrivals_p1'` for hours, producing 0 v10 champions despite healthy training.
"""

import numpy as np

from sim import config as C
from sim.envs.mushroom_env import MushroomEnv
from sim.actions import compute_mask
from training.encoder import encode_obs
from scripts.tournament import _state_to_obs_dict_for_player


V10_KEYS = (
    "arrivals_p1",
    "arrivals_p2",
    "prev_buildings_owner",
    "prev_p1_units_total",
    "prev_p2_units_total",
    "last_actions_p1",
    "last_actions_p2",
)


def _fresh_state():
    env = MushroomEnv(level_name="crossroads_6", seed=42)
    env.reset(seed=42)
    return env.state


def test_obs_dict_has_v10_keys_p1():
    state = _fresh_state()
    mask = compute_mask(state, C.OWNER_P1)
    out = _state_to_obs_dict_for_player(state, mask, C.OWNER_P1)
    for k in V10_KEYS:
        assert k in out, f"v10 key {k!r} missing from tournament obs dict (P1)"


def test_obs_dict_has_v10_keys_p2():
    state = _fresh_state()
    mask = compute_mask(state, C.OWNER_P2)
    out = _state_to_obs_dict_for_player(state, mask, C.OWNER_P2)
    for k in V10_KEYS:
        assert k in out, f"v10 key {k!r} missing from tournament obs dict (P2)"


def test_encode_obs_succeeds_on_tournament_obs_dict():
    """The actual contract — the obs dict must encode without KeyError."""
    state = _fresh_state()
    for player in (C.OWNER_P1, C.OWNER_P2):
        mask = compute_mask(state, player)
        d = _state_to_obs_dict_for_player(state, mask, player)
        vec = encode_obs(d)
        # OBS_DIM is the v10 width; if encode_obs returned a different shape
        # the caller would silently misalign with the trained net.
        from training.encoder import OBS_DIM
        assert vec.shape == (OBS_DIM,), f"player {player}: got {vec.shape}, expected ({OBS_DIM},)"


def test_p2_mirror_swaps_arrivals_and_history():
    """When the P2 obs is built, arrivals_p1↔arrivals_p2 (and similar) must
    swap — otherwise the encoder would see "my arrivals" as enemy arrivals."""
    state = _fresh_state()
    # Force non-zero arrivals so we can see the swap.
    state.arrivals_p1[0] = 7
    state.arrivals_p2[1] = 11
    state.prev_p1_units_total = 100
    state.prev_p2_units_total = 200

    mask_p1 = compute_mask(state, C.OWNER_P1)
    mask_p2 = compute_mask(state, C.OWNER_P2)
    p1_obs = _state_to_obs_dict_for_player(state, mask_p1, C.OWNER_P1)
    p2_obs = _state_to_obs_dict_for_player(state, mask_p2, C.OWNER_P2)

    assert p1_obs["arrivals_p1"][0] == 7
    assert p1_obs["arrivals_p2"][1] == 11
    assert int(p1_obs["prev_p1_units_total"]) == 100
    assert int(p1_obs["prev_p2_units_total"]) == 200

    # P2 perspective — labels swap so P2's encoder sees its own stuff as p1.
    assert p2_obs["arrivals_p1"][1] == 11, "P2 mirror: arrivals_p1 must hold P2's actual arrivals"
    assert p2_obs["arrivals_p2"][0] == 7,  "P2 mirror: arrivals_p2 must hold P1's actual arrivals"
    assert int(p2_obs["prev_p1_units_total"]) == 200
    assert int(p2_obs["prev_p2_units_total"]) == 100
