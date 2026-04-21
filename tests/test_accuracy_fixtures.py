"""Fixture-driven accuracy scenarios for the Mushroom Wars v2 simulator.

Each JSON fixture describes:
  - a tiny custom level/state
  - optional in-flight groups
  - scripted actions by tick
  - expected end-state after N ticks

The goal is to make combat/movement/production behaviour auditable with
human-readable scenario files instead of burying every case in Python code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim import config as C
from sim.actions import Action
from sim.engine import step_tick
from sim.state import empty_state, precompute_distances


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "levels" / "accuracy"

OWNER_MAP = {
    "neutral": C.OWNER_NEUTRAL,
    "p1": C.OWNER_P1,
    "p2": C.OWNER_P2,
}

PHASE_MAP = {
    "playing": C.PHASE_PLAYING,
    "p1_wins": C.PHASE_P1_WINS,
    "p2_wins": C.PHASE_P2_WINS,
    "draw": C.PHASE_DRAW,
}

PHASE_NAMES = {value: key for key, value in PHASE_MAP.items()}


def _to_internal(entry: dict, real_key: str, internal_key: str, default: int = 0) -> int:
    if internal_key in entry:
        return int(entry[internal_key])
    if real_key in entry:
        return int(round(float(entry[real_key]) * C.SCALE))
    return default


def _load_fixture(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fixture_paths():
    params = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        data = _load_fixture(path)
        marks = []
        if "xfail_reason" in data:
            marks.append(pytest.mark.xfail(strict=True, reason=data["xfail_reason"]))
        params.append(pytest.param(path, id=data["name"], marks=marks))
    return params


def _build_state(data: dict):
    state = empty_state()
    b = state.buildings
    g = state.unit_groups

    b[:] = 0
    g[:] = 0

    for building in data["buildings"]:
        slot = int(building["slot"])
        b[slot]["alive"] = int(building.get("alive", 1))
        b[slot]["owner"] = OWNER_MAP[building["owner"]]
        b[slot]["type_id"] = int(building.get("type_id", C.TYPE_BASIC))
        b[slot]["garrison"] = _to_internal(building, "garrison_real", "garrison_internal")
        b[slot]["capacity"] = _to_internal(
            building,
            "capacity_real",
            "capacity_internal",
            default=C.DEFAULT_CAPACITY,
        )
        b[slot]["x"] = int(building["x"])
        b[slot]["y"] = int(building["y"])

    precompute_distances(state)

    for group in data.get("unit_groups", []):
        slot = int(group["slot"])
        g[slot]["alive"] = int(group.get("alive", 1))
        g[slot]["owner"] = OWNER_MAP[group["owner"]]
        g[slot]["src_slot"] = int(group["src_slot"])
        g[slot]["tgt_slot"] = int(group["tgt_slot"])
        g[slot]["count"] = _to_internal(group, "count_real", "count_internal")
        g[slot]["progress"] = int(group.get("progress", 0))
        g[slot]["travel_ticks"] = int(group["travel_ticks"])

    state.tick = int(data.get("tick", 0))
    state.phase = PHASE_MAP[data.get("phase", "playing")]
    for key in state.perf:
        state.perf[key] = 0

    return state


def _actions_for_tick(data: dict, tick: int):
    action_p1 = None
    action_p2 = None

    for action in data.get("actions", []):
        if int(action["tick"]) != tick:
            continue
        decoded = Action(
            kind="send",
            type_idx=int(action["type_idx"]),
            src=int(action["src"]),
            tgt=int(action["tgt"]),
        )
        player = action["player"]
        if player == "p1":
            action_p1 = decoded
        elif player == "p2":
            action_p2 = decoded
        else:
            raise ValueError(f"Unknown player {player!r}")

    return action_p1, action_p2


def _assert_numeric(actual: int, spec: dict, prefix: str):
    if f"{prefix}_eq" in spec:
        assert actual == int(spec[f"{prefix}_eq"])
    if f"{prefix}_lt" in spec:
        assert actual < int(spec[f"{prefix}_lt"])
    if f"{prefix}_gt" in spec:
        assert actual > int(spec[f"{prefix}_gt"])
    if f"{prefix}_le" in spec:
        assert actual <= int(spec[f"{prefix}_le"])
    if f"{prefix}_ge" in spec:
        assert actual >= int(spec[f"{prefix}_ge"])


def _assert_buildings(state, expected: dict):
    for building in expected.get("buildings", []):
        slot = int(building["slot"])
        actual = state.buildings[slot]

        if "owner" in building:
            assert int(actual["owner"]) == OWNER_MAP[building["owner"]]
        if "alive" in building:
            assert int(actual["alive"]) == int(building["alive"])

        _assert_numeric(int(actual["garrison"]), building, "garrison_internal")
        _assert_numeric(int(actual["capacity"]), building, "capacity_internal")


def _assert_expected(state, rewards_total: tuple[float, float], expected: dict):
    if "tick_eq" in expected:
        assert state.tick == int(expected["tick_eq"])
    if "phase" in expected:
        assert state.phase == PHASE_MAP[expected["phase"]]
    if "unit_groups_alive_eq" in expected:
        alive = int(np.sum(state.unit_groups["alive"] == 1))
        assert alive == int(expected["unit_groups_alive_eq"])

    if "reward_p1_total_eq" in expected:
        assert rewards_total[0] == pytest.approx(float(expected["reward_p1_total_eq"]))
    if "reward_p2_total_eq" in expected:
        assert rewards_total[1] == pytest.approx(float(expected["reward_p2_total_eq"]))

    _assert_buildings(state, expected)


def _owner_name(owner: int) -> str:
    return {
        C.OWNER_NEUTRAL: "neutral",
        C.OWNER_P1: "p1",
        C.OWNER_P2: "p2",
    }.get(owner, str(owner))


def _fmt_units(internal: int) -> str:
    return f"{internal / C.SCALE:g}"


def _actual_summary(state, rewards_total: tuple[float, float], expected: dict) -> str:
    parts = []
    for building in expected.get("buildings", []):
        slot = int(building["slot"])
        rec = state.buildings[slot]
        parts.append(
            f"s{slot} {_owner_name(int(rec['owner']))} {_fmt_units(int(rec['garrison']))}"
        )

    if "unit_groups_alive_eq" in expected:
        alive = int(np.sum(state.unit_groups["alive"] == 1))
        parts.append(f"groups {alive}")

    if rewards_total != (0.0, 0.0):
        parts.append(f"r {rewards_total[0]:.1f}/{rewards_total[1]:.1f}")

    phase_name = PHASE_NAMES.get(state.phase, str(state.phase))
    if phase_name != "playing":
        parts.append(f"phase {phase_name}")

    return "; ".join(parts)


@pytest.mark.parametrize("fixture_path", _fixture_paths())
def test_accuracy_fixture(fixture_path: Path, request: pytest.FixtureRequest):
    data = _load_fixture(fixture_path)
    state = _build_state(data)

    r1_total = 0.0
    r2_total = 0.0
    for _ in range(int(data["steps"])):
        action_p1, action_p2 = _actions_for_tick(data, state.tick)
        r1, r2, _ = step_tick(state, action_p1=action_p1, action_p2=action_p2)
        r1_total += r1
        r2_total += r2

    request.node.accuracy_row = {
        "name": data["name"],
        "setup": data["setup_summary"],
        "expected": data["expected_summary"],
        "actual": _actual_summary(state, (r1_total, r2_total), data["expected"]),
        "xfail_reason": data.get("xfail_reason"),
    }

    _assert_expected(state, (r1_total, r2_total), data["expected"])
