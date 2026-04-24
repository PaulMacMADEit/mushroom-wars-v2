from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Backend-parametrised step_tick (Phase 5 of JAX_PORT_PLAN)
# ---------------------------------------------------------------------------
#
# Tests that want to exercise both sim backends consume `backend_step_tick` —
# a callable with the same signature as `sim.engine.step_tick` but routed
# through whichever backend the outer fixture selected.
#
# Usage:
#     def test_something(backend_step_tick):
#         state = reset()
#         r1, r2, done = backend_step_tick(state, a1, a2)
#         ...
#
# Under the hood, for the "jax" backend the state is lifted into a StateJax,
# stepped once, and the mutated fields written back onto the input numpy
# State so tests can keep their usual `state.buildings["owner"][i]` reads.
# The parity harness in tests/test_backend_parity.py already proves this
# round-trip is byte-identical.

def _numpy_step(state, action_p1=None, action_p2=None, events=None):
    from sim.engine import step_tick as _sn
    return _sn(state, action_p1=action_p1, action_p2=action_p2, events=events)


def _jax_step(state, action_p1=None, action_p2=None, events=None):
    """Step via the JAX backend, writing the result back onto the input State.

    Event emission is not supported on the JAX hot path (by design —
    JAX_PORT_PLAN §3.2); if `events is not None` this raises so the caller
    knows to exclude the test from the jax parametrisation.
    """
    if events is not None:
        raise NotImplementedError("JAX backend does not emit events; parametrise only the event-free cases")

    from sim.engine_jax import (
        ACTION_KIND_NOOP,
        ACTION_KIND_SEND,
        encode_action,
        step_tick_single,
    )
    from sim.state_jax import from_numpy_state, to_numpy_state

    def _enc(a):
        if a is None or a.kind == "noop":
            return encode_action(ACTION_KIND_NOOP)
        return encode_action(ACTION_KIND_SEND, a.type_idx, a.src, a.tgt)

    sj = from_numpy_state(state)
    sj, r1, r2, done = step_tick_single(sj, _enc(action_p1), _enc(action_p2))
    back = to_numpy_state(sj)

    # Write-back every gameplay field so the caller's `state` object reflects
    # the new state (mirrors numpy-engine in-place mutation).
    state.buildings_alive[:]    = back.buildings_alive
    state.buildings_owner[:]    = back.buildings_owner
    state.buildings_type[:]     = back.buildings_type
    state.buildings_garrison[:] = back.buildings_garrison
    state.buildings_capacity[:] = back.buildings_capacity
    state.buildings_x[:]        = back.buildings_x
    state.buildings_y[:]        = back.buildings_y
    state.groups_alive[:]    = back.groups_alive
    state.groups_owner[:]    = back.groups_owner
    state.groups_src[:]      = back.groups_src
    state.groups_tgt[:]      = back.groups_tgt
    state.groups_count[:]    = back.groups_count
    state.groups_progress[:] = back.groups_progress
    state.groups_travel[:]   = back.groups_travel
    state.tick  = int(back.tick)
    state.phase = int(back.phase)
    return float(r1), float(r2), bool(done)


@pytest.fixture(params=["numpy", "jax"])
def backend_step_tick(request):
    """Parametrised step_tick for tests that should run on both backends."""
    return _jax_step if request.param == "jax" else _numpy_step


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    row = getattr(item, "accuracy_row", None)
    if row is None:
        return

    status = "Pass" if report.passed else "Fail"
    row["status"] = status
    report.accuracy_row = row


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter):
    rows = []
    for outcome in ("passed", "failed", "xfailed"):
        for report in terminalreporter.stats.get(outcome, []):
            row = getattr(report, "accuracy_row", None)
            if row is not None:
                rows.append(row)

    if not rows:
        return

    terminalreporter.write_sep("=", "accuracy fixtures")
    terminalreporter.write_line("| Test | Setup | Expected outcome | Actual outcome | Status |")
    terminalreporter.write_line("|---|---|---|---|---|")
    for row in rows:
        terminalreporter.write_line(
            f"| {row['name']} | {row['setup']} | {row['expected']} | {row['actual']} | {row['status']} |"
        )
