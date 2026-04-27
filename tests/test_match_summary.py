"""Regression test for match_runner.summarize self-play counting bug.

Previously, when run_a_id == run_b_id (true self-play), `winner == a_id`
and `winner == b_id` were both true on every win — so wins_a == wins_b ==
total_wins, which made the dashboard show "P1 100, P2 100" for a 100-game
match.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.match_runner import summarize


SAME_RUN = "00000000-0000-0000-0000-000000000aaa"
OTHER    = "00000000-0000-0000-0000-000000000bbb"


def _result(winner, phase, swapped):
    return {"winner": winner, "stats": {"phase": phase, "swapped": swapped}}


def test_self_play_60_p1_40_p2():
    """100-game self-play: 60 P1 wins, 40 P2 wins (asym map). swap alternates."""
    games = []
    for i in range(100):
        swapped = (i % 2 == 1)
        # Pretend P1 wins 60 games, P2 wins 40. Distribute across swap states.
        if i < 60:
            phase = 1   # P1 wins
        else:
            phase = 2   # P2 wins
        games.append(_result(SAME_RUN, phase, swapped))

    s = summarize(games, SAME_RUN, SAME_RUN)
    assert s["self_play"] is True
    assert s["games"] == 100
    assert s["wins_p1"] == 60
    assert s["wins_p2"] == 40
    # wins_a + wins_b should equal total non-draw, NOT double-count.
    assert s["wins_a"] + s["wins_b"] == 100  # was: 200 in the old buggy code
    # Each side appears as P1 50 times under alternating swap; A's wins are
    # the games-where-A-was-P1-and-P1-won + games-where-A-was-P2-and-P2-won.
    # i in [0..59] phase=1: even-i (swap=False, A=P1) → A wins. Of these:
    #   even i in [0..59]: 0,2,...,58 → 30 wins for A.
    # i in [0..59] phase=1: odd-i (swap=True, A=P2) → P1 wins → B wins.
    # i in [60..99] phase=2: even-i (swap=False, A=P1) → P2 wins → B wins.
    # i in [60..99] phase=2: odd-i (swap=True, A=P2 = whoever was at P2... wait)
    # Actually: swap=True means B-is-P1, A-is-P2. So phase=2 (P2 wins) → A wins.
    # i in [60..99] odd: 61,63,...,99 → 20 wins for A.
    # Total A: 30 + 20 = 50. B: 50.
    assert s["wins_a"] == 50
    assert s["wins_b"] == 50


def test_head_to_head_uses_winner_id():
    """Non-self-play: wins_a/wins_b come from winner field directly."""
    games = [
        _result(SAME_RUN, 1, False),    # A wins as P1
        _result(OTHER,    2, False),    # B wins as P2 (B is P2 since swap=False, B is opponent)
        _result(OTHER,    1, True),     # B as P1, B wins
        _result(SAME_RUN, 2, True),     # A as P2, A wins
        _result(None,     3, False),    # draw
    ]
    s = summarize(games, SAME_RUN, OTHER)
    assert s["self_play"] is False
    assert s["wins_a"] == 2
    assert s["wins_b"] == 2
    assert s["draws"] == 1
    # Engine sides:
    assert s["wins_p1"] == 2
    assert s["wins_p2"] == 2


def test_empty_results():
    s = summarize([], SAME_RUN, SAME_RUN)
    assert s["games"] == 0
    assert s["wins_a"] == 0
    assert s["rate_a"] == 0.0


def test_self_play_all_p1_winning():
    """Edge case: 100 games, P1 wins all (perfectly asymmetric map).
    With swap alternating, A and B each won 50 times across the rotation."""
    games = [_result(SAME_RUN, 1, i % 2 == 1) for i in range(100)]
    s = summarize(games, SAME_RUN, SAME_RUN)
    assert s["wins_p1"] == 100
    assert s["wins_p2"] == 0
    assert s["wins_a"] == 50  # half the games A was P1 (50 wins)
    assert s["wins_b"] == 50  # half the games B was P1 (50 wins)
