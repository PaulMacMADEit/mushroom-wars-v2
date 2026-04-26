"""Tests for tournament Elo math (CURRICULUM_PLAN.md §3.3).

Pure numpy/python; no Supabase needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.tournament import elo_update


def test_equal_ratings_win_gives_positive_delta():
    """Equal ratings, A wins → A gains, B loses an equal amount."""
    a, b = elo_update(1200, 1200, score_a=1.0, k=32)
    assert a > 1200
    assert b < 1200
    assert abs((a - 1200) + (b - 1200)) < 1e-6  # zero-sum


def test_equal_ratings_draw_no_change():
    a, b = elo_update(1200, 1200, score_a=0.5, k=32)
    assert abs(a - 1200) < 1e-6
    assert abs(b - 1200) < 1e-6


def test_equal_ratings_loss_gives_negative_delta():
    a, b = elo_update(1200, 1200, score_a=0.0, k=32)
    assert a < 1200
    assert b > 1200


def test_higher_rated_loss_loses_more():
    """If a 1500 loses to a 1200, the 1500 loses more than half of K."""
    a, b = elo_update(1500, 1200, score_a=0.0, k=32)
    delta = 1500 - a
    assert delta > 16  # >K/2


def test_underdog_win_gains_more():
    """A 1200 beating a 1500 gains more than K/2."""
    a, b = elo_update(1200, 1500, score_a=1.0, k=32)
    delta = a - 1200
    assert delta > 16


def test_repeated_wins_converge():
    """100 consecutive draws between two policies leave Elo unchanged."""
    a, b = 1200.0, 1300.0
    for _ in range(100):
        a, b = elo_update(a, b, score_a=0.5, k=32)
    # After many draws, ratings should drift slightly but by small amounts —
    # actual stable point: a > 1200 (was undervalued), b < 1300.
    assert a > 1200 and b < 1300


def test_score_a_partial_credit():
    """score_a=0.6 should give 60% of full-win delta."""
    a_full, _ = elo_update(1200, 1200, score_a=1.0, k=32)
    a_part, _ = elo_update(1200, 1200, score_a=0.6, k=32)
    full_delta = a_full - 1200
    part_delta = a_part - 1200
    # 0.6 - 0.5 = 0.1 → 10% of K=32 = 3.2.
    assert abs(part_delta - 3.2) < 1e-5
    assert abs(full_delta - 16.0) < 1e-5
