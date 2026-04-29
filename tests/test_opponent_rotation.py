"""Tests for per-update opponent rotation (KARPATHY_LOG fire 65-66).

Regression tests for the bug Paul caught on 2026-04-29: the dashboard chart
collapsed all rotated opponents into a single row labeled `champion:mw2-pfsp`.
Root cause: `_rotate_opponent_for_update` extracted the label from the
*dirname* (shared across all archive members) instead of the filename
(distinct per member).

Test surface:
  - `_extract_label_from_weights_path`: pure helper that does the extraction.
  - End-to-end: many calls against a fake leaderboard should produce all
    distinct labels.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.trainer import _extract_label_from_weights_path


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

def test_pfsp_layout_extracts_champion_id():
    """The PFSP downloader writes files as
    `/tmp/mw2-pfsp-XXXX/{champ_id[:8]}-weights.pt`. The label must be the
    champion id, not the (shared) tempdir basename."""
    assert _extract_label_from_weights_path(
        "/tmp/mw2-pfsp-abc/cdcc0826-weights.pt"
    ) == "cdcc0826"
    assert _extract_label_from_weights_path(
        "/tmp/mw2-pfsp-abc/072fe893-weights.pt"
    ) == "072fe893"


def test_distinct_files_in_shared_dir_yield_distinct_labels():
    """The original bug: dirname-based extraction returned the same string
    for every archive member because they all live in one shared tempdir.
    The fix must pull from the filename so distinct files → distinct labels."""
    paths = [
        "/tmp/mw2-pfsp-Q/cdcc0826-weights.pt",
        "/tmp/mw2-pfsp-Q/072fe893-weights.pt",  # same dir, different file
        "/tmp/mw2-pfsp-Q/e61d06c0-weights.pt",
        "/tmp/mw2-pfsp-Q/abc12345-weights.pt",
    ]
    labels = [_extract_label_from_weights_path(p) for p in paths]
    assert len(set(labels)) == 4, (
        f"Expected 4 distinct labels from 4 distinct files in a shared dir, "
        f"got {labels}"
    )


def test_alternate_path_layouts():
    """Should work with various tempdir conventions, not just `mw2-pfsp-`."""
    assert _extract_label_from_weights_path(
        "/var/tmp/mw2-opp-xyz/abc12345-weights.pt"
    ) == "abc12345"
    assert _extract_label_from_weights_path(
        "/home/paul/runs/cdcc0826-weights.pt"
    ) == "cdcc0826"


def test_no_weights_suffix_falls_through_to_basename():
    """If the filename doesn't have a `-weights.pt` suffix (e.g. snapshot
    files use just `.pt`), strip just `.pt`. If neither matches, return
    the basename."""
    assert _extract_label_from_weights_path(
        "/tmp/somewhere/snap-12345.pt"
    ) == "snap-12345"
    assert _extract_label_from_weights_path(
        "/tmp/no-extension"
    ) == "no-extension"


def test_empty_or_none_returns_question_mark():
    """Defensive: empty/None inputs produce a placeholder, not a crash."""
    assert _extract_label_from_weights_path("") == "?"
    assert _extract_label_from_weights_path(None) == "?"


def test_pathlib_path_works():
    """Caller may pass a pathlib.Path instead of a str."""
    from pathlib import Path
    p = Path("/tmp/mw2-pfsp-Q/cdcc0826-weights.pt")
    assert _extract_label_from_weights_path(p) == "cdcc0826"


# ---------------------------------------------------------------------------
# End-to-end test of the rotation label loop
# ---------------------------------------------------------------------------

def test_rotation_over_many_calls_produces_all_archive_labels():
    """Simulate the rotation hot path: pick an index uniformly from a fake
    leaderboard, extract the label. After enough calls, every archive
    member's label should appear at least once.

    With 4 members and 50 trials, P(missing any member) ≈ 4 × (3/4)^50 ≈ 1e-6,
    so this is a deterministic-enough sanity check.
    """
    rng = np.random.default_rng(seed=0)
    leaderboard = [
        (f"/tmp/mw2-pfsp-abc/{cid}-weights.pt", None, 1.0)
        for cid in ["cdcc0826", "072fe893", "e61d06c0", "abc12345"]
    ]
    seen_labels = set()
    for _ in range(50):
        idx = int(rng.integers(0, len(leaderboard)))
        weights_path = leaderboard[idx][0]
        seen_labels.add(_extract_label_from_weights_path(weights_path))
    assert seen_labels == {"cdcc0826", "072fe893", "e61d06c0", "abc12345"}, (
        f"Expected all 4 archive labels to appear; got {seen_labels}"
    )


def test_pre_fix_buggy_extraction_would_collapse_all_to_one():
    """The bug: extracting from dirname instead of filename collapses all
    distinct archive members into one label. This test documents the bug
    so a future refactor can't accidentally re-introduce it."""
    import os
    paths = [
        "/tmp/mw2-pfsp-X/cdcc0826-weights.pt",
        "/tmp/mw2-pfsp-X/072fe893-weights.pt",
        "/tmp/mw2-pfsp-X/e61d06c0-weights.pt",
    ]
    # Pre-fix logic: basename(dirname(...)) — all same dir → all same label
    buggy_labels = [os.path.basename(os.path.dirname(p)) for p in paths]
    assert len(set(buggy_labels)) == 1, (
        "Expected the buggy extraction to collapse — if this no longer "
        "collapses, the test setup may be wrong."
    )

    # New logic via the helper: distinct.
    fixed_labels = [_extract_label_from_weights_path(p) for p in paths]
    assert len(set(fixed_labels)) == 3, (
        f"The fix should produce 3 distinct labels, got {fixed_labels}"
    )
