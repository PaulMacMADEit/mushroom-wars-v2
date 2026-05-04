#!/usr/bin/env python
"""Karpathy-loop preflight: catches the v10-cascade-class of bugs in <30 sec.

Run after any major version bump (encoder, sim, model arch) before letting
the karp loop queue real GPU runs. Every check has a 1:1 mapping to a bug
that bit us in production:

  Check 1  — Champion weight loadability per arch_era
             Catches: weight format bumps (v9 flat → v10 wrapped), encoder
             dispatch missing, body/obs_dim mismatch.

  Check 2  — Era-filtered archive integrity
             Catches: bench_eval cross-era opponent loading (returned ALL
             eras → silent crash → unrated runs).

  Check 3  — Encoder + obs-dict-builder contract
             Catches: encoder reads new obs keys that
             tournament._state_to_obs_dict_for_player doesn't supply
             (`KeyError: 'arrivals_p1'`).

  Check 4  — Karp YAML opponent-config consistency
             Catches: YAML says `random_legal` but
             `opponent_pool_mode=rotate_per_update` + `leaderboard_bias>0`
             still trigger archive download → bootstrap crash.

Exits 0 on full pass, 1 on any failure. Output is a punch list.

Usage:
    python scripts/karp_preflight.py
    python scripts/karp_preflight.py --skip-smoke    # skip slowest check
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PASS = "✅"
FAIL = "❌"
SKIP = "⊘"


def _hdr(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_champion_weight_loadability() -> tuple[bool, list[str]]:
    """For every champion in the archive, attempt to load its weights using
    the same code path bench_eval uses (`tournament._load_policy`). Group
    failures by arch_era — one bad era + many champions = one entry, not noise.
    """
    _hdr("CHECK 1 — Champion weight loadability (per era)")
    import torch
    from cli.db import connect

    failures: list[str] = []
    by_era: dict[str, dict[str, int]] = {}

    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT arch_era, source_run_id, label, weights_url
                  FROM champions ORDER BY arch_era, archived_at DESC
            """)
            rows = cur.fetchall()

    if not rows:
        print(f"  {SKIP}  no champions in archive — nothing to check")
        return True, []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from scripts.tournament import _load_policy

    for era, run_id, label, weights_url in rows:
        by_era.setdefault(era, {"ok": 0, "fail": 0, "samples": []})
        try:
            kind, agent, obs_norm, encode_fn = _load_policy(str(run_id), device)
            if kind != "neural":
                raise RuntimeError(f"unexpected kind={kind!r}")
            by_era[era]["ok"] += 1
        except Exception as e:
            by_era[era]["fail"] += 1
            if len(by_era[era]["samples"]) < 2:
                by_era[era]["samples"].append(f"{label}: {type(e).__name__}: {str(e)[:120]}")

    for era, stats in sorted(by_era.items()):
        ok, fail = stats["ok"], stats["fail"]
        mark = PASS if fail == 0 else FAIL
        print(f"  {mark}  {era}: {ok} loaded, {fail} failed")
        for s in stats["samples"]:
            print(f"      ↳ {s}")
        if fail:
            failures.append(f"{fail} {era} champion(s) failed to load")

    return not failures, failures


def check_archive_era_filter() -> tuple[bool, list[str]]:
    """`_get_archive()` should return ONLY rows matching `_current_arch_era()`.
    A regression here causes silent unrated runs.
    """
    _hdr("CHECK 2 — Archive era filter")
    from workers.bench_eval import _current_arch_era, _get_archive

    failures: list[str] = []
    current_era = _current_arch_era()
    print(f"  current arch_era: {current_era!r}")
    archive = _get_archive()

    bad = [c for c in archive if c["arch_era"] != current_era]
    if bad:
        failures.append(
            f"{len(bad)} cross-era champion(s) in result of _get_archive() "
            f"(should be filtered to {current_era!r})"
        )
        print(f"  {FAIL}  {len(bad)} bad rows — filter regression")
        for c in bad[:3]:
            print(f"      ↳ {c['label']}: era={c['arch_era']!r}")
    else:
        print(f"  {PASS}  {len(archive)} champions returned, all era={current_era!r}")

    return not failures, failures


def check_encoder_obs_dict_contract() -> tuple[bool, list[str]]:
    """Build a fresh state via the sim, run the obs-dict-builder, then run
    encode_obs. KeyError here = obs-dict-builder out of sync with encoder.
    """
    _hdr("CHECK 3 — Encoder + obs-dict-builder contract")
    failures: list[str] = []

    try:
        import numpy as np
        from sim import config as C
        from sim.actions import compute_mask_batched
        from sim.envs.jax_vec_env import JaxVecEnv
        from training.encoder import encode_obs
        from scripts.tournament import _state_to_obs_dict_for_player

        vec = JaxVecEnv(n_envs=1, level_name="random_close_4_5", base_seed=0)
        states = vec.snapshot_numpy_states()
        s = states[0]
        bulk_alive    = np.stack([s.buildings_alive])
        bulk_owner    = np.stack([s.buildings_owner])
        bulk_garrison = np.stack([s.buildings_garrison])
        bulk_galive   = np.stack([s.groups_alive])
        masks = compute_mask_batched(bulk_alive, bulk_owner, bulk_garrison, bulk_galive, C.OWNER_P1)

        for player in (C.OWNER_P1, C.OWNER_P2):
            obs_dict = _state_to_obs_dict_for_player(s, masks[0], player)
            obs_arr = encode_obs(obs_dict)
            print(f"  {PASS}  player={player}: encode_obs OK, shape={obs_arr.shape}")
    except KeyError as e:
        failures.append(f"encoder missing obs key: {e}")
        print(f"  {FAIL}  KeyError {e} — obs-dict-builder out of sync with encoder.encode_obs")
    except Exception as e:
        failures.append(f"encoder contract check crashed: {type(e).__name__}: {e}")
        print(f"  {FAIL}  unexpected: {type(e).__name__}: {e}")
        traceback.print_exc()

    return not failures, failures


def check_yaml_opponent_consistency() -> tuple[bool, list[str]]:
    """If YAML's training_opponent.name is `random_legal` (bootstrap mode),
    `opponent_pool_mode` and `leaderboard_bias` MUST also be off — otherwise
    the trainer downloads the archive anyway and crashes on cross-era weights.
    """
    _hdr("CHECK 4 — Karp YAML opponent-config consistency")
    failures: list[str] = []

    from cli.loop_config import load
    cfg = load()
    opp_name = (cfg.training_opponent or {}).get("name", "")
    bh = cfg.baseline_hyperparams or {}
    opm = bh.get("opponent_pool_mode", "")
    lb  = bh.get("leaderboard_bias", 0.0)

    print(f"  training_opponent.name = {opp_name!r}")
    print(f"  opponent_pool_mode     = {opm!r}")
    print(f"  leaderboard_bias       = {lb}")

    if opp_name == "random_legal":
        if opm:
            failures.append(
                f"YAML inconsistency: training_opponent=random_legal but "
                f"opponent_pool_mode={opm!r} (should be empty during bootstrap)"
            )
        if float(lb) > 0:
            failures.append(
                f"YAML inconsistency: training_opponent=random_legal but "
                f"leaderboard_bias={lb} (should be 0.0 during bootstrap)"
            )

    if failures:
        for f in failures:
            print(f"  {FAIL}  {f}")
    else:
        print(f"  {PASS}  no archive-download paths active given training opponent")

    return not failures, failures


def check_smoke_match() -> tuple[bool, list[str]]:
    """One 4-game match between the current era's most-recent champion and
    `random_legal`. Fast, end-to-end. Catches anything the unit checks miss.
    """
    _hdr("CHECK 5 — Smoke match (4 games vs random_legal)")
    failures: list[str] = []

    try:
        from workers.bench_eval import _most_recent_champion
        from scripts import tournament

        champ = _most_recent_champion()
        if not champ:
            print(f"  {SKIP}  no champion in current era — nothing to smoke-test")
            return True, []

        run_id = str(champ["source_run_id"])
        print(f"  champion: {champ['label']} (era={champ['arch_era']})")
        res = tournament.run_match(
            p1=run_id, p2="random_legal",
            games=4, level="random_close_4_5", seed=99, verbose=False,
        )
        rate = res["p1_wins"] / max(res["total"], 1)
        print(f"  {PASS}  rate vs random_legal = {rate:.1%} ({res['p1_wins']}/{res['total']})")
    except Exception as e:
        failures.append(f"smoke match crashed: {type(e).__name__}: {e}")
        print(f"  {FAIL}  crashed: {type(e).__name__}: {e}")
        traceback.print_exc()

    return not failures, failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Karpathy-loop preflight checks.")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the 4-game smoke match")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("KARP PREFLIGHT — pre-deploy invariant checks")
    print("=" * 60)

    checks = [
        check_yaml_opponent_consistency,
        check_archive_era_filter,
        check_encoder_obs_dict_contract,
        check_champion_weight_loadability,
    ]
    if not args.skip_smoke:
        checks.append(check_smoke_match)

    all_failures: list[str] = []
    for check in checks:
        try:
            ok, fails = check()
        except Exception as e:
            print(f"  {FAIL}  check raised: {type(e).__name__}: {e}")
            traceback.print_exc()
            all_failures.append(f"{check.__name__}: {type(e).__name__}: {e}")
            continue
        all_failures.extend(fails)

    print("\n" + "=" * 60)
    if all_failures:
        print(f"{FAIL}  PREFLIGHT FAILED — {len(all_failures)} issue(s)")
        for f in all_failures:
            print(f"   • {f}")
        print("=" * 60)
        return 1
    else:
        print(f"{PASS}  PREFLIGHT PASSED")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
