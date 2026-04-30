"""Diagnose the v10 train/eval obs mismatch.

The training path keeps the state in JAX and calls `encode_obs_batched_jit(state)`
directly. The eval path goes through `MushroomEnv.step()` → `obs_dict` →
`encode_obs(obs_dict)`. The parity test only covers NOOP actions, so it never
exercises the v10 features that fire under real play (arrivals_*,
prev_buildings_owner, hostile_landed, friendly_landed, ownership_changed,
last_actions_*).

This script drives the SAME deterministic game through both paths in lockstep
and compares the encoded obs vector at every decision step. It also prints
per-feature-block diffs so we can see which slice of the 1008-d vector differs.

Run:
  python scripts/diagnose_train_eval_gap.py
  python scripts/diagnose_train_eval_gap.py --level random_close_4_5 --steps 60
  python scripts/diagnose_train_eval_gap.py --seed 7 --print-every 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force CPU before any torch / JAX import — keeps the diagnostic light and
# eliminates GPU PRNG variance as a confound.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _random_legal_action_idx(state, player: int, rng) -> int:
    """Same logic as sim/envs/opponents.random_legal_opponent: uniform over
    legal action indices, with NOOP fallback when the mask is all-zero (which
    shouldn't happen mid-game but might at terminal-pending state)."""
    from sim.actions import compute_mask
    import numpy as np

    mask = compute_mask(state, player)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        # NOOP is always legal by construction; this branch is paranoia.
        from sim.actions import NOOP_INDEX
        return int(NOOP_INDEX)
    return int(rng.choice(legal))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", default="random_close_4_5")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--steps", type=int, default=40,
                    help="Decision steps (each = K=2 env ticks).")
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    import numpy as np
    import jax
    import jax.numpy as jnp

    from sim.envs.mushroom_env import MushroomEnv
    from sim.envs.opponents import random_legal_opponent
    from sim.actions import decode
    from sim.state_jax import from_numpy_state
    from training.encoder import encode_obs
    from training.encoder_jax import encode_obs_batched_jit

    print(f"[diag] level={args.level} seed={args.seed} steps={args.steps}")
    print(f"[diag] tolerance: |np - jax| < {args.tol:.0e}\n")

    # P1 driven by random_legal (same RNG seeded from `args.seed`).
    # P2 also random_legal so arrivals_p2 and ownership_changed actually fire.
    rng_p1 = np.random.default_rng(args.seed * 2 + 1)
    p2_opp = random_legal_opponent  # opponent callable; uses its own RNG

    env = MushroomEnv(level_name=args.level, opponent=p2_opp, seed=args.seed)
    obs_dict, _ = env.reset(seed=args.seed)

    # Reset-time parity check (all-zero v10 fields, fresh state).
    np_vec_0 = encode_obs(obs_dict)
    sj_0 = from_numpy_state(env.state)
    sj_0_b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], sj_0)
    jx_vec_0 = np.asarray(encode_obs_batched_jit(sj_0_b))[0]

    diff_0 = float(np.abs(np_vec_0 - jx_vec_0).max())
    print(f"[reset] max |np - jax| = {diff_0:.2e}  "
          f"{'PASS' if diff_0 < args.tol else 'FAIL'}")

    # Step the env, recompute parity each step.
    max_diff_overall = diff_0
    fail_step = None
    fail_diff = 0.0
    fail_feature_breakdown: list[tuple[str, slice, float]] | None = None

    # Feature-block boundaries match training/encoder.py:OBS_DIM=1008.
    # Globals: 80 (10 base + 4 prod + 4 topo + 2 delta + 60 history)
    # Per-bldg: 32×20 = 640
    # Per-group: 32×9 = 288
    G       = 80      # globals end
    B_END   = G + 640 # buildings end
    blocks: list[tuple[str, slice]] = [
        ("globals[0:10] base",          slice(0, 10)),
        ("globals[10:14] prod",         slice(10, 14)),
        ("globals[14:18] topo",         slice(14, 18)),
        ("globals[18:20] reward_delta", slice(18, 20)),
        ("globals[20:80] action_hist",  slice(20, 80)),
        ("per-building (32x20)",        slice(G, B_END)),
        ("per-group (32x9)",            slice(B_END, 1008)),
    ]

    for step in range(args.steps):
        if env.state.phase != 0:  # PHASE_PLAYING==0
            print(f"[step {step}] env terminated (phase={int(env.state.phase)}); stopping early")
            break

        a_idx = _random_legal_action_idx(env.state, 1, rng_p1)  # OWNER_P1=1
        obs_dict, _r, term, trunc, info = env.step(a_idx)

        np_vec = encode_obs(obs_dict)
        sj = from_numpy_state(env.state)
        sj_b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], sj)
        jx_vec = np.asarray(encode_obs_batched_jit(sj_b))[0]

        diff_vec = np.abs(np_vec - jx_vec)
        max_diff = float(diff_vec.max())
        if max_diff > max_diff_overall:
            max_diff_overall = max_diff

        # Per-block breakdown for this step.
        block_diffs = [(name, sl, float(diff_vec[sl].max())) for (name, sl) in blocks]

        ok = max_diff < args.tol
        if (step % args.print_every == 0) or not ok:
            tag = "PASS" if ok else "FAIL"
            print(f"[step {step:3d}] act={a_idx:5d} "
                  f"phase={int(env.state.phase)} tick={int(env.state.tick):3d}  "
                  f"max diff = {max_diff:.2e}  {tag}")
            if not ok:
                # Where exactly does it diverge?
                for (name, sl, d) in block_diffs:
                    if d >= args.tol:
                        # Find the worst offending index inside the block.
                        sub = diff_vec[sl]
                        worst_local = int(np.argmax(sub))
                        worst_global = sl.start + worst_local
                        np_val = float(np_vec[worst_global])
                        jx_val = float(jx_vec[worst_global])
                        print(f"           {name:32s}  block_max={d:.2e}  "
                              f"idx={worst_global} np={np_val:+.4f} jax={jx_val:+.4f}")

        if not ok and fail_step is None:
            fail_step = step
            fail_diff = max_diff
            fail_feature_breakdown = block_diffs

        if term or trunc:
            print(f"[step {step}] env returned terminated={term} truncated={trunc}; stopping")
            break

    print()
    print(f"[summary] max diff over all steps: {max_diff_overall:.2e}")
    if fail_step is None:
        print(f"[summary] PASS — encoder parity holds under real gameplay.")
    else:
        print(f"[summary] FAIL — first divergence at step {fail_step} "
              f"(max diff {fail_diff:.2e}).")
        print(f"[summary] Block-level breakdown at first failure:")
        for (name, sl, d) in (fail_feature_breakdown or []):
            tag = "✗" if d >= args.tol else "·"
            print(f"           {tag} {name:32s}  block_max={d:.2e}")


if __name__ == "__main__":
    main()
