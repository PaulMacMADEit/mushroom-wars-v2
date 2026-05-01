"""Smoke test for v12 architecture.

Validates that:
  1. sim config has the v12 shape constants.
  2. v12 encoder produces (OBS_DIM,) and (B, OBS_DIM) for batch.
  3. v12 net forward pass works end-to-end (forward_body, value, source_logits, cond_logits).
  4. PPOAgent.act_batch produces a legal action.
  5. A real env step runs.

Run: .venv/bin/python scripts/smoke_v12.py
"""

from __future__ import annotations

import sys
import numpy as np
import torch

from sim import config as C
from sim.actions import ACTION_SPACE_SIZE, NOOP_INDEX, compute_mask
from sim.envs.mushroom_env import MushroomEnv
from training.encoder import (
    BUILDING_FEATS,
    GLOBAL_FEATS,
    GROUP_FEATS,
    N_BUILDINGS,
    N_GROUPS,
    OBS_DIM,
    encode_obs,
)
from training.net import (
    ActorCritic, D_MODEL, NUM_TYPES, NUM_TYPE_CHOICES, NUM_SRC, NUM_TGT,
)


def _check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not ok:
        sys.exit(1)


def main():
    print("=== sim config ===")
    _check("MAX_BUILDING_SLOTS == 8", C.MAX_BUILDING_SLOTS == 8, str(C.MAX_BUILDING_SLOTS))
    _check("MAX_UNIT_GROUP_SLOTS == 4", C.MAX_UNIT_GROUP_SLOTS == 4, str(C.MAX_UNIT_GROUP_SLOTS))
    _check("SEND_PERCENTAGES == (50, 100)", C.SEND_PERCENTAGES == (50, 100), str(C.SEND_PERCENTAGES))
    _check("ACTION_SPACE_SIZE == 129", ACTION_SPACE_SIZE == 129, str(ACTION_SPACE_SIZE))

    print("\n=== encoder shape ===")
    _check("N_BUILDINGS == 8", N_BUILDINGS == 8)
    _check("N_GROUPS == 4", N_GROUPS == 4)
    _check("BUILDING_FEATS == 11", BUILDING_FEATS == 11)
    _check("GROUP_FEATS == 6", GROUP_FEATS == 6)
    _check("GLOBAL_FEATS == 80", GLOBAL_FEATS == 80)
    expected_obs = 80 + 8 * 11 + 4 * 6
    _check(f"OBS_DIM == {expected_obs}", OBS_DIM == expected_obs, str(OBS_DIM))

    print("\n=== env reset + encode ===")
    env = MushroomEnv(level_name="random_close_4_8", seed=0)
    obs, info = env.reset(seed=0)
    enc = encode_obs(obs)
    _check("encode_obs shape", enc.shape == (OBS_DIM,), str(enc.shape))
    _check("encode_obs dtype float32", enc.dtype == np.float32, str(enc.dtype))
    _check("encode_obs no NaN", not np.any(np.isnan(enc)))

    mask = compute_mask(env.state, C.OWNER_P1)
    _check("mask shape == ACTION_SPACE_SIZE", mask.shape == (ACTION_SPACE_SIZE,))
    _check("noop always legal", bool(mask[NOOP_INDEX]))

    print("\n=== net constants ===")
    _check("NUM_TYPES == 2", NUM_TYPES == 2, str(NUM_TYPES))
    _check("NUM_TYPE_CHOICES == 3", NUM_TYPE_CHOICES == 3, str(NUM_TYPE_CHOICES))
    _check("NUM_SRC == 8", NUM_SRC == 8)
    _check("NUM_TGT == 8", NUM_TGT == 8)

    print("\n=== net forward ===")
    net = ActorCritic()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  total params: {n_params:,}")
    _check("net params > 100k", n_params > 100_000)
    _check("net params < 5M",   n_params < 5_000_000)

    obs_t = torch.as_tensor(enc[None, :], dtype=torch.float32)
    body = net.forward_body(obs_t)
    tokens, kpm = body
    _check("body[tokens] shape (1, 13, d_model)",
           tokens.shape == (1, 1 + N_BUILDINGS + N_GROUPS, D_MODEL),
           str(tokens.shape))
    _check("body[kpm] shape (1, 13)",
           kpm.shape == (1, 1 + N_BUILDINGS + N_GROUPS),
           str(kpm.shape))

    v = net.value(body)
    _check("value shape (1,)", v.shape == (1,), str(v.shape))

    src_logits = net.source_logits(body)
    _check("source_logits shape (1, NUM_SRC)",
           src_logits.shape == (1, NUM_SRC), str(src_logits.shape))

    src_idx = torch.as_tensor([0], dtype=torch.long)
    type_logits, tgt_logits = net.cond_logits(body, src_idx)
    _check("type_logits shape (1, NUM_TYPE_CHOICES)",
           type_logits.shape == (1, NUM_TYPE_CHOICES), str(type_logits.shape))
    _check("tgt_logits shape (1, NUM_TGT)",
           tgt_logits.shape == (1, NUM_TGT), str(tgt_logits.shape))

    print("\n=== agent.act_batch (single env) ===")
    from training.agent import PPOAgent
    agent = PPOAgent(net, device="cpu")
    obs_b = enc[None, :]
    mask_b = mask[None, :]
    action, src, type_, tgt, logp, value = agent.act_batch(obs_b, mask_b)
    _check("action shape (1,)", action.shape == (1,), str(action.shape))
    _check("action in [0, ACTION_SPACE_SIZE)",
           0 <= int(action[0]) < ACTION_SPACE_SIZE,
           str(int(action[0])))
    _check("action is legal under mask", bool(mask[int(action[0])]),
           f"act={int(action[0])}")

    print("\n=== env step with sampled action ===")
    obs2, rew, term, trunc, info = env.step(int(action[0]))
    _check("step returned obs2", obs2 is not None)
    _check("reward is float", isinstance(rew, (int, float, np.floating)))

    print("\n=== JAX encoder import ===")
    try:
        from training.encoder_jax import encode_obs_batched_jit, OBS_DIM as OBS_DIM_JAX
        _check("OBS_DIM matches in JAX module", OBS_DIM_JAX == OBS_DIM)
        print("  (JAX encoder import OK; full forward needs StateJax setup)")
    except Exception as e:
        print(f"[WARN] JAX encoder import failed: {e}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
