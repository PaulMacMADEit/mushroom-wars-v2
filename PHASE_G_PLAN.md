# Phase G — close the GPU SM% gap

**Goal:** lift training-time GPU SM mean from ~3% to ≥20% by removing the host-side sync points still pinning the fused rollout. Keep PPO unchanged; do not move the policy net off torch.

**Status at plan time (2026-04-25):**
- Fused rollout (`training/fused_rollout.py`) is the default path on PaulLinux.
- Phase E profile (`docs/bench/phase_e_paullinux.txt`) confirms the system is **kernel-launch-bound**, not compute-bound. GPU SM mean = 2.8% during a 100s training run; max = 26%.
- Per-rollout-step phase attribution at K=8, n_envs=1024:
  - `step_chunk` 87.4% — JAX vmap kernel + H/D transfers
  - `pack_actions` 7.9% — host numpy: `_decode_into_slot` + `random_legal_opponent_batched` + interleave
  - `encode_mask` 3.2% — `encode_obs_batched_jit` (JAX) + 2× `compute_mask_batched` (numpy on host)
  - `act_batch` 1.4% — torch policy forward via DLPack
- Two of those four phases (`pack_actions`, `encode_mask`) include host-numpy work that pins the kernel-launch queue. After moving them on-device the rollout chain becomes act_batch (torch via DLPack) → step_chunk (JAX) → encode/mask (JAX), and JAX's natural async dispatch should keep the kernel queue ahead of Python.

**What this document is:** a continuation of `FUSED_ROLLOUT_PLAN.md`. Same constraints (commit-per-phase, ruff/pyright clean, parity-tested, `PaulMacMADEit <paul@madeit.tech>` author, never `Co-Authored-By: Claude`, pause between phases).

---

## 1. Success criteria

Done means **all** of the following hold on `main` after the final phase:

1. `pytest tests/ -q` → all 192+ tests passing on both backends, no regression. New tests under `tests/test_actions_jax.py` cover the JAX mask port.
2. **GPU SM mean ≥ 20% during a 5-minute training run** on PaulLinux RTX 3070 (vs current 2.8%). Measured via `nvidia-smi dmon -s u`. ≥30% would be the §1 criterion of the original FUSED_ROLLOUT_PLAN — that's the stretch goal.
3. **Per-rollout-step `pack_actions` + `encode_mask` together drop below 2% of step time** in the post-G2 profile (vs 11.1% combined today).
4. **Win-rate trajectory matches pre-Phase-G under matched seed within ±5pp** at the 5-minute mark. PPO is unchanged; if win-rates diverge under the same seed, we have a parity bug.
5. Rollback path: each sub-phase is one commit, revert to the prior commit returns to the previous state. No flag plumbing — this is correctness work, not behaviour-flag work.

---

## 2. Non-goals

1. **Algorithm change.** Stays PPO.
2. **Net change.** ActorCritic stays as-is.
3. **n_envs / rollout_steps tuning.** b1 already proved 1024 / 64 is right on the 3070; don't re-sweep.
4. **Self-play under fused.** G4 covers the neural-opponent-inside-JIT path but only as a follow-up gated on b3+ exposing a real bottleneck there.
5. **Mac.** Phase G is a Linux/CUDA win. Mac dev keeps working but won't see the gain.

---

## 3. Architectural decisions

### 3.1 Where the JAX mask lives

**Decision:** new file `sim/actions_jax.py`. Function `compute_mask_batched_jax(state: StateJax, player: int) -> jnp.ndarray` takes the JAX state pytree directly and returns `(N, ACTION_SPACE_SIZE) bool` on device. Numpy `compute_mask_batched` (sim/actions.py:152) stays as the reference oracle for the parity test.

**Why:** matches the `encoder_jax.py` pattern. Lets `_encode_and_masks` in fused_rollout.py drop the four `np.asarray(state.X)` host-pulls plus the two numpy mask calls, replacing them with one JIT'd device call.

### 3.2 Where the JAX action-pack lives

**Decision:** new function `pack_action_batch_jax` in `sim/actions_jax.py`. Takes:
- `p1_actions_flat: jnp.ndarray (N,)` — already on device from agent.act_batch's DLPack output, or zero-copy back from torch
- `p2_mask_dev: jnp.ndarray (N, A)` — produced by G1 on device
- `key: jax.random.PRNGKey` — for legal-action sampling
- `opponent_name: str` — static argument; controls "random_legal" vs "noop" branch

Returns the `(N, 2, 2) int64` action batch that `step_chunk` consumes.

`random_legal_opponent_batched` (sim/envs/opponents.py:47) is the numpy oracle. JAX version uses `jax.random.categorical` over the boolean mask (interpreted as logits with -inf for False), tested for distribution parity over many seeds.

**Why:** with G1+G2 both on device, the entire t-step body of the fused rollout is one device-side dataflow chain. Python only does loop bookkeeping; the kernel queue stays ahead.

### 3.3 What stays on host

- Per-step buffer writes (`src_buf[t]`, `rew_buf[t]`, etc.) — these are deferred to the end of the rollout and dumped in batched form, no per-step host work.
- Episode bookkeeping (`ep_return`, `ep_length`, `completed_episodes`) — uses `result["dones"]` which is already a host pull. This is a sync point but tiny and unavoidable without restructuring episode tracking; leave alone.
- `obs_norm.update(...)` — already lifted to once-per-rollout in Phase E.

### 3.4 Async dispatch

JAX is async by default. The reason it currently doesn't keep the kernel queue full is that `result["dones"]` and the host masks force a sync after every `step_chunk`. After G1+G2, the only forced sync per step is `result["dones"]` for episode tracking. If profile after G2 still shows kernel-launch idle time, G3 explicitly batches the per-step `dones` pull (one combined transfer per rollout) instead of per-step. Don't pre-implement this — measure first.

### 3.5 RNG semantics

Numpy `random_legal_opponent_batched` uses `np.random.Generator`. JAX uses `jax.random.PRNGKey`. These produce different sequences even with matched seeds. The parity test for G2 is **distributional**, not byte-identical:

- Sample N=10000 envs × 50 mask configurations under both RNGs
- Assert empirical action distribution matches within KL ≤ 0.01 per mask
- Assert no illegal action ever sampled (mask compliance is hard parity)

Win-rate trajectory parity (criterion 4) is what catches behavioural drift in practice. Different RNG, same statistics, same training outcome — that's the bar.

---

## 4. Sub-phases

### Phase G1 — `compute_mask_batched_jax`

**Scope:**
- Add `sim/actions_jax.py` with `compute_mask_batched_jax(state, player) -> jnp.ndarray`. JIT'd, static `player`.
- Add `tests/test_actions_jax.py` with parity test against numpy `compute_mask_batched`: 100 random states × 2 players, max abs diff = 0 (mask is bool, parity must be exact).
- Add `_encode_and_masks_jax` in `fused_rollout.py` that calls the new JAX path and keeps both masks on device. Keep the numpy `_encode_and_masks` for non-fused callers (none currently — but cheap to keep).
- Replace the use site in `fused_rollout.py:188` with the new function. P1 mask still needed by torch agent (act_batch reads it as a torch bool tensor) — DLPack across.
- P2 mask stays on device for G2.

**DoD:**
- All existing tests still pass.
- New parity test passes.
- Mac smoke run (5 updates, n_envs=64) under fused completes without regression.

**Verification:**
```bash
pytest tests/test_actions_jax.py tests/test_fused_rollout.py -q
SIM_BACKEND=jax python scripts/smoke_train.py --n-envs 64 --updates 5
```

**Commit:** `Phase G1: JAX-batched compute_mask + use in fused rollout`

---

### Phase G2 — `pack_action_batch_jax`

**Scope:**
- Add `pack_action_batch_jax(p1_actions, p2_mask_dev, key, opponent_name) -> jnp.ndarray` in `sim/actions_jax.py`.
- For `opponent_name="noop"`: P2 actions = NOOP_INDEX broadcast.
- For `opponent_name="random_legal"`: sample per-env from `p2_mask_dev` using `jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))`.
- Add `_decode_into_slot_jax` (the action-decoder mirror) to convert flat action indices back to (slot_src, slot_tgt, type) triples on device.
- Add `tests/test_actions_jax.py::test_pack_distribution`: KL ≤ 0.01 to numpy, illegal-action rate = 0.
- Replace `_pack_action_batch_with_p2_mask` call in `fused_rollout.py:174` with the JAX version. Thread a `jax.random.PRNGKey` through the rollout (split per step).
- Neural-opponent path (`_pack_action_batch_neural`) stays unchanged — that's G4.

**DoD:**
- All existing tests pass.
- Distribution parity test passes.
- Mac smoke run completes; loss values within noise of pre-G2 under matched seed (different RNG, so not byte-identical).
- 50-update training run on Mac shows finite losses, win-rate progressing.

**Verification:**
```bash
pytest tests/test_actions_jax.py tests/test_fused_rollout.py -q
SIM_BACKEND=jax python scripts/smoke_train.py --n-envs 64 --updates 50
```

**Commit:** `Phase G2: JAX-batched action pack + random_legal opponent`

---

### Phase G3 — measure + decide

**Scope:**
- Pull G1+G2 to PaulLinux.
- Run the Phase E profile script under the new code path. Record into `docs/bench/phase_g_paullinux.txt`:
  - K-sweep (per-tick / K=1 / K=4 / K=8 / K=16) ms/tick.
  - 5-minute training run with `nvidia-smi dmon -s u` running in parallel. Report SM mean / max / ≥20% / ≥30% sample fractions.
  - Per-rollout-step phase attribution. Hard target: pack_actions + encode_mask combined < 2%.
  - Win-rate at 5 minutes vs pre-Phase-G under matched seed.
- If SM ≥ 20% and win-rate within band: phase G is done. Update FUSED_ROLLOUT_PLAN.md §1 criterion 4 row to ✅. Skip G4 unless self-play work later demands it.
- If SM < 20% but pack_actions+encode_mask < 2%: the remaining sync is `result["dones"]`. Implement G3a: batched dones pull. Re-measure.
- If pack_actions or encode_mask still > 2% individually: parity test passed but performance regressed somehow. Investigate before proceeding.

**DoD:**
- `docs/bench/phase_g_paullinux.txt` committed with full numbers.
- Decision recorded in this plan: G3 done at success criterion 2, OR G3a follow-up scoped.

**Verification:** the bench file itself.

**Commit:** `Phase G3: PaulLinux RTX 3070 perf profile under JAX masks+pack`

---

### Phase G4 — opponent-inside-JIT *(gated, optional)*

**Scope:** only run if a future training session (b3+ self-play, or §13 mixed-level cycling) shows the neural-opponent path is throughput-bound and matters more than current `random_legal` baselining.

**Approach:** export the opponent torch policy to JAX (small enough — 170k params) at opponent-load time, or DLPack it into the rollout per-step. Keep numpy fallback.

**DoD when invoked:** TBD — write a sub-plan in this file when the trigger fires.

**Commit:** `Phase G4: opponent-inside-JIT for neural opponents` (when applicable).

---

## 5. Risks

| # | risk | likelihood | impact | mitigation |
|---|---|---|---|---|
| 1 | JAX random_legal sampling distribution drifts vs numpy | medium | training behaviour shift | Parity test on KL + sample-many. Win-rate trajectory check at G3. |
| 2 | DLPack of P1 mask back to torch is slow (small tensor, lots of launches) | low | partial regression | Bench at G1; if P1-mask DLPack is heavier than the np→torch path, keep P1 mask on host and only put P2 on device. P2 is the one G2 needs on device. |
| 3 | jax.random.PRNGKey threading breaks reproducibility under cfg.seed | medium | non-determinism | Thread key via `jax.random.split` from `cfg.seed`, hash test in G2. |
| 4 | Mask construction loop unrolling in JAX explodes JIT time | low | first-update slow | C.SEND_PERCENTAGES is small (~5 entries). lax.scan over types if it bloats. |
| 5 | After G1+G2, kernel-launch ceiling moves to torch policy forward | medium | gain caps below target | act_batch is currently 1.4%; even if it 5×s in relative terms, total still ≤ 7%. Acceptable. |
| 6 | n_envs 1024 / K 8 turns out to be the wrong sweet spot post-Phase-G | medium | re-tune needed | Plan §2 explicitly defers re-sweeping. If G3 profile suggests a shift, document and create a separate experiment plan; don't fold into Phase G. |

---

## 6. Constraints (inherited from FUSED_ROLLOUT_PLAN.md §8)

1. Code style: `CODING_GUIDE.md` to the letter. Ruff + pyright clean every commit.
2. One sub-phase per commit. No bundling. No drive-by refactors.
3. Commit as `PaulMacMADEit <paul@madeit.tech>`. **Never** `Co-Authored-By: Claude`.
4. Test before commit. Push after each phase.
5. Secrets in `~/Documents/Sync/secrets/.env.master`, never in repo.
6. PaulLinux access via Tailscale SSH for G3 and any further GPU-side measurement.
7. Pause for approval between phases unless explicitly told otherwise.

---

## 7. Executable checklist

- [ ] **Phase G1** — `compute_mask_batched_jax` + parity test + use in fused rollout.
- [ ] **Phase G2** — `pack_action_batch_jax` + distribution parity test + use in fused rollout.
- [ ] **Phase G3** — PaulLinux profile + decide on G3a / G4.
- [ ] **Phase G4** *(gated)* — opponent-inside-JIT for neural opponents.

---

## 8. Reference material

- `FUSED_ROLLOUT_PLAN.md` — predecessor plan; §3.2 explains why fused rollout exists at all.
- `docs/bench/phase_e_paullinux.txt` — current bottleneck profile and the lever list this plan implements.
- `training/encoder_jax.py` — pattern to mirror for `actions_jax.py`.
- `sim/actions.py:152` — `compute_mask_batched` numpy oracle.
- `sim/envs/opponents.py:47` — `random_legal_opponent_batched` numpy oracle.
- `training/fused_rollout.py:165-218` — rollout inner loop where G1+G2 use sites land.
- JAX random docs: https://jax.readthedocs.io/en/latest/jax.random.html
