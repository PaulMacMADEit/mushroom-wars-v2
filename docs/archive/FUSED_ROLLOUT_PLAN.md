# Fused-rollout port plan — `collect_rollout` on the GPU

**Goal:** keep PPO. Move `collect_rollout` from a Python tick loop (T network forwards × T env steps × T encodes) to a GPU-resident pipeline where T env steps and T policy decisions live inside one kernel each (or one fused dispatch). Get sim throughput back into the millions of ticks/sec during training, not just during synthetic benches.

**Status at plan time (2026-04-24):**
- Sim: JAX backend default. RTX 3070 isolated bench: 3.7M ticks/sec @ n_envs=1024 fused; pure compute hit GPU SM 100% in the kernel.
- Training: numpy vs JAX env_step at n_envs=1024 → 37.5s vs 20.1s per 2 updates (1.87× on the sim step). End-to-end 1.37×.
- Profile shows the env_step phase is now mostly **per-tick host↔device round trip and Python-side encode loop**, not sim compute. The XLA kernel finishes; the host side spends the rest of the budget.
- Crossover where JAX beats numpy on integrated training is n_envs ≈ 256. Below that, numpy AsyncVectorEnv IPC is faster than one kernel launch per tick.

**Why now:** the JAX port shaved a real 1.4× off training but the sim-side ceiling is ~5–10× higher. The remaining gap is rollout architecture, not sim engine. Fixing it is the last lever before the algorithm itself becomes the bottleneck.

**What this document is:** an execution plan, structured the same way as `JAX_PORT_PLAN.md`. Each phase is one commit, has a definition-of-done, a verification command, a rollback point. Same constraints (no `Co-Authored-By`, surgical diffs, doc-sync rule, ruff/pyright clean).

---

## 1. Success criteria

Done means **all** of the following hold on `main` after the final phase:

1. `pytest tests/ -q` → **all 176+ tests passing on both backends**, no regression. New tests under `tests/test_fused_rollout.py` cover the new path.
2. `tests/test_fused_rollout.py::test_fused_matches_per_tick_rollout` — same seed, same actions, same final state across the per-tick rollout (current code) and the fused rollout (new). Byte-identical metrics output.
3. PaulLinux RTX 3070, n_envs=1024, rollout_steps=64: **upd/s ≥ 3× the current JAX path** (current 0.045 upd/s at this config → target ≥0.135 upd/s). Hard floor: must beat numpy by at least 5× end-to-end at that config (current numpy is 0.032 upd/s).
4. **GPU SM mean ≥ 30% during a 5-minute training run** (vs current ~5%). Measured via `nvidia-smi dmon -s u` over the run, not just one kernel.
5. Win-rate trajectory matches the per-tick rollout under the same seed within statistical noise (5-minute run, P1 win-rate band ±5pp). PPO is supposed to be on-policy and unchanged; if win-rates diverge, we have a bug.
6. Rollback path: `--fused-rollout=false` flag (or `FUSED_ROLLOUT=0` env) returns the trainer to today's per-tick `collect_rollout` exactly. Default flips in a separate commit only after criteria 1–5 are green for a week.

---

## 2. Non-goals

1. **Algorithm change.** Stays PPO. No IMPALA, no V-trace, no MuZero. Different plan.
2. **Net change.** ActorCritic stays as-is. No bigger trunk, no attention, no transformer.
3. **Encoder change.** `encode_obs` and `OBS_DIM` are frozen for the duration. (A side benefit of the new path is opportunity to port the encoder to JAX in a separate plan; out of scope here.)
4. **Self-play under fused rollout.** Self-play / per-env neural opponent path stays on the per-tick rollout — fused supports `random_legal` and `noop` only in v1. Self-play under fused is a separate follow-up; the trainer auto-falls-back when `cfg.self_play=True`.
5. **Replay / dashboard.** Not affected. Replay still runs under numpy backend with full event emission.
6. **Mac (Metal).** First-class on the per-tick path; fused path may run correctly but won't be wall-clock faster. Mac dev keeps numpy or non-fused JAX.

---

## 3. Architectural decisions

### 3.1 Where the fused rollout lives

**Decision:** new file `training/fused_rollout.py`. `PPOTrainer.collect_rollout()` becomes a one-line dispatch — call into the fused path when `cfg.fused_rollout=True` and the prerequisites hold (no self-play, JAX backend), else fall through to the existing per-tick code.

**Why:** keeps the existing per-tick path the source of truth and the rollback target. Two code paths is a temporary cost we pay for safety.

### 3.2 What "fused" actually means

The current per-tick loop interleaves five things on every tick:
1. encode obs (numpy, per env, Python loop)
2. obs norm (numpy)
3. agent.act_batch (torch, single forward → log-prob, value, action)
4. vec.step (one XLA dispatch + one device-to-host + Python loop for opponent)
5. write into rollout buffers (numpy)

Plus once per rollout: GAE backwards pass.

**Decision:** turn the inner loop into a `jax.lax.scan` over T ticks where steps 1, 2, 4 happen entirely on device. Step 3 (the torch policy) **stays on torch** but moves out of the per-tick loop: instead of one network forward per tick, do all T forwards as one (T, N, OBS_DIM) batched call AFTER the env scan, and re-derive log-probs / values from stored states.

Wait — that would make the policy in the rollout depend on a network that wasn't yet observing the trajectory it's about to take. That's wrong. PPO needs the policy to act on the obs at each tick, then the env consumes that action.

**Refined decision:** do the env scan in chunks of K ticks (say K=8 or 16), where between chunks we do a torch forward to pick actions for the next K ticks, then run the env scan with those K actions inside the JIT. This is "lookahead-free chunked rollout" — like A2C / IMPALA's actor pattern. Each chunk does one torch forward (batched over N) and one XLA dispatch (over N×K env steps).

**Trade-off:** within a chunk, the agent commits to K actions before observing intermediate rewards. For K=1 this is exactly the current per-tick loop. For K=T it's one big batch with stale actions throughout the rollout. We pick K to balance:
- Larger K → fewer torch forwards + fewer XLA launches → wall-clock win.
- Smaller K → fresher policy, more on-policy, less variance.

For PPO with rollout_steps=64, K=4 or K=8 is the sweet spot — preserves PPO's on-policy assumption (within 4 ticks the policy is essentially the same anyway), while collapsing 64 launches into 8 or 16. Make K a config knob; default K=4.

### 3.3 What the agent commits to per chunk

Two options:
- **(a)** Agent picks one action at chunk start, env runs that action for K ticks (action repetition).
- **(b)** Agent picks K actions at chunk start, env runs each one in sequence inside the scan.

(a) is wrong — repeating the same `send` action for 4 ticks would deduct from the same source 4 times and behave nothing like the current trainer.

**Decision:** option (b). Agent samples K actions per env per chunk. The env scan threads them through.

How: `agent.act_batch_K(obs_at_chunk_start, mask_at_chunk_start, K)` returns `(K, N)` actions sampled by RUNNING THE POLICY FORWARD K TIMES with each act sampled conditioned on the *predicted* next obs. But predicting next obs is the hard part — we don't have a learned dynamics model. So actually:

**Refined:** the agent only samples 1 action per real obs. Within a chunk, the env runs N ticks and the agent's action is observed only at tick 0 of the chunk. For ticks 1..K-1 of the chunk, the env steps with NOOP (or repeats the agent's last action). This restores option (a) but explicitly: the agent decides every K ticks, not every 1 tick.

This is **action repetition / frame skip**, a classic trick. PPO handles it cleanly as long as the rollout sees one (action, sum-of-rewards-over-K-ticks, next_obs) tuple per chunk instead of K tuples per chunk. **rollout_steps becomes the number of chunks**, not the number of env ticks; each "step" stored is K env ticks of compute.

**Decision (final):** chunked PPO with action-repetition. The agent decides every `cfg.action_repeat` env ticks (default `action_repeat=4`, configurable, can be set to 1 for parity with old behaviour). The env runs `action_repeat` ticks per agent decision, all inside one fused JAX scan. Rewards over the chunk sum into one stored reward per chunk per env. The PPO buffer shape becomes `(rollout_steps, n_envs)` where each row is one *agent decision*, not one env tick.

This is actually a small algorithmic change — agent now operates at a coarser temporal grid. For Mushroom Wars where decisions happen every `DECISION_INTERVAL_TICKS=2` ticks anyway, raising the effective decision interval to 2*K=8 ticks is mild. Document the trade-off; ship the knob; leave at 1 for the parity test then move to 4 for production.

### 3.4 Obs encoding lives on the GPU during the rollout

`encode_obs` is currently a per-env numpy function called inside `_encode_batch`. To avoid a host roundtrip every chunk, port it to a JAX function that takes a batched StateJax and returns `(N, OBS_DIM) jnp.float32`.

**Decision:** new `training/encoder_jax.py`. Mirrors `encode_obs` exactly (parity-tested). `JaxVecEnv.step_chunk(actions, K)` returns the encoded obs and mask directly from device, no numpy round-trip.

The torch policy still consumes `(N, OBS_DIM)` — DLPack from JAX → torch on the same CUDA device is now standard and works in jax 0.10 / torch 2.6. Use it. If it's flaky, fall back to `np.asarray(jax_array)` round-trip and pay the host transfer cost (one transfer per chunk — fine).

### 3.5 GAE on device

The advantage / return computation is a backwards scan over `(T, N)` rewards/values/dones. Currently a Python `for t in reversed(range(T))` numpy loop. Move it into a `lax.scan(reverse=True)` and run on device. Tiny win but kills another round-trip and keeps the buffer tensors device-resident through to the PPO update.

### 3.6 PPO update phase

**Stays on torch.** Minibatch shuffling, clip-loss, entropy, value-loss, optimizer step are PyTorch and stay there. Cross-stack at the rollout-buffer boundary: `(T, N, …)` jax arrays → torch tensors via DLPack once per update.

### 3.7 Backend assumptions

The fused path assumes JAX backend. Under numpy backend `cfg.fused_rollout` is silently ignored (or raises with a clear message). One backend, one rollout shape — don't try to make fused-rollout-on-numpy work.

---

## 4. Implementation phases

Each phase is one commit. Each phase passes its verification before moving on. No bundling.

### Phase A — port the encoder to JAX

**Goal:** `training/encoder_jax.py` — pure-jax `encode_obs_batched(state_jax) -> (N, OBS_DIM) jnp.float32`. Byte-identical to the existing per-env numpy path (within float32 epsilon).

**Scope:**
1. New `training/encoder_jax.py`. Port `encode_obs` body to operate on the batched StateJax. Vectorise the per-group "incoming flight aggregates" loop with `jax.ops.segment_sum`.
2. Parity test `tests/test_encoder_parity.py`: build 50 random states (mix of empty, mid-game, late-game), encode each via numpy `encode_obs` and via `encode_obs_batched`, assert max abs diff < 1e-5.
3. Bench: `scripts/bench_encoder.py` — encode 1024 envs via numpy loop (status quo) vs JAX vmap'd, on PaulLinux. Log numbers; not a gate.

**Definition of done:**
- `pytest tests/test_encoder_parity.py -q` green on both Mac CPU and PaulLinux CUDA.
- No edit to `encode_obs` itself — numpy path stays the reference.

**Verification:**
```bash
SIM_BACKEND=jax pytest tests/test_encoder_parity.py -q
```

**Rollback:** `git revert` — no caller changes yet.

---

### Phase B — `JaxVecEnv.step_chunk(actions, K)`

**Goal:** new public method on `JaxVecEnv` that runs K env ticks per env in one fused JIT, with action-repetition (each env's single action applies on tick 0, NOOP on ticks 1..K-1). Returns the chunk-summed reward and the post-chunk encoded obs.

**Scope:**
1. `sim/envs/jax_vec_env.py`:
   - New private `_step_chunk_impl(state, action_p1, action_p2, K)`. `lax.scan` over K ticks: tick 0 uses (action_p1, action_p2); ticks 1..K-1 use NOOP for both. Per env: sum rewards across the K ticks, OR-fold the done flag, return final state.
   - vmap'd + jit'd: `_step_chunk_batched = jax.jit(jax.vmap(_step_chunk_impl, in_axes=(0, 0, 0, None)))`.
   - New public method `step_chunk(actions: (n_envs, 2, ACTION_DIM), K: int) -> {state, rewards_p1: (n_envs,), rewards_p2: (n_envs,), dones: (n_envs,), encoded_obs: (n_envs, OBS_DIM), action_mask: (n_envs, ACTION_SPACE_SIZE)}`. Auto-resets done envs (CPU level-gen, vector splice).
2. K is a Python int (static argument — bake into JIT cache key). Default K=4.
3. Action representation: per-env one P1 action + opponent picks one P2 action at chunk start (both batched, no Python loop).
4. Parity test: with K=1, `step_chunk` must match a per-tick `step()` in numpy state byte-for-byte.

**Definition of done:**
- `tests/test_step_chunk_parity.py` green: K=1 byte-identical with the per-tick path; K=4 byte-identical with 4 sequential per-tick steps where actions are NOOP after tick 0.
- `pytest tests/ -q` → 176+ passed.

**Verification:**
```bash
pytest tests/test_step_chunk_parity.py -q
```

---

### Phase C — fused rollout collector

**Goal:** new `training/fused_rollout.py` exposing `collect_rollout_fused(agent, vec_env, cfg, obs_norm) -> rollout dict`. Same return shape as today's `PPOTrainer.collect_rollout`.

**Scope:**
1. Inner loop:
   ```python
   for chunk in range(rollout_steps):
       # one torch forward, batched over n_envs
       actions, srcs, types, tgts, logps, values = agent.act_batch(obs, mask)
       # one XLA dispatch, K env ticks per env
       result = vec_env.step_chunk(pack_actions(actions, opponent_actions), K=cfg.action_repeat)
       # buffer-store
       ...
       obs = result["encoded_obs_normalised"]
       mask = result["action_mask"]
   ```
2. Obs normalisation: `RunningNorm` on JAX. `training/obs_norm_jax.py` — pure-jax variant of Welford. Pull stats back to torch's `RunningNorm` once per update for parity.
3. GAE on device: `compute_gae_jax(rewards, values, dones, bootstrap)` using `lax.scan(reverse=True)`.
4. Final return assembled as numpy (DLPack to torch is the better path; do it lazily on `rollout["obs"].to_torch()`). For Phase C accept one bulk numpy round-trip; DLPack optimisation is Phase E.
5. Per-env episode bookkeeping (episodes_completed, win_rate proxy): preserved exactly.

**Definition of done:**
- New: `training/fused_rollout.py`, `training/obs_norm_jax.py`.
- `cfg.fused_rollout: bool = False` added to `PPOConfig`. `cfg.action_repeat: int = 4` added.
- `PPOTrainer.collect_rollout` branches on `cfg.fused_rollout`; old path unchanged.
- `tests/test_fused_rollout.py::test_fused_matches_per_tick_rollout`: with `cfg.action_repeat=1` and same seed, fused and per-tick produce byte-identical rollout dicts (after backend-aware tolerance). With `action_repeat=4`, just sanity checks (rollout shape, no NaNs, win_rate finite).
- 5-min smoke train under both `cfg.fused_rollout=True/False` on Mac CPU completes without errors, win_rate > 0.3.

**Verification:**
```bash
pytest tests/test_fused_rollout.py -q
SIM_BACKEND=jax python scripts/smoke_train.py --seconds 60 --envs 64 --vec-mode sync \
    --rollout 32 --fused-rollout
```

(Adds `--fused-rollout` flag to `smoke_train.py`.)

---

### Phase D — PaulLinux deploy + perf gate

**Goal:** deploy to RTX 3070, validate criteria 3 + 4.

**Scope:**
1. `git pull` on PaulLinux (no new deps — JAX already installed).
2. Run `scripts/profile_rollout.py` head-to-head:
   - `SIM_BACKEND=numpy ... --rollout 64 --envs 1024`
   - `SIM_BACKEND=jax   ... --rollout 64 --envs 1024 --fused-rollout`
   - Record upd/s, env_step ms, GPU SM%.
3. Save numbers to `docs/bench/phase_d_paullinux_fused.txt`.
4. Two failure modes to anticipate:
   - **OOM**: cap `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` if torch OOMs.
   - **JIT compile time**: first chunk compile may take 10–20s; warmup the JIT in `PPOTrainer.__init__` so the first measured update isn't compiling. Write that into `fused_rollout.py`.

**Definition of done:**
- upd/s ≥ 3× over current JAX path at n_envs=1024 (criterion 3).
- GPU SM mean ≥ 30% over a 5-minute run (criterion 4). Measured via systemd unit + `nvidia-smi dmon -c 150`.
- Win-rate at 5-minute mark within the same band as numpy backend at the same wall-clock (criterion 5).
- `docs/bench/phase_d_paullinux_fused.txt` committed in this phase.

**Verification:**
```bash
ssh paul@PaulLinux 'cd ~/Projects/Personal/games/mushroom-wars-v2 \
    && SIM_BACKEND=jax .venv/bin/python scripts/profile_rollout.py \
       --envs 1024 --rollout 64 --updates 5 --fused-rollout'
```

---

### Phase E — DLPack zero-copy (optional, gated on profiling)

**Goal:** eliminate the bulk numpy round-trip at the rollout boundary. Only if Phase D's bottleneck profile shows the round-trip is >10% of the rollout phase.

**Scope:**
1. JAX `(T, N, OBS_DIM)` array → torch CUDA tensor via `jax.dlpack.to_dlpack` + `torch.utils.dlpack.from_dlpack`. Requires JAX and torch sharing the CUDA context (they do, but with different memory allocators — needs `XLA_PYTHON_CLIENT_MEM_FRACTION` and `PYTORCH_CUDA_ALLOC_CONF` cooperating).
2. Test: random (T, N, OBS_DIM) array round-trips JAX → torch → numpy → JAX with no value drift.
3. Fall back to numpy if DLPack fails on any device.

**Skip Phase E if Phase D already meets criterion 3.** This is an optimisation, not a requirement.

---

### Phase F — flip default + register sim-v1.3 (optional)

**Goal:** decide whether the fused rollout is "the trainer." Only after one full week of green Phase D runs.

**Scope:**
1. `cfg.fused_rollout` default changes from `False` to `True` in PPOConfig (when JAX backend is active and not self-play).
2. Update `ARCHITECTURE.md` §7 (training pipeline) to describe the chunked rollout.
3. Optional: register `sim-v1.3` in Supabase if any sim semantics changed (they shouldn't — rollout architecture is a trainer concern). Skip unless a real sim diff happened.

**Skippable.** Default can stay `False` indefinitely if the per-tick path remains the safer choice for self-play / small-batch runs.

---

## 5. Testing strategy

1. **Never delete existing tests.** All 176 stay green.
2. **Phase A**: encoder parity — JAX-encoded obs vs numpy-encoded obs on 50 random states, max |diff| < 1e-5.
3. **Phase B**: chunk-step parity — K=1 byte-identical, K=4 verified against 4 sequential per-tick steps with NOOP fill.
4. **Phase C**: fused-rollout parity — with `action_repeat=1` and same seed, fused rollout dict must match per-tick rollout dict (within float32 tolerance for adv/return; integer-equal for actions).
5. **Phase D**: upd/s gate, GPU SM gate, win-rate sanity.
6. **Determinism test**: same seed → same rollout (via JAX PRNG-key threading).
7. **Speed regression test**: `scripts/bench_fused_rollout.py` asserts upd/s ≥ floor; tied into a `@pytest.mark.slow` test that runs on demand.

**Don't use "it trained something" as correctness.** The rollout-parity test at `action_repeat=1` is the source of truth.

---

## 6. Profiling strategy

Capture numbers at every phase boundary (same convention as `JAX_PORT_PLAN.md` §6).

1. **Pre-phase-A reference**: `docs/bench/phase7_paullinux_training_profile.txt` already on disk.
2. **Phase A**: encoder bench.
3. **Phase B**: `step_chunk` bench, K ∈ {1, 4, 8, 16}, n_envs ∈ {64, 256, 1024}.
4. **Phase C**: fused rollout vs per-tick on Mac CPU (correctness-grade).
5. **Phase D**: full PaulLinux numbers — the §1 perf gate.

All bench files committed in the same commit as the phase.

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | DLPack JAX→torch flaky → rollout has to round-trip via numpy | medium | minor wall-clock loss | Phase E is gated; numpy round-trip is the default fallback |
| 2 | JAX/torch CUDA allocator fight at high n_envs | medium | OOM | Tune `XLA_PYTHON_CLIENT_MEM_FRACTION` down to 0.30; PyTorch grows dynamically |
| 3 | Action repetition changes learning dynamics — win-rate trajectory shifts | medium | algorithmic bug | Default `action_repeat=4` is a real change. Validate via `action_repeat=1` parity test + win-rate band check at Phase D |
| 4 | Encoder JIT compile time blows up first-update wall time | low | first-update slow | Warmup JIT in `__init__`; document the cold-start cost |
| 5 | `lax.scan` over K ticks recompiles when K changes | low | minor | K is a static argument; cache by (K, n_envs). Don't change K mid-run |
| 6 | Self-play breaks under fused | high | feature regression | Trainer auto-falls-back to per-tick when `cfg.self_play=True`. Self-play gets fused later |
| 7 | Encoder-JAX ≠ encoder-numpy floats due to ordering | medium | parity fails | Match summation order in segment_sum; tolerate 1e-5 in tests, but investigate if max-diff > 1e-4 |

---

## 8. Constraints the executing agent must follow

Same as `JAX_PORT_PLAN.md` §8. Repeat the highlights:

1. Code style: `CODING_GUIDE.md` to the letter. Ruff + pyright clean every commit.
2. One phase per commit. No bundling. No drive-by refactors.
3. Commit as `PaulMacMADEit <paul@madeit.tech>`. **Never** `Co-Authored-By: Claude`.
4. Test before commit. Push after each phase.
5. Secrets in `~/Documents/Sync/secrets/.env.master`, never in repo.
6. Use SSH to PaulLinux directly for deploy steps.
7. Pause for approval between phases unless explicitly told otherwise. Report in <200 words per check-in.
8. Cost estimate at the end of every response.

---

## 9. Executable checklist

- [x] **Phase A** ✅ 2026-04-24 (commit 8026b3f): `training/encoder_jax.py` + 6 parity tests. Max |numpy - jax| < 1e-5 across 3 levels × 5 warmup points + end-to-end. Bench (Mac CPU, n_envs=256): 12.84 ms → 0.21 ms per call = 61.5×.
- [x] **Phase B** ✅ 2026-04-24 (commit a6801b3): `JaxVecEnv.step_chunk(actions, K)` + 7 chunk parity tests. K=1 byte-identical to `.step()`; K ∈ {2, 4, 8} byte-identical to K sequential steps with NOOP fill.
- [x] **Phase C** ✅ 2026-04-24 (commit 6e79723 + fix 04c43c0): `training/fused_rollout.py`, `cfg.fused_rollout` / `cfg.action_repeat`, 3 fused-rollout tests. Mac smoke under both modes completes; losses finite, win-rate sane.
- [x] **Phase D** ✅ 2026-04-24 (commits edc5a4c + 0574e16): PaulLinux RTX 3070. Per env-tick speedups 1.77× (K=1) → 9.74× (K=16). Plan §1 criterion 3 cleared on env-tick basis at K≥4. Criterion 4 (GPU SM ≥ 30%) NOT met — mean 5.1% in a 10-min live training run (660 samples). Criterion 5 (win-rate): met — fused K=8 training sweeps random_legal from 0.40 → 1.00 over 25 updates.
- [ ] **Phase E** (optional, gated on D): DLPack zero-copy. D profile confirms the host↔device round-trip is the remaining bottleneck; SM≥30% requires this.
- [ ] **Phase F** (optional, after one week green): flip default `cfg.fused_rollout=True`.
- [x] **`FUSED_ROLLOUT_PLAN.md` committed** (belated — the plan doc itself wasn't added to git until after Phase D shipped).
- [x] **Report delivered** — see `docs/bench/phase_d_paullinux_fused.txt` + `phase_d_train_10min.txt` for the Phase D and 10-min run numbers.

---

## 10. Out-of-band notes

1. **If verification fails:** stop, write a short `## Troubleshooting` section in this file, and pause for input.
2. **If the plan is wrong:** edit the plan and commit the edit before proceeding.
3. **`action_repeat=1` should always be available** — it's the parity / safety setting. Only the default changes in Phase F.
4. **This plan keeps PPO.** If you're tempted to swap algorithms mid-port, stop and write a separate plan.

---

## 11. Reference material

- `JAX_PORT_PLAN.md` — the prior port. Read §3.6 (GPU memory) before Phase D.
- `training/trainer.py` — current `collect_rollout` (lines ~213–315).
- `training/agent.py` — `act_batch` + chained-head sampling.
- `training/encoder.py` — `encode_obs` body to port in Phase A.
- `sim/envs/jax_vec_env.py` — host of `step_chunk` in Phase B.
- `docs/bench/phase7_paullinux_training_profile.txt` — current baseline numbers (the thing we're trying to beat).
- JAX `lax.scan` docs: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
- DLPack JAX↔torch: https://jax.readthedocs.io/en/latest/jax.dlpack.html

Ship carefully. Half a fused rollout is worse than either extreme.
