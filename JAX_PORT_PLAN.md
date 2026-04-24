# JAX port plan — mushroom-wars-v2 sim

**Goal:** move the simulator tick loop from numpy (CPU) to JAX (GPU), unlock `n_envs=1024+` on the RTX 3070, and stop wasting 97% of GPU cycles while CPU env-step bottlenecks training.

**Status at plan time (2026-04-24):**
- Current sim: numpy, structured dtypes, async vec env on 32 CPU processes.
- Bench: 274 games/sec single-threaded, 677 games/sec at 10 parallel workers (2.3× scaling).
- During live training: GPU SM utilisation 0–3%, CPU load-avg 72–121 on a 16-thread Ryzen, each of 32 env subprocs at ~35% CPU. CPU-bound, full stop.
- Network is small (trunk=128, ~170k params). Doesn't need CUDA; the bottleneck is sim throughput, not net throughput.
- 142 sim tests pass.

**Why now:** the Karpathy loop needs per-round iteration in minutes, not hours. Every 10× in games/sec shortens the outer feedback loop by 10×. JAX+XLA on the 3070 can plausibly deliver 50–200× vs current, based on similar small-RTS workloads.

**What this document is:** an execution plan that another agent can pick up and work through in order. Each phase is a commit-sized unit with a definition-of-done, a verification command, and a rollback point. Ship nothing without passing tests on both backends.

---

## 1. Success criteria (the only things that count)

Done means **all** of the following hold on a fresh clone:

1. `SIM_BACKEND=numpy pytest tests/ -q` → **142+ passing**. Existing tests must never regress.
2. `SIM_BACKEND=jax pytest tests/ -q` → **same test count passing**. Parametrised fixtures run both backends.
3. `pytest tests/test_backend_parity.py -q` → **byte-identical state trajectories** on both backends for 100 scripted seeds × 200 ticks.
4. `python scripts/bench_jax_sim.py` → **≥10× games/sec** vs current numpy bench on PaulLinux RTX 3070, measured over ≥1024 parallel games.
5. A full 20-min Karpathy-loop training run under `SIM_BACKEND=jax` finishes with **GPU SM utilisation ≥40%** (measured via existing telemetry module, `resource_usage.gpu_sm_pct.mean`) and produces a model checkpoint whose vs-top5 is within noise of the numpy backend at equivalent wall time.
6. `sim-v1.2` registered in Supabase `simulators` table (parent_sim = `sim-v1.1`).
7. Default backend stays **numpy** until every criterion above is green. Opt-in via env var. Flip default in a separate commit with a rollback paragraph in the commit message.

---

## 2. Non-goals (things that would look related but are out of scope)

1. Porting level generation (`sim/levels.py`) to JAX. Runs once per episode, not hot. Stays numpy.
2. Porting replay recording (`sim/envs/replay.py`) to JAX. Only runs during eval, not in the training hot path. Stays numpy.
3. Changing game rules or reward shape. This is a pure performance port; gameplay semantics must be byte-identical to sim-v1.1.
4. Rewriting the PyTorch trainer. The trainer keeps consuming numpy arrays; the JAX backend returns numpy (or dlpack-exchange) at the vec-env boundary.
5. Numba. Skipped in favour of JAX — see `JAX vs Numba` note below.
6. Jobs on Modal. Local-first. Modal integration later, if needed.

---

## 3. Architectural decisions

### 3.1 Two-backend architecture (not replace)

**Decision:** keep the numpy sim as the reference oracle. Add `sim/engine_jax.py` and `sim/state_jax.py` alongside. Select at import time via `SIM_BACKEND` env var (`numpy` default, `jax` opt-in).

**Why:** the numpy sim is test-covered and known-correct. If the JAX port introduces a divergence, the numpy path is our ground truth. We never delete it.

**Shape:**
```
sim/
  engine.py              ← numpy (current), unchanged
  state.py               ← numpy (current), unchanged
  engine_jax.py          ← NEW: JAX reimplementation of engine.step_tick
  state_jax.py           ← NEW: JAX state container (pytree)
  backend.py             ← NEW: the only module that reads SIM_BACKEND;
                           exports a `step_tick` symbol that points at
                           whichever implementation is active.
```

Every caller imports `from sim.backend import step_tick, reset_state, ...`. The engine modules themselves never get imported directly outside tests + parity harness.

### 3.2 Functional, not OO

JAX's `@jit` and `vmap` require pure functions: no in-place mutation, no Python control flow over traced values, no lists/dicts in the hot path.

**Decision:** port the hot path to pure functions over pytrees (nested frozen dataclasses of `jnp.ndarray`).

```python
# engine_jax.py
@jax.jit
def step_tick_single(state: StateJax, action_p1: ActionJax, action_p2: ActionJax) -> tuple[StateJax, float, float, bool]:
    state = _apply_send(state, C.OWNER_P1, action_p1)
    state = _apply_send(state, C.OWNER_P2, action_p2)
    state = _advance_production(state)
    state, arrivals = _advance_movement(state)
    state, r1, r2 = _resolve_arrivals(state, arrivals)
    state, dr1, dr2, done = _check_victory(state)
    return state, r1 + dr1, r2 + dr2, done
```

**Consequences:**
- `sim/engine.py`'s in-place `state.buildings["garrison"][src] -= amount` → `state.replace(buildings=state.buildings.at["garrison", src].add(-amount))` (immutable update).
- The `defaultdict`-based `_simultaneous_combat` rewrites to fixed-size arrays. Two-player game means we only ever have 2 hostile owners max — precompute P1 and P2 attacker totals as separate `jnp.ndarray`, no dict needed.
- Event emission (`kind:"capture"` etc.) is **not** in the JAX hot path. When the caller needs replay events, they run under the numpy backend. This is explicit in the `backend.py` contract.

### 3.3 Parallelism via `vmap`, not multiprocessing

**Decision:** `jax.vmap(step_tick_single, in_axes=(0, 0, 0))` over a batched state → N games in one GPU kernel. Replace `AsyncVectorEnv` with a single-process JAX vec env that holds one big batched `StateJax`.

**Why:** async multiprocess IPC is why the current sim peaks at 2.3× scaling. vmap lets XLA fuse everything into one GPU dispatch; there is no IPC and no Python-per-env overhead.

### 3.4 Random keys

JAX has no implicit RNG. Every stochastic call takes a `jax.random.PRNGKey` and returns a new one.

**Decision:** thread a key through `StateJax`. On reset, seed the key. On any stochastic op (not many in the current sim — mostly deterministic), split the key. Drop the Python `np.random.default_rng` usage in the hot path.

Level generation stays on numpy+`np.random.default_rng` (called once per episode; pre-loaded into `StateJax`).

### 3.5 JAX ↔ PyTorch integration

The trainer does:
```python
actions = agent.act_batch(obs, masks)          # obs/masks are np.ndarray from vec.step()
next_obs, rewards, terminated, truncated, infos = vec.step(actions)
```

**Decision:** JAX vec env returns numpy arrays at the boundary via `np.asarray(jax_array)`. No dlpack zero-copy until profiling proves the round-trip is a measurable cost. The trainer code does not change.

**Why the boring choice:** dlpack between JAX and PyTorch sharing a CUDA context works on paper and breaks on many setups (driver versions, CUDA allocator fights). Start with the safe path; optimise only if profiling says so.

### 3.6 GPU memory: PyTorch + JAX coexistence

Both JAX and PyTorch will want CUDA memory on the same RTX 3070 (8 GiB). Default JAX grabs 75% of device memory at import. PyTorch grows dynamically.

**Decision:** set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` at worker startup. JAX gets 3.2 GiB for sim state; PyTorch gets the rest for the net + optimiser. The 170k-param net uses <200 MiB; plenty of headroom.

Document this in `workers/worker.py` startup, with a comment explaining the trade-off.

### 3.7 JAX vs Numba (why not both)

The refactor work (structured array → parallel ndarrays, dict → array, events out of hot path) is ~80% of the effort and is shared between JAX and Numba. Once done, adding a Numba backend is a couple of `@njit` decorators. However:

1. **Numba stays CPU-bound.** Even with JIT, the RTX 3070 sits idle. No path to `n_envs=1024+`.
2. **Numba with structured dtypes is brittle.** We'd be refactoring to parallel arrays anyway.
3. **JAX's ceiling is 10–100× higher.**

**Decision:** skip Numba. If JAX hits an unforeseen ceiling, add Numba as a third backend — the refactor has already paid for it.

---

## 4. Implementation phases

Each phase is one commit. Each phase must pass its verification command before moving on. No bundling. No "while we're here" refactors.

### Phase 0 — refactor numpy sim to port-friendly shape

**Goal:** reshape the current numpy sim so a JAX port is a mechanical translation, not a redesign. Numpy-only changes, semantics preserved, 142 tests still green.

**Scope:**
1. `sim/state.py`: replace structured dtype with parallel ndarrays (`buildings_alive`, `buildings_owner`, `buildings_garrison`, …). Keep the current wrapper functions (`count_owned_buildings`, etc.) as a compatibility shim. One file changes; everything that reads `state.buildings["owner"]` becomes `state.buildings_owner`.
2. `sim/engine.py`: replace `defaultdict`-based simultaneous combat with a fixed-shape array. For a 2-player game: `hostile_counts = np.zeros(3, dtype=np.int32)`, index by owner (0=neutral, 1=P1, 2=P2). Arithmetic-only resolution, no dict.
3. Pull event emission into an optional post-hoc pass (`engine.compute_events(state_before, state_after, actions)` returns a list of events). The hot `step_tick` stops appending to `events` when `events is None`. This matches what the JAX path needs.

**Definition of done:**
- `pytest tests/ -q` → 142+ passing.
- `python scripts/bench_sim.py` → games/sec is the same (±5%) as pre-refactor baseline. Record pre- and post-numbers in the commit message.
- No new public functions; internal ones renamed fine.
- Doc-sync: update `ARCHITECTURE.md §4` and `sim/state.py` docstring.

**Verification:** `pytest tests/ -q && python scripts/bench_sim.py | tee /tmp/bench_phase0.txt`.

**Rollback:** `git revert` — no external contract changes in this phase.

---

### Phase 1 — `sim/state_jax.py`

**Goal:** JAX-native state container. No engine logic yet. The simplest unit.

**Scope:**
1. New file `sim/state_jax.py` with a `flax.struct.dataclass` (or `chex.dataclass` if flax is too heavy) called `StateJax`:
   ```python
   @flax.struct.dataclass
   class StateJax:
       buildings_alive:    jnp.ndarray   # (MAX_BUILDING_SLOTS,) int8
       buildings_owner:    jnp.ndarray
       buildings_garrison: jnp.ndarray
       buildings_capacity: jnp.ndarray
       buildings_x:        jnp.ndarray
       buildings_y:        jnp.ndarray
       groups_alive:       jnp.ndarray   # (MAX_UNIT_GROUP_SLOTS,) int8
       groups_owner:       jnp.ndarray
       groups_src:         jnp.ndarray
       groups_tgt:         jnp.ndarray
       groups_count:       jnp.ndarray
       groups_progress:    jnp.ndarray
       groups_travel:      jnp.ndarray
       travel_matrix:      jnp.ndarray   # (MAX_BUILDING_SLOTS, MAX_BUILDING_SLOTS) int16
       tick:               jnp.ndarray   # scalar int32
       phase:              jnp.ndarray   # scalar int8
       rng_key:            jnp.ndarray   # PRNGKey
   ```
2. `from_numpy_state(state) -> StateJax` — converter for tests + initial state from `sim/levels.py`.
3. `to_numpy_state(state_jax) -> State` — converter back, for parity tests.
4. Pytree registration so `jax.tree_util.tree_map` works.
5. Smoke test: `tests/test_state_jax.py` — construct, round-trip numpy ↔ jax, check all fields equal.

**Definition of done:**
- `pytest tests/test_state_jax.py -q` green.
- `pyright sim/state_jax.py` clean.
- No changes to `sim/engine.py`.

**Verification:** `pytest tests/test_state_jax.py -q`.

**Dependencies:** add `jax>=0.4.30`, `jaxlib>=0.4.30`, `flax>=0.8` (for `struct.dataclass`) or `chex>=0.1.86` to `requirements.txt`. Pin minor versions.

---

### Phase 2 — `sim/engine_jax.py` single-game kernel

**Goal:** pure-function JAX port of `step_tick` for **one** game. No vmap yet.

**Scope:**
1. Port each phase helper:
   - `_apply_send_jax(state, player, action) -> state` — validity check via `jnp.where` masks (no Python `if`s over traced values).
   - `_advance_production_jax(state) -> state` — already vectorised in numpy; trivial port.
   - `_advance_movement_jax(state) -> (state, arrivals)` — arrivals as a fixed-shape `(MAX_UNIT_GROUP_SLOTS, 3)` int array with a companion `arrival_mask`.
   - `_resolve_arrivals_jax(state, arrivals, mask) -> (state, r1, r2)` — fixed 3-slot `hostile_counts` array indexed by owner, arithmetic-only proportional damage. No dict, no Python loops over arrivals.
   - `_combat_jax(garrison, owner_before, hostile_counts) -> (garrison, owner)` — branchless via `jnp.where`.
   - `_check_victory_jax(state) -> (state, r1, r2, done)` — `jnp.where` for tiebreaks.
2. `step_tick_single = jax.jit(_step_tick_single_impl)`.
3. No `vmap` yet. Process one game.

**Definition of done:**
- Parity test `tests/test_backend_parity.py` (new): 50 seeds × 200 ticks of scripted random play, numpy state == JAX state at every tick. Use `to_numpy_state` to compare.
- JIT compile succeeds on CPU (development) and on CUDA on PaulLinux.
- `jax.make_jaxpr(step_tick_single)(example_state, a1, a2)` produces a finite trace with no Python branching.

**Verification:**
```bash
SIM_BACKEND=jax pytest tests/test_backend_parity.py -q -x
```

**Fallback if parity fails:** bisect by phase. Rewrite one phase at a time, parity-test at the phase boundary.

---

### Phase 3 — `vmap` over parallel games

**Goal:** batch N games in one kernel.

**Scope:**
1. `step_tick_batched = jax.jit(jax.vmap(step_tick_single, in_axes=(0, 0, 0)))`.
2. Batched `StateJax` (leading dimension = `n_envs`).
3. New `sim/envs/jax_vec_env.py` matching the gymnasium vec-env API the trainer expects: `reset(seed)`, `step(actions)`, `close()`. Internally:
   - Actions arrive as `(n_envs, action_dim)` numpy. Convert to `jnp.array` (host→device once).
   - Step produces `(n_envs,)` reward, done, info arrays.
   - Reset-on-done is done vectorised: build a mask, regenerate levels for done envs on CPU, load their `StateJax` fields into the batched state.
4. `scripts/bench_jax_sim.py` new bench harness: 1024 parallel games × 200 ticks, measure games/sec and GPU utilisation.

**Definition of done:**
- `pytest tests/test_backend_parity.py -q -k batched` — 10 seeds, 16 parallel games, state parity vs per-game numpy baseline.
- `python scripts/bench_jax_sim.py --n-envs 1024` on PaulLinux → games/sec ≥ 10× numpy baseline (the single-threaded 274/sec number is our floor).
- GPU SM utilisation during the bench ≥ 40% (measured via `nvidia-smi dmon -s u`).

**Verification:**
```bash
SIM_BACKEND=jax python scripts/bench_jax_sim.py --n-envs 1024 --ticks 200
```

**Expected output shape:** `games/sec: 10000+, gpu_sm_pct: 50+`. Below this, keep iterating — don't move to phase 4.

---

### Phase 4 — integrate with PPO trainer

**Goal:** training under JAX backend end-to-end.

**Scope:**
1. `training/trainer.py`: no changes to logic. The trainer imports `from sim.backend import make_vec_env` (new helper in `sim/backend.py`); the backend returns either `AsyncVectorEnv` (numpy) or `JaxVecEnv` (jax).
2. `workers/worker.py`: pick up `SIM_BACKEND` from env. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` before importing jax. Record the backend in `result["sim_backend"]`.
3. `cli/push_experiments.py`: accept `--sim-backend` flag; record in `hyperparams`.
4. 5-minute smoke train under JAX: `python cli/smoke_train.py --backend jax --budget 300`. Should complete without OOM/crash and produce non-garbage metrics.

**Definition of done:**
- Smoke train under JAX completes, metrics sensible (`win_rate > 0.3` vs random at 5 min).
- Numpy smoke train still works unchanged.
- `result["sim_backend"]` present in the DB row.
- Telemetry (from the existing `ResourceSampler`) shows `gpu_sm_pct.mean ≥ 40`.

**Verification:**
```bash
SIM_BACKEND=numpy python cli/smoke_train.py --budget 300
SIM_BACKEND=jax   python cli/smoke_train.py --budget 300
```

---

### Phase 5 — JAX symmetry + accuracy parity

**Goal:** all existing sim correctness tests run against the JAX backend and pass.

**Scope:**
1. Parametrise the relevant tests in `tests/test_sim.py` and `tests/test_accuracy_fixtures.py` with a backend fixture:
   ```python
   @pytest.fixture(params=["numpy", "jax"])
   def backend(request, monkeypatch):
       monkeypatch.setenv("SIM_BACKEND", request.param)
       yield request.param
   ```
2. `test_p1_p2_symmetric_random_play_winrate` must pass on JAX (200 games, P1 rate ∈ [40, 60]).
3. All 10 accuracy fixtures must return byte-identical end-state on both backends.
4. Add parity invariant: for any 100-seed scripted random game × 200 ticks, `to_numpy_state(jax_state) == numpy_state` at every tick (this is the hardest test — leave for last).

**Definition of done:**
- `pytest tests/ -q` → all 142+ tests × 2 backends passing (effectively 284+).
- `tests/test_backend_parity.py` green with `--samples 100 --ticks 200`.

**Verification:**
```bash
pytest tests/ -q
```

---

### Phase 6 — register `sim-v1.2`, flip default

**Goal:** make JAX the default backend. Rollback plan documented.

**Scope:**
1. `sim/backend.py` default changes from `"numpy"` to `"jax"`. Single-line change, separate commit.
2. Register `sim-v1.2` in Supabase with `what_changed = "JAX backend, xla-jit tick loop, vmap parallel games, 50-200x games/sec on GPU"`.
3. Update `ARCHITECTURE.md §3` ("Sim language: numpy + JAX, XLA on GPU; numpy as reference backend").
4. Update `CODING_GUIDE.md §7` if tooling changes (it shouldn't).
5. Commit message includes explicit rollback instructions: set `SIM_BACKEND=numpy` env var to revert.

**Definition of done:**
- All criteria in §1 green.
- `sim-v1.2` row in Supabase.
- README.md updated with backend switching instructions.

**Verification:**
```bash
pytest tests/ -q
python scripts/bench_jax_sim.py --n-envs 1024
# Dashboard: navigate to runs.html, confirm new runs show sim-v1.2
```

---

## 5. Testing strategy (non-negotiable)

1. **Never delete existing tests.** All 142 tests stay green across the whole port.
2. **Parametrise tests by backend** in Phase 5. One source file, two test runs.
3. **Parity test is the source of truth.** `tests/test_backend_parity.py` runs the same seed through both backends and asserts state equality at every tick. When in doubt, this test decides.
4. **Pyright clean** across `sim/*.py` on both backends. No `# type: ignore`.
5. **Ruff clean.** Run before every commit.
6. **Golden replay fixture.** Record a 100-tick replay from numpy, hash the event stream, hard-code the hash in a test. JAX backend must produce the same replay (event order doesn't need to match — gameplay outcome does).
7. **Determinism test.** Same `rng_key`, same actions → byte-identical state under JAX. `test_jax_determinism.py`.
8. **Speed regression test.** `bench_jax_sim.py` asserts games/sec ≥ floor in its own pytest wrapper. Regressions block the build.

**Do not use the "it trained something" heuristic as a correctness signal.** A buggy sim can still train a model that wins vs random. Parity is the test.

---

## 6. Profiling strategy

Measure twice, optimise once. Capture a profile at each phase boundary.

1. **Baseline (record pre-Phase-0):**
   ```bash
   python scripts/bench_sim.py 2>&1 | tee docs/bench/pre_jax_numpy.txt
   ```
   Commit this file. It is the reference point for every claim of speedup.

2. **After Phase 3 (JAX vec env):**
   ```bash
   python scripts/bench_jax_sim.py --n-envs 1024 --ticks 200 \
       --profile-dir docs/bench/phase3_jax_profile/
   ```
   Uses `jax.profiler.trace` for XLA timeline. View with `tensorboard --logdir docs/bench/phase3_jax_profile/`.

3. **Per-phase breakdown** (existing `trainer.sim_phase_breakdown()` already gives rollout/learn split). Extend to also capture:
   - `jax_compile_ms` — time spent in XLA compile on first step.
   - `jax_dispatch_ms` — average per-step dispatch overhead.
   - `device_transfer_ms` — host↔device time.

4. **During full training (Phase 4+):** existing `ResourceSampler` (workers/telemetry.py) captures `gpu_sm_pct`, `vram_used_gib`, `cpu_pct_norm` at 2 s intervals. Compare mean GPU SM% on numpy vs JAX backend. Target: 40%+ on JAX, still 3% on numpy (unchanged).

5. **Artefact layout:**
   ```
   docs/bench/
     pre_jax_numpy.txt              # Phase 0 reference
     phase0_post_refactor.txt       # Phase 0 outcome
     phase3_jax_profile/            # XLA timeline
     phase4_training_profile.txt    # Full training resource_usage
     phase6_final_comparison.md     # Numpy vs JAX side-by-side table
   ```

6. **Commit each bench file** in the same commit as the phase it documents. Future-you will want to see the numbers without re-running the bench.

---

## 7. Deployment rollout

Local-first, then one remote machine, then flip default.

1. **Phase 0–3: Mac dev only.** Run on macOS CPU (JAX works on Mac, no GPU acceleration). Parity tests catch logic bugs without needing the PC.
2. **Phase 4 first run on PaulLinux:**
   - `git pull` on PaulLinux.
   - `.venv/bin/pip install -r requirements.txt` (adds jax, jaxlib-cuda, flax).
   - Verify CUDA JAX: `python -c "import jax; print(jax.devices())"` → `[CudaDevice(id=0)]`.
   - Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` in `~/.config/systemd/user/mushroom-worker.service` `Environment=` line.
   - `systemctl --user daemon-reload && systemctl --user restart mushroom-worker.service`.
   - Queue a 5-minute smoke train under JAX. Confirm success before queueing anything bigger.
3. **Phase 5 gated deploy:** no run under JAX backend lands in the admission pool until Phase 5 is done and parity is green. Quarantine JAX-backend runs with a label prefix `jax-`.
4. **Phase 6 default flip:** after one week of dual-backend running with zero parity failures, default flips. Communicated in commit message with rollback env var.
5. **Modal bursting:** later. Not in this plan.
6. **Mac worker:** Mac uses Metal, which JAX supports via `jax-metal` (experimental). First pass keeps Mac on numpy backend. If the PC-only JAX path is stable after a week, revisit Mac.

---

## 8. Constraints the executing agent must follow

These are repo + project rules. Not optional.

### 8.1 Code style
1. **Follow `CODING_GUIDE.md` to the letter.** Especially §1 ("simplicity first"), §3 (no hard-coded values — put in `config/*.yaml`), §4 (tests next to code, one integration test per flow), §5 (doc-sync rule).
2. **Ruff + pyright clean on every commit.** `pre-commit` hook enforces it; don't bypass.
3. **Surgical diffs.** Don't drive-by-refactor. If you see something ugly outside your scope, open an issue; don't fix it in this branch.
4. **No speculative code.** If a helper isn't used, delete it before commit.
5. **Comments are WHY, not WHAT.** Default to zero comments. A short comment is only warranted when a reader of the code would otherwise be surprised.

### 8.2 Git + commits
1. **One phase per commit.** The phases in §4 are sized deliberately — don't bundle.
2. **Commit as `PaulMacMADEit <paul@madeit.tech>`** (already the default `git config`).
3. **Never add `Co-Authored-By: Claude`.** Never. This is a project rule.
4. **Commit messages:** first line imperative subject (<72 chars); body explains WHY not WHAT; include before/after bench numbers for performance commits. Match the style of recent commits on `main`.
5. **Never force-push `main`.** Never rewrite shared history.
6. **Test before commit, every time.** `pytest tests/ -q` has to pass. If it doesn't, the commit waits.
7. **Push to GitHub after each phase.** `origin/main` is the deploy truth on PaulLinux.

### 8.3 Secrets + config
1. **Secrets live in `~/Documents/Sync/secrets/.env.master`** (on Mac). Never in the repo, never in memory files, never in commit messages.
2. **`.env` is gitignored.** Keep it that way. If you need a new secret, add the key to `.env.example` without the value.
3. **Supabase creds already live in `.env`.** Don't move them; read via existing `cli/db.py`.
4. **Per-project rules from `CLAUDE.md` (user's global):** numbered lists in user-facing responses, cost estimate at end of every response, latest model default, no enterprise-style ceremony.

### 8.4 Deployment operations
1. **Always use direct SSH access to PaulLinux** (`paul@192.168.1.137`) instead of writing instructions for Paul to run. He set up the keys for a reason.
2. **Restart the worker via systemd:** `systemctl --user restart mushroom-worker.service`. Don't `kill -9` — the SIGINT path in the worker handles graceful run termination.
3. **Mark orphaned running rows as failed** after forced restarts. Pattern is in `workers/worker.py`'s failure handler; mirror it for manual cleanups.
4. **Check running runs before restarting.** If a non-JAX run is active, let it finish. Queue-poll interval is 5s; a push-and-restart is safe within ~10s of quiescence.

### 8.5 Interaction with Paul
1. **One experiment at a time.** Karpathy loop. Don't batch.
2. **Pause for approval between rounds.** Unless explicitly told "autonomous, just go," wait after each phase with a short report.
3. **Report in under 200 words per check-in.** Tables > prose. Numbered lists > bullets.
4. **Parse voice-typing artefacts charitably** (see `feedback_voice_transcription.md` in memory). Don't interrogate ambiguities round-by-round; batch them.
5. **Cost estimate at the end of every response.** Global rule.

---

## 9. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | JAX/PyTorch GPU memory fight → OOM | high | blocks Phase 4 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` at startup. If still OOM, drop to 0.30 and profile net size |
| 2 | JIT recompile on state-shape change | medium | slow first step | Shapes are fully static (MAX_BUILDING_SLOTS etc). Verify with `jax.make_jaxpr`. Single compile at first step, cached after |
| 3 | Parity divergence — JAX produces a different state | high | blocks port | Parity test catches it. Bisect by phase helper. Rounding / integer division order is the usual culprit |
| 4 | GPU driver crash under heavy JAX load | low | blocks PC | Keep numpy backend as fallback. Telemetry flags gpu_sm=0 anomalies |
| 5 | Mac (Metal) JAX too buggy to use | medium | Mac worker stays on numpy | Accepted. Document it |
| 6 | Reset-on-done logic is expensive on GPU | medium | blocks Phase 3 perf | Build level-gen on CPU, push refreshed fields to GPU in a single `jnp.where`. Don't JIT level-gen |
| 7 | `vmap` fails on a control-flow in `_resolve_arrivals` | medium | Phase 3 blocker | Use `lax.cond` / `lax.switch` for any branching over traced values. Fixed-shape resolution means no dynamic-shape hazards |
| 8 | Flax / chex version churn breaks pytree | low | minor rework | Pin minor versions in `requirements.txt`. `chex.dataclass` is the more stable choice |

---

## 10. Executable checklist (work through in order)

Copy this into a scratch todo file as you execute.

- [ ] **Phase 0**: refactor numpy sim (structured dtype → parallel ndarrays, dict → fixed array, events out of hot path). `pytest tests/ -q` green. `scripts/bench_sim.py` ≤5% regression. Commit.
- [ ] **Phase 1**: add `sim/state_jax.py`, converters, smoke test. `pytest tests/test_state_jax.py -q` green. Add `jax`, `jaxlib`, `flax` (or `chex`) to `requirements.txt`. Commit.
- [ ] **Phase 2**: port `step_tick` to `sim/engine_jax.py` as a `jax.jit`ed single-game function. `pytest tests/test_backend_parity.py -q` green. Commit.
- [ ] **Phase 3**: `vmap` over 1024 games. New `sim/envs/jax_vec_env.py`. `python scripts/bench_jax_sim.py --n-envs 1024` ≥10× numpy bench. Commit.
- [ ] **Phase 4**: trainer integration. `SIM_BACKEND` env var. 5-min smoke train on both backends. Record `resource_usage` shows gpu_sm_pct ≥40. Commit.
- [ ] **Phase 5**: parametrise tests, full 2-backend run green. Parity test 100 seeds × 200 ticks. Commit.
- [ ] **Phase 6**: register `sim-v1.2`, flip default. Rollback paragraph in commit message. Commit.
- [ ] **Deploy to PaulLinux**: git pull, install jaxlib-cuda, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` in systemd unit, restart worker, queue a smoke train under `sim-v1.2`. Confirm success.
- [ ] **Update** `ARCHITECTURE.md` + `KARPATHY_LOG.md` + `JAX_PORT_PLAN.md` (mark phases complete).
- [ ] **Report to Paul** with: total games/sec before/after, GPU SM% before/after, wall-time per Karpathy round before/after. One table. Plus any surprises worth a memory entry.

---

## 11. Out-of-band notes for the executing agent

1. **If any phase's verification fails:** stop, write a short diagnosis in this file under a new `## Troubleshooting` section, and pause for user input. Don't skip phases.
2. **If you discover the plan is wrong** (a phase should be split, a decision should change), **edit this file** and commit the edit as its own change before proceeding. The plan is the source of truth; keep it accurate.
3. **Benchmarks go in `docs/bench/`** — create the directory if it doesn't exist. Every bench run commits its raw output, even if noisy. Future-you or Paul will want the raw numbers.
4. **If the user says "pause" or "stop":** stop immediately. Current phase may be half-done; that's fine, next session picks it up from the checklist.
5. **This is a performance port, not a rewrite.** Gameplay semantics are frozen at sim-v1.1. Any semantics change = a new sim version (v1.3+), separate PR, separate plan.

---

## 12. Reference material

- `ARCHITECTURE.md` — current design. §3 covers stack decisions, §4 data model, §8 sim data structures.
- `CODING_GUIDE.md` — enforcement rules. Re-read §1, §3, §4 before starting.
- `sim/engine.py` — current sim (numpy). Source of truth for semantics.
- `tests/test_sim.py` — 142 tests. New tests go next to it.
- `tests/test_accuracy_fixtures.py` + `tests/fixtures/levels/accuracy/*.json` — 10 human-readable scenarios; every one must pass on both backends.
- `workers/telemetry.py` — already captures GPU SM%, VRAM, CPU. Don't reinvent.
- JAX docs: https://jax.readthedocs.io/en/latest/jax-101/index.html (if refreshing on JIT/vmap/pytree).
- `flax.struct.dataclass`: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html.
- Chex (simpler alternative): https://github.com/google-deepmind/chex.

Ship carefully. The port is worth the care; a half-ported sim is worse than either extreme.

---

## 13. Post-port training protocol (applies once sim-v1.2 is the default)

These are methodology changes Paul requested on 2026-04-24 alongside the JAX port. They are **not** part of the port itself — don't bundle them into Phases 0–6. They are the first set of changes to make in a new PR **after** the port has landed and proven stable.

### 13.1 Training level must cycle through all level variants

**Rule:** during a single training run, each env draws its level from the full level catalog with uniform probability per episode. No run trains on just one level.

Current behaviour: `cfg.level_name = "random_8_24"` pins every env to the same level shape for the whole run.

Target behaviour: `cfg.level_name = "mixed:all"` (or a new `cfg.level_cycle` list) draws per-episode from the catalog:
- `crossroads_6` (static)
- `random_6_10`, `random_8_12`, `random_8_16`, `random_8_24`, `random_8_32` (symmetric)
- `asym_8_12`, `asym_8_16`, `asym_8_24`, `asym_8_32` (asymmetric)

Exact catalog to be confirmed at implementation time — read `sim/levels.py` and pick every registered variant that has a generator. The policy needs to learn a general prior over board shapes, not a specialist one per run.

Implementation hint: `sim/envs/mushroom_env.py`'s `reset()` currently passes `cfg.level_name` straight to `sim.levels.apply()`. Introduce a resolver: if `level_name` starts with `mixed:`, each `reset()` draws a concrete level name from the catalog using the env's RNG. Episode-level mixing; don't swap levels mid-game.

### 13.2 Admission evaluation must cycle over the same catalog

**Rule:** admission matches evaluate each run over the same mix of levels it trained on. Every opponent plays every level.

Current behaviour: `ADMISSION_LEVEL = "random_8_12"` in `workers/worker.py` — all admission matches use one level. This created the train/eval mismatch that confused round 1 under sim-v1.1.

Target behaviour: admission queues matches across every level in the catalog. Match count per opponent may drop per-level to stay inside the 3–5 min drain budget — e.g. 12 games × 10 opponents × 10 levels = 1,200 games ÷ 10 = 120 games/opponent. Needs tuning; start with **3 games per (opponent, level)** so total drain stays around the current ceiling.

Aggregation: compute a per-level win rate *and* a cross-level mean. Dashboard/run page should show both so regressions on one level can't hide in the average.

### 13.3 Train sim and eval sim must be identical

**Rule:** any run's admission matches must be played under the same `simulator_id` the run was trained on. No cross-sim admission.

Current behaviour: admission inherits the new run's `simulator_id` via `(SELECT simulator_id FROM runs WHERE id = %s)` in `workers/worker.py`. Good — this is already correct for the new run's side, but the **opponent** was trained on a potentially different sim. Matches mix sim-v1.0-trained and sim-v1.1-trained policies under the newer sim.

Target behaviour: admission only queues matches against opponents trained on the *same* simulator_id. Runs under sim-v1.2 evaluate against sim-v1.2 opponents only. Pool membership is sim-scoped.

Consequence: when sim-v1.2 lands, the initial pool is empty and new runs have nobody to play against. Two options for bootstrapping:
1. Run the rebench pass first (§13.4) so the pool is populated with sim-v1.2-native ratings of old checkpoints.
2. Seed the pool with `random_legal` + 2–3 fresh fast training runs under sim-v1.2 to establish a baseline before heavier experiments begin.

Pick option 1 — it's more principled and avoids a cold-start week.

### 13.4 Re-bench all existing model checkpoints under sim-v1.2

**Rule:** before the first experimental run under sim-v1.2, pull every existing run's weights (no retraining), have them play a round-robin against each other **under sim-v1.2**, and record the results as a fresh leaderboard.

**Scope:**
1. Identify every `runs.id` in Supabase with `status='done'` and a non-null `weights_url`. Current count: 64 done runs.
2. Build a new table `rebench_matches` (or re-use `matches` with a `rebench` description tag).
3. Queue matches pairwise across them on sim-v1.2, across the full level catalog from §13.1.
4. Recompute Elo from scratch from the rebench results only. Old Elo numbers from sim-v1.0 / v1.1 matches are archived but no longer ranked.
5. This is compute-heavy. 64 × 64 / 2 = 2,016 pairs × 10 levels × 3 games = 60,480 games. On sim-v1.2 with JAX at ~10,000 games/sec this is ≈6 seconds of pure sim plus policy-inference overhead. Estimate more carefully at implementation time; may need to cap opponent count to top-20-by-prior-Elo if total bloats.

**Why:** Paul is fine discarding old rankings but not old checkpoints. Retraining 64 runs would cost weeks; re-benching costs hours and gives us a clean, same-sim leaderboard.

**Gating:** §13.1, §13.2, §13.3 must land before §13.4 runs (rebench is the first thing that exercises them at scale).

### 13.5 Config hygiene — all runs compared must share settings

**Rule:** when two runs are being compared (A/B experiments, ablations), every hyperparam except the one under test must be identical. Record the full hyperparam dict in Supabase (already done via `runs.hyperparams`), and *print the diff* at comparison time on the dashboard.

Implementation: a small `dashboard/lib/hyperparam_diff.js` helper that takes two `runs.hyperparams` JSON blobs and renders a "↓ same: X. Δ: ent 0.003 vs 0.01, level mixed:all vs random_8_12" strip. No configuration drift between compared runs without it being obvious.

### 13.6 Sequencing

In order, after sim-v1.2 is merged and default-flipped:
1. Implement §13.1 (training-level cycling) — new commit.
2. Implement §13.2 (admission-level cycling) — new commit.
3. Implement §13.3 (sim-scoped admission pool) — new commit.
4. Implement §13.5 (dashboard hyperparam diff) — new commit.
5. Run the §13.4 rebench pass. Commit the rebench script + the Elo rebuild results.
6. Resume the Karpathy loop on sim-v1.2 — clean sim, clean leaderboard, cycling levels, sim-scoped pool. This is the regime where round-2 experiments should actually be queued.

Each item is its own commit with tests + doc-sync per `CODING_GUIDE.md`. No bundling.

