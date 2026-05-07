# Curriculum + reward rebalance plan — sim-v1.3

**Goal:** stop training a draw-loving policy. Three coupled changes:

1. **Sim-v1.3 reward rebalance** — make winning matter more than capturing.
2. **Two-phase curriculum** with clear graduation gate (≥95% win rate vs random_legal over 100 games):
   - Phase 1 (training-wheels): close maps, 4–10 buildings, K=4, random_legal opponent.
   - Phase 2 (into the wild): full maps, K=2, mixed self-play + leaderboard opponents.
3. **Strength-by-tournament**: Elo champion picked from head-to-head wins, not "most updates × win rate."

**Status at plan time (2026-04-26):**
- Cron-driven training loop running on PaulLinux: see `project_mushroom_wars_supabase_loop.md` in memory.
- Best checkpoint (`b3_001_endurance_2h`) plateaus at +0.187 reward vs random_legal but plays poor end-game (Paul observed: doesn't finish when it should, doesn't take obvious buildings).
- Diagnosed root causes: (a) reward shape favors capturing+surviving over winning; (b) strength detector picks by `updates × win_rate` not real strength, so curriculum never graduates past phase-1; (c) random_legal is too weak a teacher — agent never sees real threats.

**What this document is:** an execution plan that another agent can pick up. Each phase is one commit-sized unit with a definition-of-done, a verification command, and a rollback point.

---

## 1. Success criteria

Done means **all** of the following hold:

1. `pytest tests/ -q` → all existing tests still pass on numpy + jax backends. New tests under `tests/test_rewards_v13.py` and `tests/test_curriculum.py` cover the new behaviour.
2. `sim-v1.3` registered in Supabase `simulators` table (parent_sim = `sim-v1.2`).
3. Cron-agent automatically:
   - Stays in Phase 1 until the Elo-champion checkpoint hits ≥95% win rate vs random_legal in a 100-game tournament.
   - Once graduated, queues only Phase 2 runs (self-play + leaderboard) — NEVER goes back to random_legal except as a sanity floor (1 run per batch max).
4. Tournament-based strength detection in place: each batch of done runs gets head-to-head matches against the current Elo top-3, results written to `runs.elo_score` (new column or `result.elo`).
5. After 6 hours of training under the new system, the Elo champion **demonstrably finishes games**: in a 100-game eval against random_legal, ≥80% of wins occur **before tick 150** (vs current behaviour where most wins come at the timeout cap).
6. `cfg.reward_v13=True` (new flag) gates the new reward shape; default `False` preserves backward compat for any existing `cfg.fused_rollout` smoke tests.

---

## 2. Non-goals

1. **No new RL algorithm.** PPO stays. No IMPALA, no MuZero.
2. **No net change.** ActorCritic stays as-is.
3. **No K=1.** Per Paul: K=2 is the floor.
4. **No retroactive rewards.** Old checkpoints stay valid as opponents under their original reward scheme; new runs use sim-v1.3 rewards.
5. **No automatic Phase-2 → Phase-3.** There is no Phase 3 in this plan. K=2 is the production setting.
6. **Old level names stay valid.** `random_8_16`, `random_4_8`, etc. don't disappear. We add new `*_close` variants alongside.

---

## 3. Architectural decisions

### 3.1 Reward rebalance — `sim-v1.3`

| constant | sim-v1.2 | sim-v1.3 | reason |
|---|---|---|---|
| `REWARD_WIN`        | 1.0 | **5.0** | dominates the per-capture gradient when summed across a game |
| `REWARD_LOSE`       | -1.0 | **-5.0** | symmetric loss penalty |
| `REWARD_DRAW`       | 0.0 | **-0.5** | drawing is mildly bad — break the agent's preference for "don't lose, don't push" |
| `REWARD_CAPTURE`    | 0.1 | **0.05** | halve so capture-spam doesn't outweigh finishing |
| `REWARD_LOSS`       | -0.1 | **-0.05** | symmetric |
| `REWARD_SPEED_BONUS`| 0.5 | **2.0** | quicker wins compound the +5.0 base — strong incentive to push |
| `gamma` (PPO)       | 0.99 | **0.97** | shorter effective horizon so the win bonus discounts less from the early game |

Total reward range under v1.3: capture-only-then-draw ≈ `+6×0.05 - 0.5 = -0.2`. Win-via-capture ≈ `+6×0.05 + 5.0 + ~1.5 (speed) = ~+6.8`. Win bonus is now ~30× the per-capture gain — not 10× like it was. **Quick-win is now meaningfully better than slow-win.**

**Decision:** these are a set, ship them all together. Reverting any one needs a separate audit.

**Implementation:** new `cfg.reward_v13: bool = False` flag in `PPOConfig`. The trainer passes it through to `sim/config.py` reads via a thread-local override (or a runtime `set_reward_scheme(v="1.3")` call). Sim engine reads through the override.

Cleaner alternative: `sim/config.py` exposes `REWARD_*` as module-level constants today. We can't easily change them at runtime per-run because the engine reads them directly. **Decision:** restructure to `REWARD_WIN_BY_VERSION = {"v1.2": 1.0, "v1.3": 5.0}` and engine reads `REWARD_WIN_BY_VERSION[state.reward_version]`. State carries a string version field. Old states default to "v1.2" (back-compat).

That's a structural sim change; warrants the version bump.

### 3.2 Two-phase curriculum

**Phase 1 — training-wheels:**
- Levels: new `random_close_4_6` (4–6 buildings, 300×300 map) and `random_close_6_10` (6–10 buildings, 350×350 map). **Smaller map = faster travel = shorter games = more eps/sec = faster learning.**
- K = 4 (default fused-rollout setting). Coarse decisioning is fine here because random_legal can't punish coarseness.
- Opponent: 100% random_legal.
- Goal: hit Elo-champion ≥95% win rate vs random_legal in a 100-game tournament.

**Phase 2 — into the wild:**
- Levels: full mix as today (`random_4_8`, `random_6_10`, `random_8_16`, `random_16_24`). Use the full 700×700 map so the agent learns spatial reasoning.
- K = 2 (per Paul — never go below).
- Opponent mix per run: **80% self-play vs current Elo champion**, **15% leaderboard vs Elo top-3**, **5% random_legal floor sanity check**.
- Goal: monotonically improve Elo. (No automatic graduation past Phase 2.)

**Graduation trigger (Phase 1 → Phase 2):**

The cron-agent picks Phase 2 next batch when:
- The current Elo champion (per `runs.elo_score`) achieves ≥95% win rate over a 100-game tournament vs random_legal.
- This is checked at every cron fire as part of the review pass.

Backslide protection: once Phase 2 starts, we don't regress to Phase 1 even if the champion's win-rate vs random_legal drops slightly. (Reasonable assumption — a Phase-2 policy that's worse than random_legal would be a real bug worth investigating, not a curriculum-rollback case.)

### 3.3 Tournament-based strength

**Today:** `cron_agent_pulse._strongest_checkpoint()` ranks by `result.updates × result.rate`. This is broken: `b3_001_endurance_2h` has 300+ updates and rate=1.0; every newer run has fewer updates so never wins. The "champion" never changes.

**Fix:** add an Elo column. Each cron fire's review pass:
1. Pull the last N done runs (N=20 by default).
2. For runs without a recorded Elo score, run a 100-game tournament vs the current top-3 by Elo.
3. Update Elo using a simple Elo-update formula (initial rating 1200, K=32). Write to `runs.elo_score`.
4. Pick the Elo champion as the next-batch self-play opponent.

The tournament can use `scripts/tournament.py` (already accepts Supabase run ids). Each match takes ~3 seconds at n=64. 20 runs × 3 opponents × 1 match = 60 matches = 3 minutes. Fits in the cron's 3h cadence trivially.

**Schema change:** add `elo_score float DEFAULT 1200` to the `runs` table. Backfill = 1200 for all existing rows. Optional: also add `elo_n_matches int DEFAULT 0` for confidence.

### 3.4 Reward gating mechanism

We need backward compatibility — existing fused-rollout tests use sim-v1.2 rewards. Two paths:

**(a)** Per-process global. Set `os.environ["SIM_REWARD_VERSION"] = "v1.3"` at trainer init. Engine reads at module import. Simple but tests can't run both versions in the same Python process.

**(b)** State-carried. `StateJax.reward_version: int8` field (0 = v1.2, 1 = v1.3). Engine indexes into the per-version constants array. Cleanly per-state but adds a field to the pytree.

**Decision:** (b). Cost is one int8 in the pytree (negligible). Lets old parity tests keep running unchanged in the same process as new training runs. Numpy `State` gets the same field.

### 3.5 Map-size variants

Shorter map = faster travel = shorter games. We add new level generators:

```python
# sim/levels.py
def generate_random_close_level(n: int, rng) -> list:
    """Like generate_random_level but on a smaller map (300×300 default)
    so games end faster. Same building rules; just _MAP_SIZE override."""
```

New level names: `random_close_4_6`, `random_close_6_10`. These exist alongside the existing `random_4_8`, etc. — both map sizes available; phase 1 prefers the close ones.

---

## 4. Implementation phases

Each phase is one commit. Each must pass its verification before moving on.

### Phase A — sim-v1.3 reward gating + new constants

**Goal:** introduce the reward-version mechanism and the new constants. No behaviour change unless the new flag is set.

**Scope:**
1. `sim/config.py`:
   - Define `REWARD_BY_VERSION = {"v1.2": {...}, "v1.3": {...}}` table with all six reward constants (WIN, LOSE, DRAW, CAPTURE, LOSS, SPEED_BONUS).
   - Module-level `REWARD_*` constants stay as-is (= v1.2) for backward-compat reads.
2. `sim/state.py` + `sim/state_jax.py`:
   - Add `reward_version: int8` field (0=v1.2, 1=v1.3). Default 0 in factories.
   - `StateJax` pytree gets the same field.
   - Numpy↔JAX converters preserve it.
3. `sim/engine.py` + `sim/engine_jax.py`:
   - Replace direct `C.REWARD_*` reads in `_resolve_arrivals` and `_check_victory` with version-indexed lookups: `C.REWARD_BY_VERSION[reward_version_str][...]` (numpy) or `jnp.where(reward_version == 0, v12_const, v13_const)` (JAX).
4. `sim/levels.py`:
   - `apply()` and `reset()` accept an optional `reward_version: str = "v1.2"` parameter and write it to the State.

**Definition of done:**
- All 193 existing tests still pass with no flag changes (default v1.2 preserves behaviour).
- New `tests/test_rewards_v13.py` runs the same 10 accuracy fixtures with `reward_version="v1.3"` and asserts the v1.3 reward values (e.g. WIN gets +5.0 instead of +1.0).
- Backend parity test still passes — both numpy and jax respect the new field.

**Verification:**
```bash
SIM_BACKEND=jax pytest tests/ -q
pytest tests/test_rewards_v13.py -q
```

**Rollback:** `git revert` — no schema or trainer changes yet.

---

### Phase B — Curriculum config + map-close levels + reward_v13 cfg flag

**Goal:** new level names + new training cfg field that gates the v1.3 rewards. No cron-agent changes yet.

**Scope:**
1. `sim/levels.py`:
   - New `_CLOSE_MAP_SIZE = 350`. Add `generate_random_close_level(n, rng, map_size=350)` mirroring `generate_random_level`.
   - `_resolve_level` regex match `random_close_<min>_<max>` → calls `generate_random_close_level`.
2. `training/trainer.py`:
   - `PPOConfig` gets `reward_v13: bool = False`. When True, the trainer passes `reward_version="v1.3"` to `level_reset` / through to the State factory. Default False keeps current behaviour.
3. `tests/test_levels.py`: smoke test that `random_close_4_6` produces a valid State with all buildings within 350 units (corner checks).

**Definition of done:**
- `pytest tests/ -q` → 195+ passing (added 2 tests).
- Smoke train under the new flag works on Mac CPU: `SIM_BACKEND=jax python scripts/smoke_train.py --seconds 30 --envs 16 --rollout 32 --fused-rollout` with reward_v13=True (CLI flag added).

**Verification:**
```bash
pytest tests/ -q
SIM_BACKEND=jax python scripts/smoke_train.py --seconds 30 --envs 16 --vec-mode sync --rollout 32 --fused-rollout --reward-v13
```

---

### Phase C — Tournament-based Elo + schema change

**Goal:** runs table gets an `elo_score` column; `scripts/tournament.py` updates Elo after a match; cron-agent uses Elo to pick the champion.

**Scope:**
1. **Schema migration**: `ALTER TABLE runs ADD COLUMN elo_score float DEFAULT 1200; ALTER TABLE runs ADD COLUMN elo_n_matches int DEFAULT 0;`. Run via `cli/db.py` script or a one-shot `cli/migrate_v13.py` script.
2. `scripts/tournament.py`:
   - Accept `--update-elo` flag. After running matches, compute Elo deltas using standard formula:
     ```
     E_a = 1 / (1 + 10^((R_b - R_a) / 400))
     R_a' = R_a + K * (S_a - E_a)    # K=32, S_a = 1 win / 0.5 draw / 0 loss
     ```
   - Write back to `runs.elo_score` and increment `elo_n_matches`.
3. `scripts/cron_agent_pulse.py`:
   - Replace `_strongest_checkpoint(runs)` with `_elo_champion(runs)`: pick the run with highest `elo_score` AND `elo_n_matches >= 3` (need at least 3 matches to be ranked).
   - In the review pass, before deciding next batch:
     - Find done runs without an Elo score (or with `elo_n_matches < 3`).
     - For each, run a 100-game tournament vs the current top-3 Elo runs (or vs `random_legal` if there are no other ranked runs yet — bootstrap path).
     - Update Elo.
4. The `_decide_curriculum_phase` function now uses the Elo champion's `result.rate` (vs random_legal in a separate 100-game eval) to gate Phase 1 → Phase 2:
   - Run a 100-game `tournament.py` of champion vs random_legal each cron fire (cheap — 3 seconds).
   - If win_rate ≥ 0.95: enter Phase 2.
5. New helper: `scripts/eval_vs_random.py` (or extend tournament.py with a `--vs-random N` mode) that runs a fixed-100-game match.

**Definition of done:**
- Schema migration applied to Supabase (verify `elo_score` column exists).
- Cron-agent dry-run shows Elo computed and champion correctly picked.
- One real cron fire in production successfully rates ≥3 runs and writes Elo scores back.

**Verification:**
```bash
python scripts/tournament.py --p1 <run_a> --p2 <run_b> --update-elo --games 100
python scripts/cron_agent_pulse.py --dry-run
```

---

### Phase D — Curriculum activation in cron-agent

**Goal:** the cron-agent now actively runs the two-phase curriculum.

**Scope:**
1. `scripts/cron_agent_pulse.py`:
   - Replace the existing `CURRICULUM` table with a two-row version:
     ```python
     CURRICULUM = {
         "phase1_close": {
             "levels": {"random_close_4_6": 0.5, "random_close_6_10": 0.5},
             "K": 4,
             "opponent_mix": {"random_legal": 1.0},
             "reward_v13": True,
         },
         "phase2_wild": {
             "levels": {"random_4_8": 0.2, "random_6_10": 0.3,
                        "random_8_16": 0.3, "random_16_24": 0.2},
             "K": 2,
             "opponent_mix": {"self_play_elo_champ": 0.80,
                              "leaderboard_top3": 0.15,
                              "random_legal": 0.05},
             "reward_v13": True,
         },
     }
     ```
   - `_decide_next_batch` reads from this table per the current phase.
   - Per-run opponent selection samples from `opponent_mix`:
     - `self_play_elo_champ` → use Elo champion's `runs.id` as `opponent_run_id`.
     - `leaderboard_top3` → randomly pick one of the top-3 by Elo.
     - `random_legal` → standard random_legal opponent.
   - Phase decision: `_decide_curriculum_phase()` returns `"phase2_wild"` if the eval-vs-random win rate ≥ 0.95 was achieved at any prior cron fire (track in a small JSON state file or new column `runs.batch_meta`), else `"phase1_close"`.
2. Persistence of "have we graduated?" — simplest is a one-row `kv` table: `key='curriculum_phase', value='phase1_close'|'phase2_wild'`. Or repurpose an existing table. **Decision: add small `kv` table** to avoid coupling phase state to any single run.

**Definition of done:**
- Cron dry-run with no Elo champ → picks Phase 1 → all queued runs are Phase 1.
- Cron dry-run after manually setting `kv.curriculum_phase='phase2_wild'` → all queued runs are Phase 2 with the right opponent mix.
- After running 6 hours under Phase 1 and confirming Elo champion hits ≥95% vs random, the next cron fire automatically advances to Phase 2 (verified by reading `kv` table + the queue).

**Verification:**
```bash
python scripts/cron_agent_pulse.py --dry-run
```

---

### Phase E — Sim-v1.3 registration + monitoring

**Goal:** sim-v1.3 is registered in Supabase, the cron-agent submits runs against it, and we have a small monitor for "is the policy actually finishing games?"

**Scope:**
1. Register `sim-v1.3` in Supabase:
   ```bash
   python cli/register_sim.py --id sim-v1.3 --name "Python sim v1.3 (reward rebalance + close maps)" --parent-sim sim-v1.2 --what-changed "Reward rebalance: WIN=5.0, LOSE=-5.0, DRAW=-0.5, CAPTURE=0.05, LOSS=-0.05, SPEED_BONUS=2.0; gamma 0.99→0.97; new close-map level generators."
   ```
2. `scripts/cron_agent_pulse.py`:
   - `DEFAULT_SIM_ID = "sim-v1.3"`.
   - All queued runs include `"reward_v13": True` in hyperparams.
3. `scripts/eval_finish_speed.py` (new): runs a 100-game eval of the current Elo champion vs random_legal and reports:
   - Mean ticks-to-end.
   - % of wins before tick 150.
   - Histogram of game lengths.
   Logs to a JSON file under `monitoring/`. Useful sanity check.

**Definition of done:**
- `sim-v1.3` row in Supabase.
- Cron-agent queues runs with `simulator_id='sim-v1.3'` and `hyperparams.reward_v13=True`.
- After 6 hours of runs under v1.3, `scripts/eval_finish_speed.py` shows ≥80% of wins before tick 150 (criterion 5).

---

## 5. Testing strategy

1. **Phase A**: backend parity still byte-identical with `reward_version=0`. Add accuracy fixtures with `reward_version=1` checking the new constants.
2. **Phase B**: smoke train under both v1.2 and v1.3 reward schemes works.
3. **Phase C**: tournament-Elo numerically correct (10×10 round-robin sanity check; symmetric matchups produce same Elo deltas).
4. **Phase D**: cron-agent dry-run picks the right phase based on Elo champion win-rate.
5. **Phase E**: end-to-end — eval_finish_speed.py on a freshly-trained v1.3 champion shows quick-win behaviour.

---

## 6. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | New rewards make training unstable (gradient too large) | medium | training collapses | `cfg.reward_v13` is opt-in, default off; can revert per-run via hyperparams |
| 2 | Schema migration locks the runs table briefly | low | minor | run during low-activity window; ALTER TABLE ADD COLUMN with DEFAULT is fast |
| 3 | Cron-agent picks a bad Elo champion early (first 1-2 ranked runs) | high | self-play against weak opponent | bootstrap rule: if all Elo champs have `elo_n_matches < 5`, fall back to random_legal opponent for that batch |
| 4 | Phase-1 graduation never triggers (champion plateaus at 90%) | medium | infinite Phase 1 | manual override via `kv.curriculum_phase` SET command |
| 5 | Map-close levels too easy → champion overfits to close maps and tanks on full | medium | Phase 2 starts weak | Phase 2 immediately mixes 80% full maps; champion will adapt or get dethroned |
| 6 | Reward_v13 rewards added but `gamma` change breaks PPO update math | low | NaN losses | gamma is a normal PPO hyperparam — both 0.99 and 0.97 are well within the safe range |
| 7 | Old checkpoints (v1.2 rewards) used as Phase-2 opponents teach v1.3 agent the wrong thing | low | confused gradients | the policies don't care about the reward scheme, just the obs → action mapping. Mixing is fine |

---

## 7. Constraints

Same as all prior plans. Specifically:

1. Follow `CODING_GUIDE.md`. Surgical diffs. No drive-by refactors.
2. Commit as `PaulMacMADEit <paul@madeit.tech>`. **Never** `Co-Authored-By: Claude`.
3. Test before commit. Push after each phase.
4. Re-read `JAX_PORT_PLAN.md` §3.6 (XLA mem fraction) before any worker-related change — the worker's systemd unit already sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`; don't break it.
5. Doc-sync rule: update `ARCHITECTURE.md` for the reward + curriculum changes when Phase E lands.

---

## 8. Executable checklist

- [ ] **Phase A**: sim-v1.3 reward constants + version field on State + StateJax. `pytest tests/ -q` green; new `tests/test_rewards_v13.py` green. Commit.
- [ ] **Phase B**: close-map level generators + `cfg.reward_v13` flag. Smoke train under v1.3 works on Mac. Commit.
- [ ] **Phase C**: schema migration (`elo_score`, `elo_n_matches`, `kv` table) + `tournament.py --update-elo`. Cron review pass writes Elo. Commit.
- [ ] **Phase D**: two-phase curriculum active in cron-agent + auto-graduation gate. Dry-run shows Phase 1 → Phase 2 transition. Commit.
- [ ] **Phase E**: register `sim-v1.3` in Supabase. Cron submits v1.3 runs. `scripts/eval_finish_speed.py` reports finish-speed of new champion. Commit.
- [ ] **Six-hour shakedown**: let the cron run for 6h under the new system. Verify the Elo champion changes at least once (i.e. self-play actually produces stronger checkpoints).
- [ ] **Report**: a concrete before/after table — old champion's mean-ticks-to-win vs new champion's mean-ticks-to-win, head-to-head Elo, win-rate vs random.

---

## 9. Out-of-band notes

1. **If the executing agent finds a phase decision genuinely contradicts the data**, edit this plan and commit the edit before changing the implementation. The plan is the source of truth.
2. **The agent should NOT touch `cfg.action_repeat=2` settings on already-running Phase-2 runs.** Only NEW runs use the new K. Mid-rollout K changes break PPO.
3. **The 95% graduation gate is the only opinionated number.** If it never trips, the agent can lower it to 90% with Paul's confirmation — but should NOT lower it without asking.
4. **No K=1.** The plan is K=4 in Phase 1, K=2 in Phase 2. K=1 is permanently out of scope.

---

## 10. Reference material

- `JAX_PORT_PLAN.md` — sim port plan; §3.4 levels, §3.6 GPU memory.
- `FUSED_ROLLOUT_PLAN.md` — fused rollout; explains action_repeat semantics.
- `project_mushroom_wars_supabase_loop.md` (memory) — the standing infrastructure; how the cron + worker work today.
- `project_mushroom_wars_fused_k4.md` (memory) — what we learned about K, lr, levels.
- `sim/config.py` — current reward constants; will gain the version table in Phase A.
- `scripts/cron_agent_pulse.py` — the loop's brain; gets curriculum + Elo upgrades in Phases C+D.
- `scripts/tournament.py` — eval; gains `--update-elo` in Phase C.
- Supabase: `runs` table schema; needs `elo_score`, `elo_n_matches`, `kv` table in Phase C.

Ship carefully. The reward rebalance is a real behaviour change — the policy *will* play differently. That's the point.
