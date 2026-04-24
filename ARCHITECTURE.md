# Mushroom Wars v2 — Architecture

Design document for the ground-up rebuild of the Mushroom Wars RL training stack.
Captures every core decision, the reasoning, and the rejected alternatives.

---

## 1. Project goals

Build an RL training platform for Mushroom Wars that:

1. **Scales to bigger models.** Current 58k-param nets hit their ceiling; we want to reach 1M-10M+ param architectures (CNN/transformer over entities).
2. **Runs on GPU** — Apple Metal on Mac, CUDA on PC, cloud GPUs (Modal) on demand.
3. **Runs truly autonomously.** Orchestration survives laptop sleep, worker crashes, network outages. Any machine can push work or pull work from anywhere.
4. **Tracks experiments first-class.** Models, simulator versions, and runs are distinct entities with their own lifecycle, stats, and retention.
5. **Keeps the game playable by humans.** Phaser-based TS game stays for dogfooding. Sim logic is parity-tested between the Python RL sim and the TS playable sim.
6. **Has a proper dashboard.** Runs in progress, models, sims, leaderboards — all live, all filterable.
7. **Is the foundation for future projects too.** Supabase + Modal + worker pattern generalizes; the Kerri research project and other future work can share the same infra.

Non-goals:
- Online multiplayer PvP (future, not now).
- Mobile client (not now).
- Fancy 3D rendering (keep the simple 2D Phaser view).

---

## 2. Why rebuild instead of refactoring

The current `mushroom-wars/` repo has accumulated layers of legacy:

- 5 observation encoders, 6 action decoders, QAgent + PPO mixed together
- BullMQ + Redis + rsync + pc.sh + runs.json + history.json + legacy orchestrator + queue orchestrator
- Dynamic slot re-sort in the encoder (v3.1) baked a pathology into every version up through v8.0
- Training state lives in local Redis which dies whenever the Mac sleeps

Each cleanup we've attempted has compounded on top of old decisions. The repo is now ~15k lines with ~40% dead code. Every new feature costs more than it should.

**Rebuild is cheaper over 6 months than continued patching.** We preserve what demonstrably works (the game simulation logic, a handful of champion model weights) and drop everything else.

---

## 3. Stack decisions

| layer | choice | rejected alternatives | why |
|---|---|---|---|
| RL training language | **Python + PyTorch** | TypeScript (current), JAX | PyTorch is the RL ecosystem. GPU primitives, gymnasium, RLlib, SB3, etc. TS scalar JS caps ~1 GFLOP; unable to scale nets. |
| Sim language | **Python + numba + AsyncVectorEnv** | Rust, C++/pybind11, JAX | numba gets ~100× speedup on hot paths without leaving Python. Parallel sims via AsyncVectorEnv. Rust saved for later if numba bottlenecks. |
| Queue / state | **Supabase Postgres** | BullMQ + Redis (current), Firebase, SQS | One table does the entire queue. Cloud-hosted Postgres survives your laptop. Free tier plenty. Atomic claim via `FOR UPDATE SKIP LOCKED`. |
| Artifact storage | **Supabase Storage** | Git LFS, S3, R2 | Included with Supabase, S3-compatible, free 1 GB then $0.021/GB. No second service. |
| Compute — home | **Mac (Metal) + PC (CUDA) daemons** | Pure cloud | Home compute is already paid for. Zero marginal cost per run. |
| Compute — cloud burst | **Modal** | HF Jobs, RunPod, AWS, Vast.ai | Best Python DX (`@modal.function(gpu="a10g")`). Per-second billing. $30/mo free. Same worker code as home. |
| Dashboard | **Supabase Studio + Next.js on Vercel** | Static HTML, Grafana, Metabase, Retool | Studio for admin / SQL. Next.js for polished pages (model detail, tournament, replay viewer). Vercel free tier covers it. `git push` = deploy. |
| Human-playable game | **dropped** (revisit later if wanted) | TypeScript + Phaser with parity tests | Parity between two sims is a recurring maintenance cost. Keep Python as the only canonical sim. Replay viewer in the dashboard handles dogfooding. Build a browser game later as a separate project if desired. |

---

## 4. Data model

Three canonical entities, all in Supabase Postgres:

### 4.1 `models` — the neural net designs

A **Model** is a specific architecture: obs shape, action shape, layer topology, encoder+decoder pair.
Rare new entries. `v9.0` is a Model. A new Model is created only when the design itself changes.

```sql
CREATE TABLE models (
  id             TEXT    PRIMARY KEY,         -- 'v9.0' (semantic slug)
  project        TEXT    NOT NULL,
  name           TEXT,
  created_at     TIMESTAMPTZ DEFAULT now(),
  created_by     TEXT,
  parent_model   TEXT    REFERENCES models(id),
  what_changed   TEXT,                        -- why this version exists
  obs_size       INTEGER NOT NULL,
  num_actions    INTEGER NOT NULL,
  obs_encoder    TEXT,                        -- reference to code module
  action_decoder TEXT,
  layers         JSONB   NOT NULL,            -- structured topology
  total_params   INTEGER,
  keep_weights   BOOLEAN DEFAULT false,
  run_count      INTEGER DEFAULT 0,           -- trigger-maintained
  best_rate      REAL,
  best_run_id    UUID
);
```

### 4.2 `simulators` — game engine versions

A **Simulator** is a specific version of the game engine. Distinct from models. Rule changes, performance improvements, or bug fixes that could alter agent behavior bump the sim version.

```sql
CREATE TABLE simulators (
  id             TEXT    PRIMARY KEY,         -- 'sim-v1.0'
  project        TEXT    NOT NULL,
  name           TEXT,
  created_at     TIMESTAMPTZ DEFAULT now(),
  parent_sim     TEXT    REFERENCES simulators(id),
  what_changed   TEXT,
  git_sha        TEXT,                        -- commit at registration
  features       JSONB,                       -- key game constants
  benchmark      JSONB,                       -- {ticks_per_sec, games_per_hour, ...}
  run_count      INTEGER DEFAULT 0
);
```

### 4.3 `runs` — individual training experiments

A **Run** is one training experiment: a specific Model × Simulator × hyperparams × seed.
Common. Dozens per day. Every training job = one row.

```sql
CREATE TABLE runs (
  id                   UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id             TEXT    NOT NULL REFERENCES models(id),
  simulator_id         TEXT    NOT NULL REFERENCES simulators(id),
  project              TEXT    NOT NULL,
  label                TEXT    NOT NULL,
  description          TEXT,
  status               TEXT    NOT NULL,       -- queued/running/done/discarded/failed
  result               JSONB,
  error                TEXT,
  budget_ms            INTEGER NOT NULL,
  seed                 TEXT,
  hyperparams          JSONB,                  -- lr, entropy, tail size, etc.

  -- continue-training support: a run that extends a previous run's weights
  parent_run_id        UUID    REFERENCES runs(id),
  is_continuation      BOOLEAN DEFAULT false,
  cumulative_budget_ms BIGINT,                 -- this run's + all ancestors'

  games_played         INTEGER,
  training_games       INTEGER,
  eval_games           INTEGER,
  ticks_played         BIGINT,
  machine              TEXT    NOT NULL,       -- 'mac' | 'pc' | 'modal-a10g'
  launch_at            BIGINT  NOT NULL,
  queued_at            TIMESTAMPTZ DEFAULT now(),
  started_at           TIMESTAMPTZ,
  finished_at          TIMESTAMPTZ,
  wall_ms              INTEGER,
  weights_url          TEXT,                   -- state_dict pointer
  optimizer_url        TEXT,                   -- Adam state for resumption
  obs_norm_url         TEXT,                   -- running mean/std for resumption
  log_url              TEXT,
  keep_weights         BOOLEAN DEFAULT false
);

CREATE INDEX ON runs(project, status, queued_at);
CREATE INDEX ON runs(model_id, status);
CREATE INDEX ON runs(simulator_id);
CREATE INDEX ON runs(machine, status);
CREATE INDEX ON runs(parent_run_id);
```

Triggers maintain `models.run_count`, `models.best_rate`, `simulators.run_count`.

### 4.4 `matches` and `games` — head-to-head tournaments

Separate from training runs. Pick two trained models, play N games, dig into any game.

```sql
CREATE TABLE matches (
  id             UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  project        TEXT    NOT NULL,
  description    TEXT,
  model_a_run_id UUID    NOT NULL REFERENCES runs(id),   -- specific weights
  model_b_run_id UUID    NOT NULL REFERENCES runs(id),
  simulator_id   TEXT    NOT NULL REFERENCES simulators(id),
  games_planned  INTEGER NOT NULL,
  status         TEXT    NOT NULL,
  created_at     TIMESTAMPTZ DEFAULT now(),
  finished_at    TIMESTAMPTZ,
  summary        JSONB                        -- {a_wins, b_wins, draws, avg_duration, ...}
);

CREATE TABLE games (
  id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  match_id        UUID    REFERENCES matches(id) ON DELETE CASCADE,
  game_index      INTEGER NOT NULL,
  seed            INTEGER NOT NULL,
  map_name        TEXT,
  player_1_run_id UUID    NOT NULL,
  player_2_run_id UUID    NOT NULL,
  winner          TEXT,
  duration_ms     INTEGER,
  stats           JSONB,
  actions_url     TEXT,                       -- Storage: compact action log (replay-able)
  log_url         TEXT                        -- Storage: narrated human-readable log
);

CREATE INDEX ON matches(project, status, created_at);
CREATE INDEX ON games(match_id, game_index);
```

Why separate from runs: matches are evaluation, not training — no `budget_ms`, no weights produced, no opponent pool. Own tables keep both clean.

### 4.5 Storage bucket layout

```
supabase-storage/
  models/
    <run-id>/
      weights.pt            ← PyTorch state_dict (~4 MB)
      optimizer.pt          ← Adam state (for continue-training, ~8 MB)
      obs_norm.pt           ← running mean/std (for continue-training, ~3 KB)
      metrics.json          ← training-curve timeseries
  logs/
    training/<run-id>.log   ← full training stdout
    games/<game-id>.log     ← narrated game log (tournament games)
  replays/
    <game-id>/actions.json  ← compact action log, re-runnable to reproduce a game
```

### 4.6 Retention policy

**Auto-pin champions (system-managed):**
- Top-3 runs by `result.rate` per (model_id, simulator_id) pair → `keep_weights = true` automatically at completion
- Your best v9.0 on sim-v1.0 is preserved forever without intervention

**Manual pin (user-managed):**
- `models.keep_weights = true` → all runs of that model kept forever
- `runs.keep_weights = true` → that specific run kept forever
- Toggle from the dashboard

**Default (unpinned) runs:**
- Weights + optimizer + obs_norm deleted 30 days after `finished_at` OR when model is >100 models old, whichever is LATER
- Postgres rows stay forever (metadata is cheap)
- Tournament replays: 30 days unless parent match pinned

**Pruner:** nightly Supabase Edge Function. Deletes Storage blobs, nulls out `*_url` columns on affected rows.

---

## 4.7 Cross-version compatibility — what works and what doesn't

The honest truth: **true backwards/forwards compatibility is impossible when the architecture itself changes.** Here's what we can and can't do:

| change type | same weights continue training? | old weights play new code? |
|---|---|---|
| Hyperparameter (lr, entropy, budget) | ✅ yes — continue_training just works | ✅ yes |
| Slot semantics change (e.g. v8→v9 static slots) | ⚠️ shapes match, but input distribution is different — weights degrade. Don't. | ✅ yes — each model uses its own encoder; both play the same sim |
| Net width (h64 → h128) | ❌ tensor shapes differ | ✅ yes — shape is private to the model |
| Action/obs size (14→32 slots) | ❌ output/input layers differ | ⚠️ old model can only use first 14 slots; new code must fall back |
| New features added to obs | ❌ input shape differs | ⚠️ new code must zero-fill missing features for old models |

**Practical policy:**

1. **Continue-training (same Model + Sim)**: fully supported via `parent_run_id`. This is the "keep iterating on a good seed" flow you wanted.
2. **Cross-version eval / play**: always works. Each model loads with its own Model spec → its own encoder/decoder. They coexist in the same tournament.
3. **Weight transplant across architectures**: don't attempt. Treat old champions as opponents, not starting points.

**The one hedge we're taking:** designing with `MAX_BUILDING_SLOTS = 32` from day 1, so level/capacity expansion doesn't force a Model bump. Same for `MAX_UNIT_GROUP_SLOTS = 32`. This absorbs the next 2-3 years of map expansion without architectural churn.

---

## 5. Versioning rules

### When to bump a Model version

- **Major** (v8.0 → v9.0): I/O shape change, new encoder, new decoder, new head topology, any change that makes old checkpoints incompatible
- **Minor** (v7.1 → v7.2): Training correctness fix, sampling bug fix, anything that changes effective training regime without changing tensor shapes
- **No bump**: Hyperparameter tuning, new seeds, new compute. These are Run variations, not Model variations.

Rule of thumb: **if someone else's trained weights can be loaded and work correctly, no version bump is needed.**

### When to bump a Simulator version

- Rule changes (combat logic, production timing, victory conditions)
- Decision-cadence changes (phase-lock fix was a sim change — it altered observed state at decision time)
- Performance improvements you want to measure (that's the whole point of tracking sim versions)
- Map generator changes

Not sim-bump-worthy: cosmetic refactors, test additions, comment changes.

### When a Run is just a Run

A Run is never its own "version" — it's always a (Model, Simulator, hyperparams, seed) tuple. The label `v9.0.200-20m-probe-a` decomposes into:
- `model_id`: `v9.0`
- `simulator_id`: whatever `SIM_VERSION` was at push time
- `budget_ms`: 20 min = 1200000
- `tag`: `probe`
- `seed`: `a`

---

## 6. Code layout

```
mushroom-wars-v2/
├── ARCHITECTURE.md              ← this file
├── README.md
├── pyproject.toml               ← Python project config
├── package.json                 ← TS (Phaser game only)
│
├── sim/                         ← Python simulator (canonical)
│   ├── __init__.py
│   ├── simulation.py            ← game loop + entity manager
│   ├── entities.py              ← Building, UnitGroup, Player (numpy-backed)
│   ├── systems/
│   │   ├── production.py        ← 1Hz production tick
│   │   ├── movement.py
│   │   ├── combat.py
│   │   ├── victory.py
│   │   └── waypoint.py
│   ├── levels.py                ← Crossroads + random map generator
│   ├── config.py                ← constants, tick rates, physics
│   ├── sim_version.py           ← SIM_VERSION = 'sim-v1.0' (single source)
│   └── envs/
│       ├── mushroom_env.py      ← gymnasium Env wrapper
│       ├── vec_env.py           ← AsyncVectorEnv configuration
│       └── replay.py            ← replay recorder
│
├── training/                    ← PyTorch RL
│   ├── __init__.py
│   ├── encoder.py               ← ObservationEncoder (static slots)
│   ├── decoder.py               ← ActionDecoder (static slots, 5-wide type)
│   ├── net.py                   ← PyTorch nn.Module (body + 4 heads + embed)
│   ├── agent.py                 ← PPOAgent (action sampling, value, logprob)
│   ├── trainer.py               ← PPO training loop (GAE, clipped loss, updates)
│   ├── evaluator.py             ← champion-pool eval
│   └── model_version.py         ← MODEL_VERSION = 'v9.0' (single source)
│
├── workers/                     ← Long-running daemons
│   ├── worker.py                ← Postgres-polling worker (claim→run→report)
│   ├── benchmark.py             ← sim performance benchmark (for register-sim)
│   ├── prune.py                 ← nightly storage pruner
│   ├── launchd/
│   │   └── com.mushroomwars.worker.plist  ← macOS LaunchAgent
│   └── systemd/
│       └── mushroom-worker.service         ← Linux systemd unit
│
├── cli/                         ← Operator tools
│   ├── register_model.py        ← introspect code → INSERT INTO models
│   ├── register_sim.py          ← benchmark + INSERT INTO simulators
│   ├── push_experiments.py      ← queue a batch
│   ├── reeval.py                ← re-run eval for a stored model under current sim
│   └── migrate_legacy.py        ← one-time: import v7.2/v8.0 champions
│
├── modal_app/                   ← Cloud-burst entry points
│   ├── worker_image.py          ← Modal Image definition
│   └── run_worker.py            ← @modal.function wrapping workers/worker.py
│
├── infra/
│   ├── schema.sql               ← Supabase schema (tables, indexes, triggers)
│   ├── rpc.sql                  ← stored procedures (claim_next_run, etc.)
│   ├── rls.sql                  ← row-level security policies
│   └── storage_buckets.sql      ← bucket setup
│
├── dashboard/                   ← Static HTML, deployed to GitHub Pages
│   ├── index.html               ← project landing
│   ├── runs.html                ← live runs list
│   ├── models.html              ← all models
│   ├── model.html               ← single model detail (?id=v9.0)
│   ├── sims.html                ← all sim versions
│   ├── sim.html                 ← single sim detail
│   ├── leaderboard.html         ← top-N across current sim
│   ├── run.html                 ← single run detail (config, metrics, log)
│   └── lib/
│       ├── supabase.js          ← REST client
│       └── ui.js                ← shared helpers
│
├── tests/
│   ├── sim/                     ← Python sim unit tests
│   ├── parity/                  ← Python sim ↔ TS sim parity tests
│   ├── training/                ← encoder/decoder/agent tests
│   └── integration/             ← end-to-end: push → worker → storage → dashboard
│
└── game/                        ← TS + Phaser playable game
    ├── src/
    │   ├── sim/                 ← TS sim (must match Python sim, parity-tested)
    │   ├── rendering/           ← Phaser scenes
    │   ├── input/
    │   └── main.ts
    ├── vite.config.ts
    └── package.json
```

### Rationale for directory structure

- **`sim/` is its own package** so RL code can depend on it without pulling rendering code. Importable standalone.
- **`training/` never imports from `workers/` or `cli/`** — it's pure library code. Keeps it GPU-burstable without dragging in orchestration.
- **`workers/`, `cli/`, `modal_app/` are entry points** that compose `sim/` + `training/` + Supabase.
- **`dashboard/` is 100% static** — no build step, deployable to any static host.
- **`game/` is its own sub-project** with its own `package.json`. Doesn't share Python code at all. Parity is enforced via test artifacts, not imports.
- **`infra/` holds DB-as-code.** Schema migrations, stored procedures, RLS policies — all versioned alongside the app code.

---

## 7. Training pipeline — one run's lifecycle

```
1. Decision to run an experiment
   (by you, by a Modal cron job, by a Claude remote trigger, by another agent)
   
2. CLI push (or direct SQL):
   push_experiments.py --model v9.0 --config '{lr:0.0003,ent:0.01}' --budget 20m --seeds a,b,c
   → INSERT INTO runs (..., status='queued', model_id='v9.0', 
                       simulator_id='sim-v1.0', hyperparams={...})
   
3. Worker (anywhere: Mac/PC/Modal) polls:
   SELECT claim_next_run(project='mushroom-wars', machine=hostname);
   → atomic transition queued → running
   
4. Worker trains:
   - Load model architecture spec from models.layers
   - Initialize PyTorch net on GPU/MPS/CPU
   - Create vec env of 32-64 sims
   - Run PPO training for budget_ms wall time
   - Every N steps: snapshot weights, eval vs champion pool
   - On completion: eval full champion pool for final rate
   
5. Worker reports:
   - Upload weights.pt to Supabase Storage
   - Upload training log to Storage
   - UPDATE runs SET status='done', result={rate, perOpponent, ...},
                     weights_url=..., log_url=..., finished_at=now()
                     
6. Trigger re-ranks leaderboard; updates models.run_count, best_rate, etc.

7. Dashboard sees the new row in realtime (Supabase JS client subscription).
```

The worker is ~200 lines of Python. No orchestrator. No event bus. Just atomic SQL statements.

---

## 8. Simulator design — Python + numba + AsyncVectorEnv

### 8.1 Representation

- Entities as **parallel numpy ndarrays** (struct-of-arrays, not array-of-structs). Each field is a contiguous 1-D `ndarray` of shape `(MAX_BUILDING_SLOTS,)` or `(MAX_UNIT_GROUP_SLOTS,)` — the layout XLA/vmap consumes cleanly for the JAX backend.
- Buildings: `buildings_alive`, `buildings_owner`, `buildings_type`, `buildings_garrison`, `buildings_capacity`, `buildings_x`, `buildings_y` (all length MAX_BUILDING_SLOTS).
- Unit groups: `groups_alive`, `groups_owner`, `groups_src`, `groups_tgt`, `groups_count`, `groups_progress`, `groups_travel` (all length MAX_UNIT_GROUP_SLOTS).
- One game state = these ndarrays plus `travel_matrix`, `distance_matrix`, and scalars (`tick`, `phase`, `perf`).
- Structured-dtype access (`state.buildings["owner"]`, `state.unit_groups[mask]`) remains available via a proxy in `sim/state.py` for existing test/replay/script code; hot paths use the parallel ndarrays directly.

### 8.2 Hot paths compiled with numba

- Production tick (1 Hz): increments all buildings' garrison, bounded by capacity. `@njit`.
- Movement update: advances unit group progress, resolves arrivals. `@njit`.
- Combat resolution: computes damage when attacker arrives. `@njit`.
- Mask computation: which actions are valid right now. `@njit`.

Expected speedup: 50-100× over pure Python for these functions.

### 8.3 Parallelism

- **Gymnasium `AsyncVectorEnv`**: spawns 32-64 sim processes via `multiprocessing`.
- Each process runs one game independently.
- Observations stack into a `(n_envs, obs_size)` tensor each step.
- Trainer does one batched forward pass on GPU → n_envs actions.
- Actions dispatched back via pipe; each process steps its sim.

For PPO rollout collection, this is standard and gives big throughput gains.

### 8.4 Determinism for replays

Given (sim state, action, seed), the sim produces deterministic output. This is essential for replay:
- A `games.actions_url` stores the compact action log (~1-10 KB)
- Replay = re-run the sim from the initial state with that log, tick-by-tick
- Dashboard replay viewer fetches the action log, runs an in-browser sim (compiled Python-to-WASM via Pyodide, or a hand-maintained JS copy of the sim), renders step-by-step
- No need to store per-frame game state — it's reconstructable

### 8.5 No TS sim — Python is canonical

Earlier drafts of this doc contemplated a TS sim for a Phaser playable game, with parity tests. **Dropped.** One sim, one language, no drift problem. If we want a browser playable game later, we build it as a separate project using a Pyodide-compiled Python sim or a thin web UI over a Python sim HTTP server.

---

## 9. Neural architecture — v9.0 baseline

The v9.0 model is the first-class architecture for the rebuild. Same as the current v9.0, cleanly expressed in PyTorch.

### 9.1 Capacity — support big maps from day 1

Baseline uses `MAX_BUILDING_SLOTS = 32`, `MAX_UNIT_GROUP_SLOTS = 32` — well beyond Crossroads' 9 buildings. The unused slots are zero-padded and masked. This lets future levels with 20-30 buildings reuse the same trained model without a version bump.

- Observation: ~1150 floats (vs 681 in the old 14-slot design)
- Action space: 4 types × 32 × 32 + 1 = **4097**
- Empty-slot flag: `obs[slot_base + 0] = 0` signals "no building here"; mask zeros out its source/target eligibility

The cost of the extra slots is negligible on GPU. The benefit is no capacity-related version bumps as we expand levels.

### 9.2 Observation

- **Size:** ~1150 floats
- **Encoder:** `training/encoder.py`
- **Slot assignment:** static, by immutable building position (canonical-X → y → id). Building at slot K is the same building for the entire game.
- **Structure:**
  - 26 global features (timing, totals, threat, rolling stats)
  - 32 × 27 per-building features (ownership, garrison, capacity, position, incoming, capped flag, etc.)
  - 32 × 16 per-unit-group features (flight progress, source/target slot, speed, attack)
  - 15 action-history features
  - 4 reward-delta features
  - 18 rolling-stats features

### 9.3 Action space

- **Size:** 4097 (4 send percentages × 32 source × 32 target + 1 noop)
- **Decoder:** `training/decoder.py`, uses the same static slot map as encoder
- **Encoding:** `action = typeIdx * 1024 + src * 32 + tgt` for sends; `4096` for noop
- **Mask:** per-step validity mask (source exists AND owned AND not upgrading AND garrison ≥ 4; target exists AND target ≠ source)

### 9.4 Chained policy heads

- **Body:** `1150 → 128 → 128` (MLP, ReLU; bumped from 64 since we're on GPU now)
- **Source head:** `128 → 64 → 32` (picks source slot first)
- **Type head:** `(128 + 16_src_embed) → 64 → 5` (conditioned on source)
- **Target head:** `(128 + 16_src_embed) → 64 → 32` (conditioned on source)
- **Value head:** `128 → 64 → 1`
- **Embedding table:** 32-slot source embedding, 16-dim each

Sampling: source → type → target. Log-prob = sum of three component logs.
Total: ~300k params, ~500k FLOPs/forward. Still tiny for GPU. Plenty of room to scale body further if needed.

### 9.4 Scaling path

Once the baseline is working on Python + GPU, the obvious upgrades:

- **Body**: `681 → 256 → 256 → 256` with residual connections
- **Entity-attention body**: transformer over the 14 building slots + 15 unit-group slots, attend between them. Much more expressive; GPU-native.
- **Value network separation**: shared body with a deeper value head
- **Recurrent policy**: LSTM over the decision sequence to carry state across the game's 200 decisions

All of these are trivial in PyTorch. None are feasible in pure JS.

---

## 10. Training — PPO in PyTorch

### 10.1 Algorithm

- Clipped PPO (standard)
- GAE (λ=0.95, γ=0.99)
- Entropy bonus (0.01 default, swept)
- Value-function coefficient 0.5
- Clip range 0.2
- Adam optimizer, LR 0.0003 default
- Orthogonal initialization on the policy head (small init)
- Obs normalization (Welford running mean/std)

### 10.2 Rollout

- 64 parallel envs via AsyncVectorEnv
- Roll for N steps (e.g. 2048) OR N episodes, then update
- Compute GAE advantages
- 4 epochs of minibatch updates per rollout

### 10.3 Self-play

- Maintain a pool of frozen opponents (last N checkpoints + top-K leaderboard models)
- Each rollout episode: opponent sampled from pool (80% latest self, 20% random from pool)
- Mix league-play pattern (old and new opponents prevent forgetting)

### 10.4 Continue-training a run

Any completed run can be extended with more training. The schema supports this via `parent_run_id`, `is_continuation`, and `cumulative_budget_ms`.

Worker behavior when `parent_run_id` is set on a queued run:
1. Download parent's `weights.pt`, `optimizer.pt`, `obs_norm.pt` from Storage
2. Initialize net + optimizer + observation normalizer from that state
3. Train for this run's `budget_ms`
4. Save new state as its own row, link back via `parent_run_id`

CLI:
```bash
python cli/continue_training.py --parent <run_id> --budget 20m
# Inserts a new run row, status=queued, is_continuation=true, parent_run_id=<run_id>
```

On the model page, chains render as a single trajectory:
```
v9.0.abc-20m-a   →   cont1-20m  →   cont2-40m   →   cont3-30m
20m cumulative       40m            80m              110m
61.5% rate           74.8%          82.1%            85.3%
```

Champion discovery workflow: train for 20m, see something promising, extend for another 20m, decide again. Cheap iteration on a good seed.

### 10.5 Eval

- Every M minutes: pause training, run K games against each of the top 10 leaderboard models
- Final eval at training end: full champion pool, 30 games each
- `result.rate = (wins across all eval games) / (total eval games)`

### 10.5 GPU / device selection

- `torch.device('cuda')` on PC
- `torch.device('mps')` on Mac (Apple Silicon)
- `torch.device('cpu')` fallback (small models only)
- Auto-detected at worker start via `torch.cuda.is_available()` / `torch.backends.mps.is_available()`

---

## 10.6 Observability — comprehensive logging

Following Karpathy's "obsessive visualization" philosophy. Every training run captures three tiers of data:

### 10.6.1 Training scalars (per optimizer step, ~every 100 steps at high-freq, every 10 at diagnostic intervals)

Stored as a timeseries in `metrics.json` (uploaded to Storage at run completion, or streamed to a `run_metrics` table for realtime viewing).

**PPO loss components:**
- `policy_loss` — the clipped surrogate loss
- `value_loss` — MSE on returns
- `entropy_loss` — encourages exploration (negative mean entropy)
- `total_loss` — weighted sum

**PPO diagnostic:**
- `approx_kl` — KL divergence from rollout policy to current policy (detects big updates)
- `clip_fraction` — % of samples where PPO clipping activated (high = policy changing fast)
- `explained_variance` — 1 - Var[returns − V(s)] / Var[returns] (how good is the value function, 0=useless, 1=perfect)

**Optimizer health:**
- `grad_norm` — global L2 norm of the gradient before clipping
- `grad_clip_fraction` — % of updates where gradient was clipped
- `learning_rate` — current LR (if scheduled)

**Throughput:**
- `games_per_sec` — rollout throughput
- `sim_ms_per_decision` — sim efficiency (tags the (sim_id, model_id) for efficiency comparisons)
- `nn_ms_per_forward` — neural net time per decision
- `wall_ms_per_update` — optimizer step wall time

**Rollout summary (per batch):**
- `mean_reward`, `reward_std`
- `mean_episode_length`
- `wins`, `losses`, `draws` (percentage of games in this batch)

### 10.6.2 Distributions (per N rollouts, captured as histograms)

These are the "inspect for pathologies" dimensions. Stored as JSON in metrics.json.

**Action distributions** (exactly what we used to diagnose source-collapse):
- `action_type_hist` — [send25%, send50%, send75%, send100%, noop] fractions
- `source_slot_hist` — per-slot % of send actions
- `target_slot_hist` — per-slot % of target picks
- `noop_rate_overall` — how often is the agent just idle

**Policy confidence:**
- `entropy_per_decision` histogram — are decisions sharp or uncertain
- `logprob_of_chosen_action` histogram — how confident was each pick

**Economic signals:**
- `send_size_when_sending` distribution (mean, p25, p50, p75)
- `garrison_at_decision_time` histogram (am I always sending from full bases or idle ones)

### 10.6.3 Weight/activation inspection (per checkpoint, every few min)

Karpathy's advice applied to our specific net:

- **Source embedding cosine-similarity matrix** (14×14 or 32×32) — the diagnostic we hand-rolled in this session becomes automatic. Flags embedding collapse.
- **Per-layer weight L2 norms** — detect weight explosion / collapse
- **Per-layer dead-neuron fraction** — % of ReLU outputs = 0 for a random batch (catches dead neurons)
- **Body output L2 norm distribution** — body output statistics across a sample batch
- **Value prediction vs actual return scatterplot** — inspect value-function quality visually

### 10.6.4 Fixed-eval tracking

At run start, pick **N canonical mid-game states** (saved as a fixture file) — e.g.:
- "Opening move: you own 3 buildings, enemy owns 3"
- "Mid-game: you own 7, enemy owns 4, 5 units in flight"
- "Desperate defense: you own 2, enemy owns 10"

Every checkpoint, run the policy on these exact states and log the action distribution:
- `fixed_eval/state_0/action_probs` — all 4097 action probs on state 0
- `fixed_eval/state_0/value` — value prediction
- `fixed_eval/state_0/top_action` — argmax action

Plot how these change over training. Karpathy: *"how these predictions move will give you incredibly good intuition for how the training progresses."*

### 10.6.5 Sample episode capture

Every checkpoint, capture **one full game** with narrated log (same format as tournament game logs). Upload to Storage. Dashboard shows: "at step 10k this run played like X, at step 100k it plays like Y."

Cheap — one log file per checkpoint, ~100 KB each. 20 checkpoints = 2 MB per run. Kept with the run's other artifacts.

### 10.6.6 Narrated game logs — format

Every action-level log entry includes:

```
[ 12.0s] P1 decision:
         State summary: {my: 4 bldgs / 28 units, enemy: 3 bldgs / 22 units, in-flight: 2 mine, 1 theirs}
         Mask: 47 valid source-target pairs (out of 4097)
         Policy probs: top-5 = 
           send75 slot 3 → slot 7  (0.42)
           send50 slot 3 → slot 7  (0.21)
           send100 slot 3 → slot 7 (0.14)
           noop                    (0.08)
           send25 slot 3 → slot 7  (0.05)
         Sampled: send75 slot 3 → slot 7
         Value prediction: 0.67
         Logprob: -0.87
         Reason (derived from features): source is fullest owned building, target is weakest adjacent enemy
[ 12.0s] Execute: building-E (slot 3, garrison 12) sends 9 units to enemy-W (slot 7, garrison 4), ETA 2.1s
[ 14.1s] Combat at slot 7: 9 attackers vs 4 defenders → capture (P1 owns slot 7, 5 units garrison)
```

This is the **primary debugging artifact**. When you suspect "the agent is doing something stupid," you pull up this log and read the decisions. Mask + top-5 probs + value together explain *why* the agent made each choice.

### 10.6.7 Training-time live log vs post-hoc analysis

- **Live log** (streamed to Storage every N sec): condensed — scalars only, no distributions. Dashboard tails it.
- **Full metrics dump** (at run end): `metrics.json` with all scalars + distributions + fixed-eval data. Used for post-mortem plotting.
- **Narrated game samples** (per checkpoint): uploaded immediately.

---

## 11. Compute topology

### 11.1 Home workers (always on)

- **Mac**: launchd agent runs `python -m workers.worker` on boot. Claims jobs tagged for Mac or anywhere.
- **PC**: systemd service runs the same command. Claims jobs tagged for PC or anywhere.
- Both write to same Supabase. Both see same queue. Both publish to same Storage bucket.
- Crash or reboot = service restarts automatically.

### 11.2 Cloud burst (Modal)

For runs that need bigger GPU or higher parallelism than home has:

```python
@app.function(
    image=modal.Image.debian_slim().pip_install_from_requirements("requirements.txt"),
    gpu="a10g",  # or "a100-40gb" for larger models
    secrets=[modal.Secret.from_name("supabase")],
    timeout=3600,
)
def worker():
    from workers.worker import run_loop
    run_loop(max_jobs=1)  # one-shot; Modal spins up, claims one job, done

# Spawn N Modal workers on demand
@app.local_entrypoint()
def burst(n: int = 5):
    worker.spawn_many(count=n)
```

- Pay per second, no idle cost
- Can spawn 10+ workers in parallel for fast batch completion
- Secrets (Supabase URL + service key) injected via Modal's secret store
- Each worker runs exactly one job then exits, or loops if `max_jobs` unset

### 11.3 Future providers

Any provider that can run a Python container with env vars can be a worker:
- HuggingFace Jobs
- RunPod (spot GPUs, cheapest)
- Lambda Labs (mid-range)
- Vast.ai (spot marketplace)
- AWS EC2 / GCP GKE (enterprise)

Same worker code, same Supabase, same dashboard. Zero integration work.

---

## 12. Dashboard — static HTML, no framework

### 12.1 Two-tier approach

**Supabase Studio** (built-in, free): SQL editor, raw table views, storage browser, auth user management. Use for admin work and ad-hoc queries.

**Custom static dashboard** (`dashboard/` subfolder): vanilla HTML + TypeScript (compiled via esbuild) + Supabase JS client. No framework. No server. Just files.

Two hosting options, pick either (or both):
- **Local**: `python -m http.server 8000 --directory dashboard/` → `http://localhost:8000`. Only accessible from the Mac, but costs zero, needs no cloud setup.
- **Supabase Storage public bucket**: upload `dashboard/dist/` to a public bucket → accessible from anywhere at `https://<project>.supabase.co/storage/v1/object/public/dashboard/index.html`. No auth, no build pipeline, no deploy step beyond `supabase storage upload`.

No Next.js, no Vercel, no React. Just HTML with fetch calls to Supabase REST. Realtime updates via Supabase's JS client subscriptions. This is dramatically simpler than the earlier Next.js plan — we give up some UX polish but gain zero-infrastructure hosting.

No auth day 1 — public dashboard, data is non-sensitive.

### 12.2 Pages

| file | purpose |
|---|---|
| `index.html` | landing: active runs, today's completed, leaderboard snapshot, **progress-over-time chart** (prominent) |
| `runs.html` | full runs table with filters |
| `run.html?id=<uuid>` | single run: config, **training curves**, **action distribution over training**, per-opponent eval, log tail, download weights, "Continue training" button |
| `models.html` | all models, run counts, best rates, pin state |
| `model.html?id=v9.0` | single model: architecture, lineage, runs by (sim / machine / budget), **progress over time for this model**, pin toggle |
| `sims.html` | all sim versions, benchmarks, **efficiency over time** |
| `sim.html?id=sim-v1.0` | single sim: features, benchmark, models tested |
| `leaderboard.html` | top-N runs, sortable |
| `tournaments.html` | all head-to-head matches |
| `tournament.html?id=<uuid>` | single tournament: game table |
| `game.html?id=<uuid>` | **single game: full narrated log + replay viewer** — primary tool for debugging agent behavior |
| `progress.html` | big progress-over-time chart: best rate per week across all models, annotated with key milestones |

### 12.3 Stack

- Vanilla HTML + TypeScript (optional — can use plain JS)
- esbuild or Vite if we want modules/bundling (~1 line in package.json)
- [Supabase JS client](https://supabase.com/docs/reference/javascript/introduction) for REST + realtime subscriptions
- [Chart.js](https://www.chartjs.org) or [uplot](https://github.com/leeoniya/uPlot) for charts (lightweight, no framework)
- No CSS framework required; a single `dashboard.css` for shared styles
- Hosted on Supabase Storage or served locally

Total dashboard code: probably ~1500 lines of HTML/TS. Much less than a Next.js app.

### 12.4 Progress-over-time view (the "see improvement" page)

This is your primary success-tracking view. What it shows:

```
┌───────────────────────────────────────────────────────────────────────┐
│ PROGRESS OVER TIME                            [◦ all models  ◦ v9+ ] │
│                                                                       │
│ 100% ┤                                                    ← target   │
│      │                                               ▲               │
│  90% ┤                                         ▲▼▲ ▲▼                │
│      │                               ▲     ▲  ▼   ▼                 │
│  80% ┤                      ▲▼  ▲  ▼▲  ▼▼▲▼▲                        │
│      │                 ▲▼▲ ▼  ▼ ▼   ▼                                │
│  70% ┤         ▲     ▲▼▲                                             │
│      │        ▼ ▼   ▼                                                │
│  60% ┤   ▲▼▲ ▼                                                       │
│      │  ▼                                                            │
│  50% ┤                                                               │
│      └───┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴─────     │
│        Feb     Mar    Apr    May    Jun    Jul    Aug    Sep         │
│                                                                       │
│ Legend: ▲ = training run best rate   ▼ = re-eval of old model        │
│                                                                       │
│ Milestones:                                                           │
│  Mar 12  v7.0 chained heads         77% → 85%                         │
│  Apr 19  v7.2 sampling fix          85% → 91%                         │
│  Apr 21  v8.0 phase-lock                                              │
│  Apr 21  v9.0 static slots                                            │
│  Apr 23  v10.0 rebuild              ...                              │
└───────────────────────────────────────────────────────────────────────┘
```

Each point = best run-rate on that week. Hover a dot to see model details. Click to jump to the run. Milestones annotated automatically from new-model registrations.

All rates on this chart use the **current sim version** (old models are re-evaluated on the current sim when sim bumps happen — see §5.2). Fair comparison across time.

### 12.5 Cross-version "see models improve" tooling

New CLI tool to re-evaluate old models on the current sim:

```bash
python cli/reeval.py --all-pinned --sim current
# For each pinned run, re-eval against current champion pool on sim-v1.0
# Results go into a 'reeval_runs' shadow table so original run records stay intact
```

When a new sim version is registered, auto-queue re-evals of all pinned runs on the new sim. This keeps the progress-over-time chart honest across sim changes.

### 12.6 Tournament + replay viewer

(Unchanged from earlier design — see §4.4 and §12.2. `tournament.html` and `game.html`.)

The `game.html` replay viewer is less about watching and more about **debugging**:
- Narrated log with every decision, mask state, and value prediction
- Scrubber lets you step through tick-by-tick
- "Jump to next send" / "Jump to noop" filters
- Download the raw action log JSON for offline analysis

### 12.7 Plotting (optional, later)

If Chart.js / uPlot start feeling limited, Grafana over Postgres gives polished BI-style views. Skip until missing.

### 12.4 Tournament + replay viewer (the big new UX)

The flow for digging into model behavior:

1. You pick two models on `/tournaments/new` (dropdown lists top runs by rate)
2. Set N games, optional seed range, click Run
3. Dashboard creates a `matches` row, dispatches work to a worker
4. Worker plays N games, inserting `games` rows as each completes
5. `/tournaments/[id]` shows the table live as games finish:

```
┌────────────────────────────────────────────────────────────────────────┐
│ v9.0.abc-20m-x  vs  v7.2.111-lr3-b   (sim-v1.0, Crossroads)          │
│ 8 won / 2 lost / 0 draws · avg duration 78s                            │
├────────────────────────────────────────────────────────────────────────┤
│ # │ winner │ dur    │ a_caps │ b_caps │ a_units_prod │ actions │       │
│ 1 │ A      │ 52.7s  │ 12     │ 0      │ 487          │ 53      │ View  │
│ 2 │ A      │ 60.8s  │ 11     │ 2      │ 512          │ 61      │ View  │
│ 3 │ B      │ 302.0s │ 3      │ 14     │ 345          │ 301     │ View  │
│ …                                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

6. Click "View" on any game → `/games/[id]` opens the replay viewer:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Game 3 — B won at 5:02                                                 │
│                                                                        │
│ [REPLAY VIEWER: canvas rendering the game tick-by-tick]                │
│ ◀◀  ◀  ▶  ▶▶  ■   |██████████████████████░░░░░░░| 68%  03:24 / 05:02  │
│                                                                        │
│ GAME LOG (narrated)                                                    │
│  0.0s  Start — Crossroads, seed 14                                    │
│  0.0s  P1 (v9.0.abc) owns: capital, fort-NE, outpost-W                │
│  0.0s  P2 (v7.2.111) owns: capital, fort-SW, outpost-E                │
│  1.0s  P1: send 50% from capital → neutral-center (reaches 3.1s)      │
│  1.0s  P2: send 75% from capital → neutral-center (reaches 2.9s)      │
│  2.9s  P2 5 units arrive at neutral-center → captured by P2           │
│ 10.0s  P1: send 100% from fort-NE → P2's capital (reaches 15.2s)      │
│ 11.0s  P2: send 50% from fort-SW → P1's outpost-W                     │
│ 15.2s  P1 attack on P2's capital fails (defenders held)               │
│ …                                                                      │
└────────────────────────────────────────────────────────────────────────┘
```

The narrated log is pre-generated during the game and stored as `logs/games/<game-id>.log`. The replay viewer consumes `replays/<game-id>/actions.json` to reconstruct the state at any tick.

### 12.5 Plotting (optional, later)

Grafana connected to Postgres for fancy views:
- Sim efficiency over time (ticks/sec per version)
- Training throughput per worker per day
- Elo trajectories for the top 10 models

Skip until an actual need shows up. Next.js charts via recharts cover the day-to-day.

---

## 13. Remote compute / autonomous operation

### 13.1 Anthropic remote triggers

[claude.ai/v1/code/triggers](https://claude.ai/v1/code/triggers) schedules fresh agents on a cron. They can be configured to:
- Read the Supabase runs table (via REST with the anon key)
- Analyze results (win rates, per-opponent breakdowns)
- Decide the next experiments
- `git commit` new configs to the repo
- Or directly INSERT into the `runs` table to queue work

This is the "autonomous AI researcher" layer. Runs on Anthropic infra, no local dependency.

### 13.2 Cron jobs within this Claude Code session

`CronCreate` with `durable: true` schedules prompts to fire in this session. Use for:
- Periodic status checks while long batches run
- Prompting me to analyze results and suggest next steps
- Running nightly prune jobs if Modal-scheduled isn't enough

### 13.3 Local CronCreate vs Anthropic remote trigger

| need | use |
|---|---|
| Fires in this conversation; session must be open | CronCreate |
| Fires on Anthropic infra; works even if laptop is off | Remote trigger (RemoteTrigger API) |
| Runs inside a worker schedule (not Claude at all) | Supabase Edge Function on cron |

---

## 14. Migration plan — what we're keeping vs rebuilding

### 14.1 Transplanted code (port with minimal changes)

| old location | new location | notes |
|---|---|---|
| `src/simulation/` (TS) | `sim/` (Python) | Port the logic; same behavior. Keep TS copy at `game/src/sim/` for Phaser. |
| `src/training/NeuralNet.ts` | `training/net.py` | Rewrite in PyTorch; much simpler. |
| `src/training/ObservationEncoder_v3_2.ts` | `training/encoder.py` | Port to numpy. Drop the caching hacks. |
| `src/training/ActionDecoder_v71.ts` | `training/decoder.py` | Port. Static-slot resolution built-in. |
| `src/training/PPOAgent.ts` | `training/agent.py` + `training/trainer.py` | Rewrite on PyTorch. Much shorter. |
| `src/training/RollingStatsTracker.ts`, `RewardDeltaTracker.ts`, `ActionHistoryTracker.ts` | `training/trackers.py` | Port together, used by encoder. |
| `src/tournament/MixedSimulation.ts` | `training/evaluator.py` | Eval loop against champion pool. |

### 14.2 Legacy champions to import (as historical benchmarks)

Upload weights to Supabase Storage, register as legacy models:

| model | source | register as | mark |
|---|---|---|---|
| `v7.2.111-20m-lr3-b` | `public/models/leaderboard/e0c31dc_9133_v7.2.111-20m-lr3-b.json` | models.id=`v7.2` (legacy) | `keep_weights=true` |
| `3m-seed-b` | `public/models/leaderboard/25b735a_4562_3m-seed-b.json` | model.id=`v0.5-legacy` | `keep_weights=true` |
| `p2-h96-c` | leaderboard | legacy | `keep_weights=true` |
| `halve-lr-anti-turtle` | leaderboard | legacy | `keep_weights=true` |
| `b5-rule-c` (rule-based) | special | register as non-NN opponent | `keep_weights=true` |

These become the initial champion pool for v9.0+ eval. Everything else gets tarballed and archived.

### 14.3 Dropping

- All `src/training/ObservationEncoder{,.v2,.v3,.v3_1}.ts` (obsolete encoders)
- All `src/training/ActionDecoder{,_v05,_v06,_v07}.ts` (superseded by v71)
- `src/training/QAgent.ts`, `runTraining.ts`, `TrainingLoop.ts` (DQN-era dead code)
- `src/training/.worker-bundle.mjs`, `trained_weights.json`, `best_weights.json` (build artifacts)
- `autoresearch/` entire directory (orchestrator, queue, evaluator, rebench, elo, glicko — all replaced)
- `scripts/pc.sh`, `scripts/queue-drain-*.ts` (Redis/BullMQ infrastructure)
- `public/history.json`, `public/runs.json`, `public/rebench*.json`, `public/glicko-state.json` (moves to Postgres)
- `autoresearch/orchestrator-runs*/` (thousands of files; the important ones get migrated to Storage)

### 14.4 Archiving

- `tar -czf mushroom-wars-v1-final-<date>.tar.gz mushroom-wars/`
- Upload to Google Drive / external storage
- Commit a `LEGACY-README.md` to the new repo pointing at the archive
- Delete old repo from local disk

---

## 15. Development workflow

### 15.1 Adding a new experiment batch

```bash
# Define configs in a Python file or inline
python cli/push_experiments.py \
    --model v9.0 \
    --configs 'lr=0.0003 ent=0.01' 'lr=0.0003 ent=0.02' \
    --seeds a,b,c,d,e \
    --budget 20m \
    --sim-version auto \
    --tag hyperparam-sweep
```

That inserts rows into Supabase. Workers (home + Modal) pick them up. Dashboard shows progress.

### 15.2 Adding a new Model

```bash
# 1. Change code (new encoder, new architecture, etc.)
# 2. Update MODEL_VERSION constant in training/model_version.py
# 3. Commit to git
# 4. Register the model in the database
python cli/register_model.py v9.1
    # prompts for parent, what_changed, auto-introspects layers
```

Subsequent `push_experiments.py --model v9.1` works.

### 15.3 Adding a new Simulator version

```bash
# 1. Change sim code
# 2. Update SIM_VERSION constant in sim/sim_version.py
# 3. Commit to git
# 4. Register with auto-benchmark
python cli/register_sim.py sim-v1.1
    # prompts for parent, what_changed
    # runs 10 rule-vs-rule games to measure benchmark
    # captures git SHA
```

Subsequent runs auto-tag with the new sim version.

### 15.4 Bursting to Modal

```bash
# Spawn 5 GPU workers to chew through the backlog
modal run modal_app/run_worker.py --count 5
```

Workers pull work from the same Supabase queue as home machines. No difference to them.

---

## 16. Cost model

| resource | cost | why free/cheap |
|---|---|---|
| Supabase Postgres | $0 free tier (500 MB) | Far more than we need for years |
| Supabase Storage | $0 free tier (1 GB), then $0.021/GB | Weights are ~4 MB; 250 runs free, $5/mo for 10k runs |
| GitHub Pages | $0 | Static dashboard hosting |
| Modal GPU | $0.50-1.50/hr when running, $0 idle | Only pay while training |
| Home electricity | negligible | Mac + PC are already on |
| Claude remote triggers | included in Claude subscription | Used for autonomous decision-making |

Target: **stay in free tiers for normal development.** Only Modal costs real money, and only when we choose to burn cycles.

A heavy month (say 500 runs, half on Modal GPU):
- Supabase: $0 (still under 1 GB)
- Modal: 250 runs × 20 min × $0.50/hr ≈ **$42**

Call it **$50/mo ceiling** for serious training. Roughly what we'd pay for a streaming service.

---

## 17. Decisions finalized + deferred

### 17.1 Dashboard auth — ✅ skip

No auth. Public, read-only for now. If ever needed, Supabase Auth + magic-link is 15 minutes of work.

### 17.2 Replay capture — ✅ capture for tournaments, skip for training

- **Training runs**: no replay capture (would blow up Storage; can regenerate deterministically from weights if needed)
- **Tournament games**: capture always — that's the whole point of the tournament viewer. Action logs (~10 KB) + narrated logs (~100 KB) per game. 30-day default retention.

### 17.3 Multi-project — ✅ single-project, but keep the column

Schema has `project` column on every row (`project = 'mushroom-wars'`). Costs nothing. If a second project ever joins later, zero migration — just INSERT with a different slug. Dashboard doesn't show the filter unless more than one project exists.

### 17.4 Serving a trained AI for playable mode — ✅ later, on Mac

When we want this: a small FastAPI service runs on the Mac, loads a champion checkpoint, exposes an HTTP endpoint `/act`. Browser game (if built) calls it. Not a day-one concern.

### 17.5 Distributed training — ✅ not needed

One run = one GPU. PPO parallelizes across envs via AsyncVectorEnv, which is enough for mid-size models. If models ever outgrow a single GPU, PyTorch DDP exists. Revisit in 6+ months.

### 17.6 ~~Sim parity enforcement~~ — ✅ obsoleted by dropping TS sim

No more two-sim problem. Python is the only canonical sim. See §8.5.

### 17.7 Continue-training UX — ✅ supported via `parent_run_id`

See §10.4. Run has a "Continue" button on its detail page → opens a form for additional `budget_ms` → creates a new queued run with `parent_run_id` set.

### 17.8 Head-to-head tournament + replay viewer — ✅ supported via `matches` / `games` tables

See §4.4 and §12.4. CLI + dashboard both support launching tournaments. Replay viewer reconstructs games from stored action logs.

### 17.9 Capacity expansion (14 → 32 slots) — ✅ baked in from day 1

See §9.1. Model v10.0 (the first rebuild model) launches with MAX=32. Future level expansion is free until we exceed that.

### 17.10 Still deferred

- **Fancy plotting** (Grafana, training-curve comparisons across runs) — skip until a missing chart becomes painful.
- **Authenticated admin views** — skip until non-trivial access control needed.
- **Real-time training-curve streaming** (worker → dashboard while training) — skip; use periodic snapshots via Supabase realtime subscriptions on the `runs` table.
- **Big models** (transformer body, attention over entities) — the architecture supports this; swap in a new `net.py` and bump model version. Not planned for v10.0 baseline.

---

## 18. Build phases

### Phase 0 — Infra (day 1)

- Create Supabase project
- Apply `infra/schema.sql`, `infra/rpc.sql`
- Create Storage buckets
- Create Modal account, test one function
- Commit `ARCHITECTURE.md` (this file)

### Phase 1 — Sim port (days 1-2)

- Port `sim/` from TS to Python
- Write parity fixtures
- Write basic `gymnasium` env wrapper
- Benchmark sim throughput (target ≥ 10k game-seconds/sec single-core)
- Numba-optimize hot paths until happy

### Phase 2 — Training (day 3)

- PyTorch `net.py`, `agent.py`, `trainer.py`
- Small smoke training (1-minute budget) to verify end-to-end
- Compare learning curve to TS baseline to sanity-check

### Phase 3 — Supabase integration (day 3)

- `workers/worker.py` (Postgres-polling)
- `cli/register_model.py`, `cli/register_sim.py`, `cli/push_experiments.py`
- `cli/migrate_legacy.py` — import 5-10 champion weights

### Phase 4 — Dashboard (day 4)

- Static HTML pages pointing at Supabase REST
- Deploy to GitHub Pages
- Smoke test: end-to-end flow from push → worker → Storage → dashboard

### Phase 5 — Cloud burst (day 4)

- `modal_app/` — wrap worker in Modal function
- Test: spawn 3 Modal workers, verify they claim jobs and report results

### Phase 6 — Home workers (day 5)

- `workers/launchd/` for Mac, `workers/systemd/` for PC
- Register services so they survive reboots
- Smoke test: crash one, confirm auto-restart claims next job

### Phase 7 — TS playable game (day 5)

- Port TS sim as needed
- Wire Phaser game to use it
- Ensure parity tests pass

---

## 19. Success criteria for the rebuild

The rebuild is "done" when all of these are true:

1. A training run can be pushed from Anywhere (local CLI, phone, remote trigger, Modal) and picked up by any worker anywhere.
2. Both Mac + PC run as unattended services, auto-claiming jobs.
3. Modal can burst to N GPU workers with a single command.
4. `/models`, `/model?id=v9.0`, `/runs`, `/run?id=...`, `/sims`, `/sim?id=...`, `/leaderboard` all load live from Supabase.
5. A v9.0 run at 20 min matches or beats the original TS v9.0 sanity training in quality.
6. A v9.0 run on GPU matches CPU baseline in quality and is at least 3× faster end-to-end.
7. Killing the Mac mid-run and letting PC pick up works seamlessly (idempotent claim).
8. A sim change + `register-sim` step propagates correctly; subsequent runs tag with new sim version; dashboard shows sim breakdown on model pages.
9. Nightly prune keeps Storage usage below the free-tier ceiling.
10. All legacy code is deleted or archived. `mushroom-wars-v2/` is the only active codebase.

---

*Last updated: 2026-04-21. Author: Paul + Claude. Status: design, not yet implemented.*
