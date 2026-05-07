# mushroom-wars-v2

RL training platform for Mushroom Wars. Python + PyTorch policy net + JAX sim.
Self-play with PFSP archive rotation. Supabase-backed run tracking + champion
storage. Static GitHub Pages dashboard at
[paulmacmadeit.github.io/mushroom-wars-v2](https://paulmacmadeit.github.io/mushroom-wars-v2/).

## Status

Operational. b6 (`phase1_full_mix`, 2026-04-27) is the v2 baseline champion;
v13 (chain reorder + head capacity) is the active line — see
[V13_PLAN.md](./V13_PLAN.md). The Karpathy hyperparameter loop runs hourly
with continuation chains — see [KARPATHY_LOG.md](./KARPATHY_LOG.md).

## Sim backend

Two interchangeable backends:

- **`jax`** (default): `sim/engine_jax.py` — `jax.jit` + `jax.vmap` over
  `StateJax`. One XLA dispatch per tick regardless of batch. Scales to
  `n_envs ≥ 1024` on CUDA.
- **`numpy`** (reference): `sim/engine.py` — every sim test asserts against
  this. Parity harness keeps both backends byte-identical.

Switch with `SIM_BACKEND=numpy` for the rollback path.

## Running things

```bash
# Smoke-test training (10 updates, no upload)
python scripts/smoke_train.py

# Queue a run (writes to Supabase; PaulLinux worker picks it up)
python scripts/queue_v13_validation.py

# Watch the worker
python scripts/watch_run.py <run_id_prefix>

# Bench a champion archive entry
python cli/rebench.py --run <run_id_prefix>

# Play live in your browser vs any trained champion
python scripts/play_live.py     # then open http://localhost:8765
```

The dashboard's [Play Live tab](https://paulmacmadeit.github.io/mushroom-wars-v2/play-live.html)
shows the launch instructions. The browser-side opponent dropdown loads the
full champion list from Supabase; switching mid-session swaps the model and
starts a new game.

## Layout

| Dir | What |
|---|---|
| `sim/` | Game engine (numpy + JAX backends, parity-tested) |
| `training/` | PPO trainer, ActorCritic nets, encoders, opponent pool |
| `workers/` | Job claimer, match runner, bench-eval, telemetry |
| `cli/` | DB helpers, run management, migrations |
| `scripts/` | Entry points: queue scripts, benchmarks, eval, play, diagnose |
| `dashboard/` | Static GitHub Pages site (vanilla HTML/JS + Supabase) |
| `configs/` | YAML: training_levels, bench_eval, worker, karpathy_loop |
| `tests/` | pytest — sim, encoder parity, env, rewards, JAX parity |
| `infra/` | Supabase SQL: schema, RLS, RPCs |
| `docs/archive/` | Historical plan docs (FUSED_ROLLOUT, JAX_PORT, PHASE_G, CURRICULUM) — all shipped |

## Live plan docs

- [V13_PLAN.md](./V13_PLAN.md) — current architecture line (chain reorder + head capacity)
- [KARPATHY_LOG.md](./KARPATHY_LOG.md) — running log of hyperparam sweeps
- [ARCHITECTURE.md](./ARCHITECTURE.md) — original design doc (partly historical; see header note)
- [CODING_GUIDE.md](./CODING_GUIDE.md) — coding rules of the road

## Quick links

- [Supabase project](https://supabase.com/dashboard/project/lwkljcyspyqklyoagnmo)
- [.env.example](./.env.example) — copy to `.env`, fill in Supabase keys
