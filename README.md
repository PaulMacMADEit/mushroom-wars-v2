# mushroom-wars-v2

RL training platform for Mushroom Wars. Python + PyTorch (v2 rebuild).
Supabase-backed runs tracking, Modal-burstable GPU training, static dashboard.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Status

Design complete. Schema + code not yet scaffolded — start at Phase 0 in ARCHITECTURE.md §18.

## Sim backend

The simulator has two interchangeable backends:

- **`jax`** (default as of sim-v1.2): `sim/engine_jax.py` — `jax.jit` + `jax.vmap` over a `StateJax` pytree. One XLA dispatch per tick regardless of batch size. Scales to `n_envs≥1024` on CUDA.
- **`numpy`** (reference): `sim/engine.py` — the numpy engine that every sim test asserts against. Always available; the parity harness holds both backends byte-identical.

Switch at runtime:

```bash
# force numpy (rollback path for JAX issues)
SIM_BACKEND=numpy python cli/smoke_train.py

# per-run pinning via push_experiments
python cli/push_experiments.py --model v9.0 --sim sim-v1.2 \
    --label jax-sweep --budget 1200 --sim-backend jax
```

On PaulLinux (RTX 3070) the worker also respects `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` so JAX doesn't fight PyTorch for VRAM. See [JAX_PORT_PLAN.md](./JAX_PORT_PLAN.md) §3.6.

## Quick links

- [ARCHITECTURE.md](./ARCHITECTURE.md) — every design decision, with rationale
- [JAX_PORT_PLAN.md](./JAX_PORT_PLAN.md) — the dual-backend port plan, phase-by-phase
- [.env.example](./.env.example) — copy to `.env`, fill in Supabase keys
- [Supabase project](https://supabase.com/dashboard/project/lwkljcyspyqklyoagnmo)
