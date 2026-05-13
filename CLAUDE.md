# Mushroom Wars v2 — Claude operating rules

<!-- PROJECT-CLAUDE-CANARY-v1 — loader health signal for the session-start greeting score -->

RL training platform: Python + PyTorch policy net + JAX sim. Self-play with PFSP archive rotation, Supabase run tracking, static GitHub Pages dashboard.

## Read first

- [README.md](./README.md) — status, what's operational, how to run things
- [V13_PLAN.md](./V13_PLAN.md) — current active line of work (chain reorder + head capacity)
- [KARPATHY_LOG.md](./KARPATHY_LOG.md) — hourly hyperparameter loop log; check before proposing new sweeps
- [CODING_GUIDE.md](./CODING_GUIDE.md) — the rules of the road; defaults to simplicity-first, surgical changes, no speculative flexibility
- [ARCHITECTURE.md](./ARCHITECTURE.md) — original design doc; status-noted (some decisions didn't land — see top of file)

## Path-scoped rules

`.claude/rules/` contains training-discipline rules that auto-load when editing `training/`, `workers/`, `cli/`, `configs/`, `KARPATHY_LOG.md`, `V13_PLAN.md`, etc. The most load-bearing:

- **One variable at a time** when A/B testing across infra changes. Multi-variable swings produce uninterpretable results.
- **Major version bump** (v5.x → v6.0) on any change that invalidates existing checkpoints (new obs/action shape, new topology, new action semantics). Minor bumps only for sampling order, aux losses, hyperparam tuning within the same I/O contract.

## Session startup

- Skim [README.md](./README.md) for what's operational right now.
- If the user references a champion (`b6`, etc.) or version (`v13`, etc.), check `KARPATHY_LOG.md` and `V13_PLAN.md` before answering — these change frequently.
- Sim backend: `jax` is default (`SIM_BACKEND=jax`); `numpy` is the reference for parity checks (every sim test asserts against numpy). Don't change which is default without an explicit ask.

## Repository

`PaulMacMADEit/mushroom-wars-v2` — push to main. Long-running training state lives in Supabase, not git.
