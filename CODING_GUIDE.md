# Coding Guide

The rules of the road for this project. Short, enforceable, no filler.
Read once, bookmark, re-read when you catch yourself drifting.

---

## 1. Core principles

1. **Think before coding.** If requirements have multiple interpretations, list them. If a simpler path exists, say so. Ask when unclear.
2. **Simplicity first.** Minimum code that solves the problem. No speculative flexibility. Would a senior say this is overcomplicated? Then it is.
3. **Surgical changes.** Touch only what you must. Don't refactor working code. Match existing style.
4. **Goal-driven execution.** Every task has a verifiable success criterion. Bug fix = write the failing test first.
5. **No hidden state that changes between calls.** Functions should behave the same way when called with the same inputs. 

---

## 2. Modularity & structure

- **One module, one responsibility.** `encoder.py` encodes. `trainer.py` trains. No multi-purpose files.
- **Library code doesn't import orchestration code.** `sim/` and `training/` never reach into `workers/`, `cli/`, or `modal_app/`. Entry points compose libraries; libraries don't know about entry points.
- **No circular imports.** If you hit one, redesign — don't paper over.
- `**__init__.py` exposes the public surface only.** Internal helpers get a leading underscore.
- **Delete dead code on sight.** Not "just in case." Not "maybe later." Git keeps it if you change your mind.

---

## 3. Configuration — no hard-coded values

- **YAML configs live in `configs/`** — actual files: `training_levels.yaml` (curriculum + level mix), `karpathy_loop.yaml` (sweep axes), `bench_eval.yaml` (champion archive + promo gates), `worker.yaml` (worker tunables). Splitting by lifecycle, not by domain.
- **Game constants live in `sim/config.py`.** `MAX_BUILDING_SLOTS`, `SEND_PERCENTAGES`, `DECISION_INTERVAL_TICKS`, the reward tables — Python module, not YAML, because they're shape-defining and any change is a model-version bump (referenced from sim+training+JAX simultaneously). YAML is for *runtime* tunables, not invariants.
- `**.env` for secrets only.** Not for game balance, not for hyperparams.
- **Every run's config is stored in Postgres** (`runs.hyperparams` jsonb). Reproducing any past run = re-read its stored config.
- **Defaults work out of the box.** Clone repo → `python scripts/smoke_train.py` → training runs. No flag-flipping dance.

---

## 4. Testing

- **Unit tests next to code.** `sim/production.py` ↔ `sim/tests/test_production.py`. No far-away monorepo test tree.
- **Every sim system has unit tests.** Production, movement, combat, mask computation. *These are exactly where the sneaky bugs live.*
- **One integration test per major flow.** A full scripted game with a deterministic seed. If it changes, git flags it.
- **Encoder golden tests.** Hand-designed state → expected obs vector. Silent encoder drift is a disaster (ask me how I know).
- **Pre-commit runs the fast subset.** `pytest -m fast` in <10 seconds. Full suite (slow sim benchmarks, training smoke) runs in CI.
- **CI blocks merges on test failure.** GitHub Actions on every push.
- **Property tests for invariants.** Use `hypothesis` for "garrison is never negative" / "total units conserved during flight." Catches edge cases you didn't think of.

---

## 5. Documentation — the sync rule

- **Docstring every public function.** One line: what it does, what it assumes, what it guarantees. No restating the code.
- `**ARCHITECTURE.md` is the design doc.** Any design decision that affects the shape of the project goes there.
- `**README.md` per package** (short, ~20 lines) — what this package is for, what it's NOT for, how to run its tests.
- **Link docs from code and vice versa.** Public entry points mention the relevant section of `ARCHITECTURE.md`. `ARCHITECTURE.md` links back to actual file paths.
- **🔴 DOCUMENTATION SYNC RULE.** Before committing: for every file you changed, skim the docs that describe that area. If the change makes the docs wrong, update the docs in the same commit. Pre-commit hook prompts you; CI checks for obvious drift (e.g. referenced files no longer exist).
- **Kill comment rot.** If a comment contradicts the code, delete the comment. Don't preserve lies.

---

## 6. Working with AI (Karpathy-style)

These apply whether you're driving Claude Code, Copilot, Cursor, or any agent:

- **Make it surface assumptions.** "Before you write code, list what you're assuming about X."
- **Prefer surgical diffs.** If AI wants to rewrite large sections it wasn't asked to, push back.
- **Reject speculative code.** If AI added a feature you didn't ask for, delete it. If it added error handling for impossible states, delete that too.
- **Success criteria up front.** "Fix the bug" → "Write a test that reproduces it, then make it pass." Verifiable loop, not vibes.
- **Read every line before committing.** Non-negotiable. AI code that nobody understands = bugs that nobody can fix.
- **One change at a time.** Don't let AI bundle "fix this" with "refactor that" in one PR.

---

## 7. Tooling

- `**ruff` for format + lint.** One config, fast, replaces black+flake8+isort.
- `**pyright` for type checking.** Fast, catches real bugs.
- `**pre-commit` runs:** ruff → pyright → fast tests → doc-sync check. On every `git commit`. Blocks if anything fails.
- `**uv` for dep management.** Lockfile via `uv.lock`; deterministic installs forever.
- **Don't fight the tools.** If ruff flags something, fix it or `# noqa: <rule>` with a comment explaining why.

---

## 8. Types & data

- **Type-annotate every public function.** Private helpers optional.
- `**pydantic` for config, `dataclass(frozen=True)` for values, `TypedDict` for JSON shapes.** No bare dicts as structured state.
- **Avoid `Any`.** If you must use it, comment why.

---

## 9. Logging & errors

- `**logging` module, not `print()`.** One logger per module: `logger = logging.getLogger(__name__)`.
- **Log levels matter.** DEBUG = internals, INFO = events, WARNING = recoverable, ERROR = failure. Be consistent.
- **Raise specific exceptions.** `ValueError`, `KeyError`, custom domain exceptions. Never bare `raise Exception("...")`.
- **Fail fast at module boundaries.** Validate inputs when they enter. Don't propagate garbage.
- **Don't silently swallow exceptions.** Catch = handle meaningfully OR re-raise.

---

## 10. Reproducibility

- **Seed everything.** numpy, torch, python `random`, sim RNG. Record the seed on the `runs` table.
- **Deterministic tests.** Any test involving randomness fixes its seed.
- **Lockfile pinned.** `uv.lock` in git. Running the code in a year produces the same behavior.
- **Same seed + same config + same sim version ⇒ same result.** No wall-clock, PID, or time-of-day state leaking in.

---

## 11. Git

- **Atomic commits.** One logical change per commit.
- **Subject = what, body = why.** Git already shows what changed; the message explains the reasoning.
- **Branch per feature/fix.** Merge to main via PR, even solo.
- **Small PRs.** <500 lines diff preferred. Big PRs get rubber-stamped and ship bugs.
- **Never force-push to main.** Feature branches only.
- **Never commit secrets.** If one slips in: rotate immediately, even after revert.

---

## 12. Project-specific rules (lessons from v1)

These are the lessons we paid for. Don't repeat them.

- **Slot identity is stable within a game.** No re-sorting entities mid-game. Static map at game start, used everywhere.
- **Decision clocks preserve residual.** `timer -= INTERVAL`, never `timer = 0`. Phase drift silently corrupted training for weeks.
- **Version bumps are recorded.** Model or sim change → database row with `what_changed` filled in. Git history is not enough.
- **One canonical sim.** No parallel implementations to keep in sync.
- **Model weights are complete artifacts.** State dict + optimizer state + obs-norm stats stored together — required for resume-training.
- **Log distributions, not just means.** The source-collapse bug showed up in a histogram, not in the mean.

---

## 13. Checklist before pushing

```
[ ] Tests pass locally (`pytest -m fast`)
[ ] Ruff clean (`ruff check . && ruff format --check .`)
[ ] Pyright clean (`pyright`)
[ ] Docs updated for any touched area (ARCHITECTURE.md, README.md, docstrings)
[ ] No secrets committed (`git diff --cached` review)
[ ] Commit message explains the WHY
```

If you did AI-assisted work: also confirm you've read every line of generated code.

---

*Last updated: 2026-04-21. This guide is enforceable: pre-commit hooks check most of it. Violating a rule on purpose needs a comment explaining why.*