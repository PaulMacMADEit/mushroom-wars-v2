#!/bin/bash
# Daily mushroom-wars-v2 batch scheduler — Claude Code in headless mode.
#
# Invoked by mushroom-scheduler.timer at 11:00 Pacific each day. Reads recent
# Supabase state, designs the next batch (b{N+1}), generates the queue script,
# commits + pushes, and runs it to insert the rows into Supabase. Worker
# (mushroom-worker.service) chews through the queue continuously.
#
# Headless flags rationale:
#   --dangerously-skip-permissions: no interactive prompts can stall the run.
#   --print: one-shot, non-interactive (exits when prompt completes).
#
# Required setup (one-time on PaulLinux): `claude setup-token` (NOT
# `auth login`) to seed the headless OAuth token.

set -uo pipefail

REPO=~/Projects/Personal/games/mushroom-wars-v2
LOG_DIR=~/.local/log/mushroom-wars
LOG_FILE="$LOG_DIR/scheduler.log"

mkdir -p "$LOG_DIR"
cd "$REPO"

PROMPT='You are the daily Mushroom Wars v2 training batch scheduler running on PaulLinux. Run `date` first if you need the time.

## Your job

Decide and ship the next training batch (b{N+1}) for the mushroom-wars-v2 RL project. Push to main and queue runs into Supabase. End-to-end automation — no PR, no manual handoff.

## Steps

1. **Inventory state.**
   - Activate the venv: `source .venv/bin/activate`. All python commands below assume this venv.
   - Read `docs/runs_summary.md` for the last-24h Supabase results (auto-written daily by `scripts/write_runs_summary.py`).
   - Read `PHASE_G_PLAN.md` and `docs/bench/phase_g_paullinux.txt` if context on perf is needed.
   - `ls scripts/queue_b*.py | sort -V` — find the highest existing batch number; new one = N+1.
   - `git log --oneline -10` for recent momentum.

2. **Decide the next batch.**
   - **Bias toward 60+ minute runs.** b3 confirmed 30 min = coin-flip parity vs strong neural opps; 60 min = 0.67 win-rate.
   - **Pick the strongest opponent**: the highest-update-count done run from runs_summary.md with rate >= 0.55 *trained against a neural opponent* (NOT random_legal). Re-pick each fire — promote the latest winner. The b3 endurance run (id 0385b326, rate 0.672, 20 updates, 60 min vs neural) was the right pick for b4. If a b4 ceiling probe lands at higher rate vs that opponent, b5 should use it.
   - **Default cfg first.** lr=3e-4, entropy=0.01, gamma=0.99, clip=0.2, K=4, n_envs=1024, rollout=64, level random_8_16. Only test cfg variants if evidence specifically calls for them at this run length.
   - **Total budget ~5-6h.** Layout suggestion (adjust based on results): 4 × 60min consistency seeds + 1 × 90min ceiling probe + optionally 1 × 60min cfg variant only if data supports testing one. Skip variants by default — b3 proved time matters, not config.

3. **Generate the queue script.** Copy `scripts/queue_b{N}.py` → `scripts/queue_b{N+1}.py`. Adjust the docstring (this batch, this question), the `_build_batch` function (new layout), the hardcoded opponent constant if applicable, and the label prefix. Match the b4 pattern.

4. **Commit + push.** `git -c user.name="PaulMacMADEit" -c user.email="paul@madeit.tech" commit -m "b{N+1}: <one-line summary>"`. Push to origin main. Never use `Co-Authored-By: Claude` — Paul standing rule.

5. **Queue immediately.** `python scripts/queue_b{N+1}.py` — inserts the rows into Supabase. The worker (`mushroom-worker.service`) picks them up.

6. **Update state file.** `echo {N+1} > ~/.local/state/mushroom-wars/last_run_batch` so the legacy pull-batch timer would skip if accidentally re-enabled.

7. **Output a brief report** at the end (under 200 words): batch design summary, opponent chosen + why, commit sha, runs queued, anomalies if any.

## Project context (b1-b4 findings, baked in)

- **b1+b2** (random_legal): K=4 default cfg is the sweet spot. Default cfg (lr=3e-4, entropy=0.01, gamma=0.99, clip=0.2) is well-tuned. lr=1e-4 hurts (40 vs 22 updates), clip=0.1 hurts (41 updates). Random_legal trivially exhausted (~22 updates to 0.99).
- **b3** (vs neural opp da2205e1, 11 runs): 30-min runs cluster at coin-flip parity (mean 0.498, range 4.7pp). Cfg variants do not separate at 30 min. 60-min endurance hit 0.672 (rate >0.55 → eligible as b4 opponent).
- **b4** (vs neural b3-endurance, 5 runs): IN-FLIGHT as of this session. Will land before next fire. Read its results from runs_summary.md to decide b5.
- **Phase G shipped**: JAX mask + on-device action pack; pack+encode 11% → 0.2%. Still kernel-launch-bound (SM ~5%).
- **n_envs=1024 production default**; 4096 has 12% throughput headroom but untested in training.

## Stop conditions / sanity

- If `runs_summary.md` is missing or stale (>26h old), abort with a warning — dont blindly schedule against unknown state.
- If no runs in the last 24h have rate >= 0.55 against a neural opp AND no clear progression is visible, fall back to a plain default-cfg variance batch vs whatever the latest credible neural opponent is. Never queue zero runs.
- If git push fails: do not run `python scripts/queue_b{N+1}.py` (would create runs against a script not on remote — pull-batch timer cannot find it next time). Log the error and abort.
- If 5+ runs are still in `running` state from prior batches (worker is deadlocked or backed up), abort and log — let the queue drain first.

Now do it.'

{
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] scheduler fire"
  echo "============================================================"

  # Activate the venv so claude inherits it. Use absolute path to claude
  # since systemd user services don't always inherit ~/.local/bin in PATH.
  source .venv/bin/activate
  CLAUDE_BIN="$HOME/.local/bin/claude"

  "$CLAUDE_BIN" --dangerously-skip-permissions --print "$PROMPT"
  RC=$?

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] claude exit=$RC"
} >> "$LOG_FILE" 2>&1

# Tee the latest run to systemd journal too.
tail -n 80 "$LOG_FILE"
