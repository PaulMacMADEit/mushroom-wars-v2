# Karpathy Loop — hyperparam sweep log

**Evaluation system change 2026-04-27.** Replaced random_legal-anchored auto-rate
with the champion archive + `bench_eval` system. Old sweeps below used "vs
baseline (random_legal)" + "vs top-5" as axes — those axes no longer exist.
New format columns:

- **PFSP** — `runs.pfsp_weight`. Harmonic mean of `1 - |wr - 0.5|` across the
  bench vector. Peaks at 50% wr (most informative); falls toward 0 at wr=0/1.
- **vs champ** — win-rate over 16 games vs the most-recent champion at the
  time the run finished. ≥60% promotes the run to the archive.
- **promoted?** — Y/N — did this run get archived as a new champion.
- **arch sweep avg** — mean of the bench_vector entries (sweep across all
  champions, 8 games each). Coarse "general strength" reading.

Baseline config: `v9.0-full`, level mix `random_4_8 / 6_10 / 8_16 / 16_24`
(reward_v13), self-play, `n_envs=32`, `rollout_steps=128`,
`snapshot_every=10`, `lr=3e-4`, `entropy_coef=0.01`, budget 1h. PFSP
opponent sampling in trainer **not yet implemented** — current self-play
still uses leaderboard-Elo opponent picks. Treat sweeps below the divider
as run under partial system.

---

## Sweeps run under new bench_eval (2026-04-27 →)

### Loop fire 1 — 2026-04-27 22:56 PT — observe-only (queue full)

Queue depth 12 (3 running, 9 cron-agent queued ahead). Skipped queueing
per loop step-3 stop rule.

Observations from last 90 min (5 finished runs, all `cron-260428-0407-
phase2_selfplay-short-NN`):

| run | training rate | PFSP | bv n | promoted? | Elo |
|---|---|---|---|---|---|
| short-03 (072fe893) | 0.052 | 0.803 | 8 | **Y** | 1034 |
| short-02 (5d177b07) | 0.050 | 0.818 | 7 | **Y** | 1009 |
| short-01 (8c1fc31f) | 0.050 | 0.869 | 6 | **Y** | 1003 |
| short-00 (2ee773c4) | 0.046 | 0.869 | 6 | **Y** | 1005 |
| med-03 (491fdac9)   | 0.101 | 0.782 | 6 | **Y** | 1023 |

**Champion archive: 9 entries, all era v9, all from the last 6 hours.**

**Two flags:**
1. *Every* recent run promoted. Either the 60% gate is too lax against the
   tail-champion, or self-play training is genuinely stair-stepping. After
   the queue drains, run a 16-game match between champion #1 (oldest
   archived, `d53a2871` from 01:13) and champion #9 (newest, `02651ce3`
   from 05:47) — if newest wins ≥75%, gate is calibrated; if ≤55%, the
   archive is full of near-duplicates and PROMO_THRESHOLD should rise.
2. *Training rate is 0.05-0.10* — meaning the agent only wins 5-10% of
   training games. Expected under PFSP-weighted self-play (matched against
   strong-but-informative opponents) but worth confirming once a non-cron
   karp- run finishes — Karpathy axes will be fixed-opponent which should
   show a higher rate.

Next fire: re-check queue depth; if drained <5 runs, queue first axis
sweep (entropy_coef: 0.003 / 0.01 / 0.03 against current champion).

### Loop fire 1.5 — 2026-04-27 23:30 PT — clear + queue entropy sweep

Paul: "Karpathy runs take precedence over every other run. Clear out runs
if they are not karpathy loop." Acted:

- **Stopped + disabled `mushroom-cron.timer` on PaulLinux.** The 3h cron
  pulse was filling the queue ahead of karp- runs every 3h. Will re-enable
  when the loop is wound down. The worker daemon (`mushroom-worker`) stays
  active — that's what runs the karp- sweeps.
- **Marked 3 stale "running" runs as `failed`** (started 2-9h ago, all
  >5x past their 30-min budget). Worker had crashed/restarted without
  resetting their state.
- **Marked 7 queued non-karp runs as `discarded`** (cron-agent backlog).
- Left genuinely-running `cdcc0826` (started 06:22 UTC, 7min into
  30-min budget) alone — finishes ~23:52 PT.

**Queued first karpathy sweep — entropy_coef:**

| label | entropy_coef | other |
|---|---|---|
| karp-260427-2330-entropy-lo  | 0.003 | self_play=true, lb_bias=0.3, pfsp source |
| karp-260427-2330-entropy-mid | 0.01  | (baseline) |
| karp-260427-2330-entropy-hi  | 0.03  | |

Other cfg is the b6 champion's: `gamma=0.97, n_envs=1024, rollout_steps=64,
fused_rollout=true, action_repeat=2, level_mix=random_4_8/6_10/8_16/16_24
(reward_v13), sim_backend=jax`. Budget 20min each = 60 min total. All run
self-play with the new PFSP champion-archive sampling
(`leaderboard_source='pfsp'`, `leaderboard_bias=0.3` → ~30% of envs draw
from the 9-deep champion archive, weighted by PFSP).

Sweep finishes around 00:55 PT (after the live cron run drains).

---

## Archive (legacy axes — pre-2026-04-27, random_legal anchor)

The sweeps below used the old `_auto_rate_run` 3-match-vs-random and a
"top-5 by Elo" loop. After the 2026-04-27 rebench, that Elo was shown to be
high-variance and most of the "wins" were measurement noise. Findings here
are not directly comparable to anything under the new system. Keep for
design-history reference; do not cite as priors for new sweeps.

### Sweep 1 — entropy_coef @ 15 min

| run | entropy_coef | vs baseline (20 games) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-ent-003 | 0.003 | 70% (14/20) | 2% |
| kar-ent-010 | 0.01 (baseline) | **90% (18/20)** | **8%** |
| kar-ent-030 | 0.03 | 85% (17/20) | 4% |

**Finding (legacy):** baseline `entropy_coef=0.01` wins on both axes. Too little
exploration (0.003) hurts the most; too much (0.03) is chaotic but not broken.
Caveat: 15 min is short; low entropy might catch up at longer budgets.

### Sweep 2 — lr @ 15 min

| run | lr | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-lr-1e4 | 1e-4 | 50% (10/20) | 4% |
| kar-lr-3e4 | 3e-4 (baseline) | 75% (15/20) | 8% |
| kar-lr-1e3 | 1e-3 | **85% (17/20)** | **10%** |

**Finding (legacy):** **higher lr (1e-3) wins** at 15-min budgets. Low lr (1e-4)
barely beats random — learning too slow for the budget. Seed noise is real:
prior kar-ent-010 at baseline config hit 90%, kar-lr-3e4 at same config hit
75%.

### Sweep 3 — rollout_steps @ 15 min

| run | rollout_steps | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-rs-64 | 64 | 85% (17/20) | 12% |
| kar-rs-128 | 128 (baseline) | 70% (14/20) | 12% |
| kar-rs-256 | 256 | **90% (18/20)** | 10% |

**Finding (legacy):** mixed / within noise. `rollout_steps` isn't a major knob
at 15 min. 256 slightly better vs baseline; 64 ties on top-5. Baseline 128
unexpectedly weakest vs baseline — likely seed variance.

### Sweep 4 — snapshot_every @ 15 min

| run | snapshot_every | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-snap-5 | 5 (freshest) | 85% (17/20) | 4% |
| kar-snap-10 | 10 (baseline) | 55% (11/20) | 6% |
| kar-snap-20 | 20 (stalest) | **90% (18/20)** | **20%** |

**Finding (legacy):** **snap=20 wins decisively**, especially on vs-top-5 (20%
— 3-5× the other two). At short budgets, a stale self-play pool helps: fresh
snapshots of a half-trained agent add noise; infrequent updates give the
policy time to breathe. Would likely invert at long budgets where fresh
opponents matter more.

### Sweep 5 — entropy_coef @ 30 min (confirm)

| run | entropy_coef | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-ent30-003 | 0.003 | **100% (20/20)** | 16% |
| kar-ent30-010 | 0.01 (baseline) | 80% (16/20) | 16% |
| kar-ent30-030 | 0.03 | 90% (18/20) | 16% |

**Finding (legacy):** **flip from 15-min result** — at 30 min, low entropy
(0.003) wins vs baseline. Confirms hypothesis: low exploration needs more
time to exploit. Overall skill jumped vs 15-min runs.

### Sweep 6 — level_name breadth @ 15 min

| run | level | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-lvl-8-12 | random_8_12 (narrow) | **85% (17/20)** | 6% |
| kar-lvl-8-24 | random_8_24 (medium) | 70% (14/20) | **12%** |
| kar-lvl-8-32 | random_8_32 (wide) | 65% (13/20) | 4% |

**Finding (legacy):** vs-baseline rewards narrow training (specialist wins the
specialist test) but **vs-top-5 flips — medium (8-24) is the sweet spot**.
Suggests the prod config (8-12) is too narrow for generalization. *(This was
the seed for the b6 phase1_full_mix curriculum — see
`project_mushroom_wars_b6.md` in memory.)*

### Sweep 7 — capacity @ varied budgets (in flight when system was replaced)

Testing trunk-width-as-ceiling hypothesis. Sizes:
- `v9.0-full` (BODY=128, ~170k params)  — 30 min budget
- `v9.0-256`  (BODY=256, 395k params)   — 30 min budget
- `v9.0-512`  (BODY=512, 915k params)   — 30 min budget
- `v9.0-1024` (BODY=1024, 2.3M params)  — 3 hours

Results indeterminate under legacy axes; rerun once new system has a stable
archive baseline if capacity is still suspected.
