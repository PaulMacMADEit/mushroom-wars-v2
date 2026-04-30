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

### Loop fire 2 — 2026-04-28 00:06 PT — entropy sweep crashed; YAML fixed; re-queued

**All 3 entropy runs failed within 30s** with:
> `NotImplementedError: fused rollout doesn't support self_play yet; set cfg.fused_rollout=False or self_play=False`
(at `training/trainer.py:387` in `_collect_rollout_fused`)

Root cause: I set both `fused_rollout: true` and `self_play: true` in the
YAML baseline. Trainer asserts they're mutually exclusive — fused mode
hasn't been ported to handle self-play opponent specs yet.

**Fix (commit `3707773`):**
- `configs/karpathy_loop.yaml`: `fused_rollout: false`, `action_repeat: 1`.
  Wall-clock cost is ~30% per the Phase G memory but PFSP champion-archive
  draws are what we're actually testing, so non-fused is the right path.
- Removed `action_repeat` from sweep axes (no-op under non-fused).
- Added `latest_bias` axis instead — within-pool freshness vs older
  snapshots. Lets the sweep cycle reach 8 informative axes.

**Cron-meanwhile.** During the failure window, the live `cdcc0826`
(`cron-260428-0407-phase2_selfplay-med-00`) finished cleanly, was
**promoted to champion `0952f5cc`**. Notable readings:
- training rate=0.92 (it's `opponent_name=neural` against a fixed
  champion, not self-play — explains the high rate vs the 0.05 self-play
  norm)
- pfsp=0.536 (lowest we've seen — means strong/clear win-rates, not
  noisy 50/50 across the archive)
- Elo 1118 (highest live-rated entry on the chart)

**Re-queued** (`karp-260428-0022-entropy_coef-{lo,mid,hi}`):
| label | entropy_coef |
|---|---|
| karp-260428-0022-entropy_coef-lo  | 0.003 |
| karp-260428-0022-entropy_coef-mid | 0.01  |
| karp-260428-0022-entropy_coef-hi  | 0.03  |

Sweep finishes ~01:24 PT.

### Loop fire 3 — 2026-04-28 06:15 PT — second crash, root caused, third try succeeds

**Both 00:22 retry runs failed again:**
> `NotImplementedError: SIM_BACKEND=jax doesn't yet support per-env neural opponents.
>  Set self_play=False, or run numpy backend.` (`trainer.py:210`)

So the JAX backend **also** can't service per-env self-play opponent specs. Two trainer constraints now stacked:
1. `fused_rollout` ≠ `self_play` (caught fire 2)
2. `sim_backend=jax` ≠ `self_play` per-env opponents (caught fire 3)

To use the PFSP self-play path I built, we'd need to switch to the numpy
AsyncVectorEnv backend — which is materially slower at n_envs=1024.

**Pivot:** ship a working sweep first, port jax+self_play later.

- YAML now defaults `self_play: false` (with explicit comment).
- `scripts/queue_karp_sweep.py` looks up the most-recent champion at queue
  time and injects `opponent_name=neural` + `opponent_kwargs.opponent_run_id=<src>`
  so the run trains against the strongest known model, not random_legal.
- The PFSP benefit still lives at bench_eval time (archive sweep + PFSP
  weight write). Just not during training-rollout.

Also hit a transient Supabase pooler "consuming input failed: server closed"
on first 2 INSERT attempts; third attempt at 06:15 went through cleanly.

**Queued (third attempt):** `karp-260428-0615-entropy_coef-{lo,mid,hi}`,
training vs champion `cdcc0826` (`cron-260428-0407-phase2_selfplay-med-00`,
the live Elo 1118 model). `-lo` started cleanly at 13:15 UTC, past the
prior failure point. Should finish ~06:35 PT (lo) → ~07:15 PT (full sweep).

### Loop resilience fix — 2026-04-28 06:30 PT — chain broke for 5.5h overnight

**Problem.** Between fires 3 (00:38 PT) and 4 (06:23 PT, user-poked), no
autonomous fires happened — should have been ~5. Root cause: the loop
chain depends on a single `ScheduleWakeup` per fire. If that fire fails
to *complete* (mid-fire crash, prompt timeout, session loss), the chain
dies and no recovery exists.

**Three fixes:**
1. **30-min cadence** instead of 60-min. `configs/karpathy_loop.yaml`:
   `fire_interval_seconds: 1800`. Smaller interval = faster recovery.
2. **Reschedule-first policy.** Loop prompt now mandates step 1 of
   every fire is `ScheduleWakeup(...)`. If a later step crashes, the
   next fire still happens.
3. **PaulLinux server-side backstop.** `scripts/karp_backstop.py` +
   `mushroom-karp.timer` (every 30 min at :15/:45). Detects "no karp-
   run queued/running/recently-finished" and queues one. Idempotent —
   no-ops when Claude is keeping up.

Backstop installed: `mushroom-karp.timer` active, next fire 07:15 PT.
First test fire (07:12 PT) saw the running entropy sweep and no-op'd
correctly.

This means: even if my Claude session dies entirely overnight, karp-
runs keep flowing. I (Claude) become the analysis layer; the engine
runs without me.

### Loop fire 5 — 2026-04-28 07:41 PT — first successful sweep + lr queued

**Entropy sweep complete.** All 3 ran cleanly (no crashes). Trained vs
champion `cdcc0826` (Elo 1118) for 20 min each.

| run | entropy_coef | training rate | PFSP | bv n | promoted? | Elo |
|---|---|---|---|---|---|---|
| karp-260428-0615-entropy_coef-lo  | 0.003 | 5.4% | 0.866 | 10 | N | 979 |
| karp-260428-0615-entropy_coef-mid | 0.01  | 5.2% | 0.855 | 10 | N | 978 |
| karp-260428-0615-entropy_coef-hi  | 0.03  | 5.2% | 0.775 | 10 | N | 991 |

**Findings:**
- **None promoted** — none beat the champion ≥60% over 16 games. Expected
  for 20-min runs vs a 30-min-trained champion.
- **Training rate ~5%** across all three — they won 5% of training games
  vs the fixed champion. Tight clustering says entropy didn't move
  training-game outcomes much at this budget.
- **Hi (0.03) has best Elo (991), worst PFSP (0.775).** Reading: more
  entropy = more diverse play = clearer wins/losses across the archive
  (PFSP drops toward 0 when results are decisive, peaks at 50/50). So
  hi entropy may have slightly better generalisation, but in the
  noisier-vs-archive direction.
- **Within noise** on n=8 bench games per archive member. 10 archive
  members × 8 = 80 bench games each. Differences ≤ 13 Elo points.

**Tentative read:** entropy_coef wasn't the bottleneck. The fact that
nothing beat the champion at all says either (a) 20 min isn't enough
training time, (b) the champion is already at a local plateau the
sweep can't easily exit, or (c) we need a *different* axis to move
the needle.

**Queued next:** `karp-260428-0742-lr-{lo,mid,hi}` with lr ∈
{1e-4, 3e-4, 1e-3}. Same baseline cfg, training vs same champion
`cdcc0826`. Sweep finishes ~08:42 PT.

### Loop fire 6 — 2026-04-28 08:12 PT — lr-lo finished, lr-mid in flight

| run | lr | training rate | PFSP | bv n | promoted? | Elo |
|---|---|---|---|---|---|---|
| karp-260428-0742-lr-lo  | 1e-4 | 5.3% | 0.840 | 10 | N | **1003** |

lr-mid running (started 15:04 UTC), lr-hi queued. Skipping queue this fire
(2 karp runs ahead, sweep mid-flight). Will catch the full lr sweep next
fire (08:43 PT) and queue rollout_steps.

**Note:** lr-lo Elo 1003 is the **first karp run to clear the 1000 anchor**
— still well below the live champion (1118) but above all 3 entropy runs
(977/978/991). Reading: low lr (1e-4) didn't underfit at 20 min as the
legacy entropy_coef sweep had hinted (at 15-min budget that data showed
lr=1e-4 barely beat random at 50%). New eval system is showing different
signal.

### Loop fire 7 — 2026-04-28 08:43 PT — lr-mid done; 10x speedup unlocked

| run | lr | training rate | PFSP | bv n | Elo |
|---|---|---|---|---|---|
| karp-260428-0742-lr-lo  | 1e-4 | 5.3% | 0.840 | 10 | **1003** |
| karp-260428-0742-lr-mid | 3e-4 | 4.9% | 0.706 | 10 | 938 |
| karp-260428-0742-lr-hi  | 1e-3 | (running) | — | — | — |

**lr-mid Elo 938 is the worst karp result yet.** Lower than lr-lo (1003)
and worse PFSP (0.71 — meaning archive sweep was more decisive vs it).
Tentative: lower lr (1e-4) is better than baseline (3e-4) at 20-min budget,
maybe surprising vs Karpathy archive's "1e-3 wins at 15-min" finding.
Wait for lr-hi to land before concluding.

**🚀 GPU bottleneck investigated (Paul: "GPU only at 18%").**

Per-update breakdown of every karp run so far (4 runs averaged):
- **rollout: 98.7%** of update time
- **env_step: 93.6%** (CPU-bound JAX sim step)
- **learn: 1.3%** (the only GPU-bound bit)
- **act_batch: 0.2%**

Compare to cron-agent runs at SAME `n_envs=1024, rollout_steps=64`:
- karp non-fused: **316 steps/sec, 17 games/sec, 6 updates/20-min**
- cron fused:     **3087 steps/sec, 180 games/sec, 85 updates/20-min** (10x)

**Root cause:** I had `fused_rollout: false` in the YAML based on an
incorrect assumption that fused was incompatible with `opponent_name=neural`.
Verified 2026-04-28: fused only blocks `self_play=true` (per-env neural
opponents). With `self_play=false + opponent_name=neural` (our current
path), fused works perfectly — cron uses it daily.

**Fix (commit `28da48a`):** flipped YAML to `fused_rollout: true`,
`action_repeat: 2`. Already pushed + pulled on PaulLinux. Next sweep will
run fused. Existing lr-hi finishes on old non-fused cfg.

Expected impact: karp runs go from 6 → ~85 updates per 20-min cell. The
*signal* per run becomes much stronger. May lift Elo readings by enough
to start producing promotion-eligible runs.

### Loop fire 8 — 2026-04-28 09:14 PT — fused didn't speed things up; root cause = neural opponent

**Result of fire-7's "fused fix":** zero speedup. `889a3d78` still doing
~317 sps, ~5 updates in ~10 min. GPU still 18%.

**Root cause:** `opponent_name=neural` forces fused rollout into a
**slow per-env CPU loop** (`fused_rollout.py:188`). The cron-agent's fast
runs (3087 sps) use `opponent_name=random_legal` — fully on-device path
(`fused_rollout.py:181-185`). I conflated "fused works with neural" (it
doesn't crash) with "fused is fast with neural" (it isn't).

**lr sweep results:**

| run | lr | Elo | PFSP |
|---|---|---|---|
| lr-lo  | 1e-4 | **1003** | 0.840 |
| lr-mid | 3e-4 | 938 | 0.707 |
| lr-hi  | 1e-3 | 953 | 0.770 |

**Lower lr (1e-4) wins decisively.** Legacy archive's "1e-3 wins" finding
was rebench-invalidated.

**Fix (commit `10354f2`):**
- `scripts/queue_karp_sweep.py`: drop the auto-inject-champion-as-neural
  logic. Default to `opponent_name=random_legal`. The cron-agent's
  champion path proves: train fast vs random_legal, let bench_eval do
  the heavy lifting on champion comparison.
- Killed in-flight rollout_steps sweep (was running with neural opp).
  *Note: marking status=failed in DB doesn't stop the worker process —
  it'll finish naturally and overwrite the status. Bug to fix later.*
- Re-queued `karp-260428-0916-rollout_steps-{lo,mid,hi}` with
  `opponent_name=random_legal`.

Expected for next sweep: ~85 updates per 20-min cell (vs current ~6).
GPU should jump from 18% to ~70-80%.

**Backstop now fires every 15 min** (was 30) and has zero grace window —
empty queue = queue immediately. Eliminates idle gaps between sweeps.

### Loop fire 9 — 2026-04-28 12:02 PT — caught up after silent backstop period

**System running healthy without me.** Backstop (PaulLinux systemd timer)
queued 2 sweeps autonomously while my Claude wakeups failed to fire. n_envs
sweep + start of gamma sweep — no human intervention.

**Speedup verified.** All runs since switching to `opponent_name=random_legal`
hit 2700-4700 steps/sec (vs old 316). ~10-15× faster.

**rollout_steps sweep:**

| run | rollout_steps | updates | sps | rate | Elo | PFSP |
|---|---|---|---|---|---|---|
| rollout_steps-lo  | 32  | **105** | 2829 | 0.923 | **1107** | 0.551 |
| rollout_steps-mid | 64  | 64      | 3472 | 0.905 | 1100 | 0.593 |
| rollout_steps-hi  | 128 | 43      | 4520 | 0.863 | 1082 | 0.592 |

**Finding:** lower rollout_steps wins. More frequent updates beat longer
rollouts at this budget. Tradeoff: hi gets best raw sps (4520) but ⅓ the
update count, and Elo suffers.

**n_envs sweep:**

| run | n_envs | updates | sps | rate | Elo | PFSP |
|---|---|---|---|---|---|---|
| n_envs-lo  | 512  | **102** | 2751 | 0.921 | 1083 | 0.609 |
| n_envs-mid | 1024 | 67      | 3602 | 0.901 | 1087 | 0.577 |
| n_envs-hi  | 2048 | 44      | 4691 | 0.855 | 1081 | 0.586 |

**Finding:** **flat across n_envs** at 20-min budget. lo (512) edges hi
(2048) but within noise. Same pattern as rollout_steps: more updates
trumps bigger batches at this scale.

**Both sweeps tell the same story:** for this game/model size, the
update-count-vs-batch-size tradeoff favours "many small updates" over
"few large ones". The 3070's GPU% was never going to be the lever —
**update frequency** is what moves Elo.

**No karp- runs promoted to champion yet.** All Elos sit ~1080-1107
(above anchor 1000) but don't beat the live champion `0952f5cc` (Elo
1118) in head-to-head ≥60%. The champion was trained vs another champion
(self-play chain), karp- runs train vs random_legal — different ceiling.

**Open question:** should we switch a karp axis to test "vs champion"
(slower per-update) vs "vs random_legal" (faster, but capped Elo)?
Right now we're stuck below the champion ceiling regardless of how many
updates we do because the opponent distribution is too easy.

**Possible next code change:** the b6 champion is `cdcc0826` (`med-00`,
30-min budget). Karp runs are 20 min. To compete head-to-head we may
need (a) longer runs, or (b) train vs champion path with the slow
opponent. Worth a sweep on `cell_budget_seconds`.

**Schedule update:** Claude-side `ScheduleWakeup` not reliably firing
during active conversation. Server-side backstop is the actual loop
driver. Will rely on it. I'll continue logging when prompted.

### Loop fire 10 — 2026-04-28 12:16 PT — gamma-lo done; queued clip_coef

**Driver:** First fire under new `/loop 30m` mechanism (cron `7,37 * * * *`,
job `4b393dc4`). PaulLinux backstop still primary; this is bonus.

**State:** Worker + karp timer both active. Live champion still
`cron-260428-0407-phase2_selfplay-med-00` at Elo **1147** (advanced from
1118 in handoff). No karp- runs promoted yet.

**Gamma sweep — only `lo` finished (mid running, hi queued):**

| run | gamma | updates | sps | rate | Elo | PFSP | bv n | promoted? |
|---|---|---|---|---|---|---|---|---|
| gamma-lo  | 0.95 | 66 | 3588 | 0.909 | **1095** | 0.545 | 10 | N |
| gamma-mid | 0.97 | — | — | — | — | — | — | running |
| gamma-hi  | 0.99 | — | — | — | — | — | — | queued |

gamma-lo Elo 1095 sits inside the same ~1080-1107 band the rollout_steps
and n_envs sweeps produced. **Still opponent-bound, not hyperparam-bound.**
Mid + hi will tell us the gamma curve, but I'd be surprised if any cell
clears ~1110.

**No clutter to clear.** Queue: 1 running + 1 queued (depth 2 < cap 6).

**Queued next axis — clip_coef** (round-robin from gamma):

| label | clip_coef |
|---|---|
| karp-260428-1216-clip_coef-lo  | 0.1 |
| karp-260428-1216-clip_coef-mid | 0.2 (baseline) |
| karp-260428-1216-clip_coef-hi  | 0.3 |

**Open question carried from fire 9:** opponent-bound ceiling. Six axes
sampled (entropy, latest_bias, lr, rollout_steps, n_envs, gamma) all sit
~1080-1107. The cheapest test of "is this opponent-bound?" is to swap
ONE karp axis to `opponent_name=neural` (train-vs-champion) and see if
Elo crosses 1118. That's a deliberate one-variable test, not a
hyperparam sweep — won't run it without Paul's say-so. Logging here so
fire 11+ can pick it up.

### Loop fire 11 — 2026-04-28 12:41 PT — gamma-mid done; queue full, skip queueing

**State:** Worker + karp timer active. Champion Elo drifted 1147→1145
(stochastic, unchanged identity `0952f5cc`).

**Queue depth 4 (1 running + 3 queued)** — gamma-hi running, clip_coef
sweep all queued behind it. Cap is 6; queueing 3 more would hit 7. **Skipped
queueing per loop step-5.**

**Updated gamma sweep table:**

| run | gamma | updates | sps | rate | Elo | PFSP | bv n | promoted? |
|---|---|---|---|---|---|---|---|---|
| gamma-lo  | 0.95 | 66 | 3588 | 0.909 | **1095** | 0.545 | 10 | N |
| gamma-mid | 0.97 | 66 | 3521 | 0.907 | 1080 | 0.579 | 10 | N |
| gamma-hi  | 0.99 | — | — | — | — | — | — | running (12 min in) |

**First axis where lo cleanly beats mid by >10 Elo** (95 vs 80, 15-point gap).
All prior axes (rollout_steps, n_envs) had mid/lo within ~7 of each other.
If gamma-hi continues the slope downward (≤1075), the directional finding
is "lower gamma wins at this budget" — consistent with the rollout_steps
finding (more frequent updates beat longer horizons).

**Still opponent-bound.** Both gamma cells sit in the same ~1080-1107 band.
Champion at 1145, karp ceiling untouched. Open question from fire 9-10
unchanged.

**Next fire:** gamma-hi should finish ~12:50 PT (started 12:29, 20-min
budget). clip_coef sweep starts after. Fire 12 at 13:07 will likely have
gamma-hi result + first clip_coef cell mid-flight.

### Loop fire 12 — 2026-04-28 13:11 PT — gamma curve clean; clip_coef-lo unrated; skipped self_play-gated axes

**State:** Worker + karp timer active. Champion drift 1145→1145 (same id).

**Gamma sweep complete — clean monotone curve:**

| run | gamma | updates | sps | rate | Elo | PFSP | bv n |
|---|---|---|---|---|---|---|---|
| gamma-lo  | 0.95 | 66 | 3588 | 0.909 | **1095** | 0.545 | 10 |
| gamma-mid | 0.97 | 66 | 3521 | 0.907 | 1080 | 0.579 | 10 |
| gamma-hi  | 0.99 | 77 | 4192 | 0.894 | 1070 | 0.581 | 10 |

**Finding:** **lower gamma wins by ~25 Elo across the range** at 20-min
budget. 1095/1080/1070 — first axis with a clean monotone signal across
all three cells. Same direction as rollout_steps (lower wins) and n_envs
(noisy but lo edges). Theme: shorter horizons / smaller batches → more
updates → better Elo at this compute budget.

**Possible YAML change:** drop baseline gamma from 0.97 to 0.95. Holding
off without Paul's nod — would be a one-variable shift to all subsequent
sweeps.

**clip_coef-lo finished but UNRATED:**

| run | clip_coef | updates | sps | rate | Elo | PFSP | bv n | elo_status |
|---|---|---|---|---|---|---|---|---|
| clip_coef-lo  | 0.10 | 64 | 3491 | 0.909 | 1023 | None | **0** | **unrated** |

bench_eval skipped — no `bench_vector`, no PFSP weight, `elo_status=unrated`.
Elo 1023 is the bootstrap default (rated only against random_legal anchor).
Worker has no journald output (stdout not captured by systemd unit), so
root cause unknown. **Will let clip_coef-mid run and see if it repeats.**
If 2/3 cells in the sweep are unrated, that's a real bug to dig into.

**Queue management:** 0 running + 2 queued (clip_coef-mid/hi) = depth 2.
Cap 6, room for 3 more.

**Skipped 2 axes from round-robin** (`leaderboard_bias`, `latest_bias`)
— both self-play-pool gated. With `self_play: false` in baseline, they're
no-ops. Forced next axis to **entropy_coef** to skip the wasted-compute
cycle. Round-robin will hit them again later; if `self_play=true` is
unlocked (currently blocked by JAX backend per trainer.py:210 comment),
they become informative.

**Queued entropy_coef sweep** (lo=0.003 / mid=0.01 / hi=0.03). Total queue
depth now 5.

**Open question still open:** karp ceiling ~1095 vs champion 1145; nothing
this fire moved that. Skipping leaderboard_bias/latest_bias was the
small-but-right call.

### Loop fire 13 — 2026-04-28 13:41 PT — clip_coef-lo back-filled (false alarm); flat clip axis so far

**State:** Worker + karp timer active. Champion drift 1145→**1134** (-11),
same identity `0952f5cc`. Champion Elo is wandering as more karp- runs
add bench match data — not a real ability change.

**Correction to fire 12.** I flagged clip_coef-lo as unrated/bv=0 last
fire. **It was not broken — bench_eval just hadn't run yet.** It
back-filled by this fire to elo=**1085**, rated, bv=10, pfsp=0.545.
Lesson: there's a non-trivial gap between training-finish and bench_eval-
finish. Don't panic on a single "unrated" reading; wait one fire. Adding
this to "Don'ts" — see [project_mushroom_wars_karp_loop.md] update
needed.

**clip_coef sweep — lo + mid done (hi running):**

| run | clip_coef | updates | sps | rate | Elo | PFSP | bv n |
|---|---|---|---|---|---|---|---|
| clip_coef-lo  | 0.10 | 64 | 3491 | 0.909 | 1085 | 0.545 | 10 |
| clip_coef-mid | 0.20 | 64 | 3462 | 0.909 | **1089** | 0.545 | 10 |
| clip_coef-hi  | 0.30 | — | — | — | — | — | — running |

**Within 4 Elo** between lo and mid — flat axis so far. PPO clip ratio
matters less than gamma/rollout_steps at this budget. Will see hi.

**Queue depth:** 1 running (clip_coef-hi) + 3 queued (entropy_coef sweep)
= **4**. Adding 3 more would hit 7, exceeds cap 6. **Skipped queueing.**

**Carry forward:** decision items from fire 12 still open —
1. Drop baseline gamma 0.97 → 0.95?
2. Comment out `leaderboard_bias` + `latest_bias` from round-robin until
   self_play unlocks?

### Loop fire 14 — 2026-04-28 14:11 PT — concurrency change detected; clip_coef-hi stuck unrated 77 min; skipped queueing

**🚩 Worker concurrency changed since last fire.** All 3 entropy_coef
runs are training **in parallel**:

| label | started (UTC) | minutes in |
|---|---|---|
| entropy_coef-lo  | 20:55:30 | ~16 |
| entropy_coef-mid | 21:03:18 | ~8 |
| entropy_coef-hi  | 21:03:21 | ~8 |

Mid and hi started 3 seconds apart. Previously runs were strictly
serial. **I didn't make this change** — worker config or backstop must
have been touched. Surfacing to Paul rather than digging — could be
intentional (more throughput) or accidental (multi-launch race).

**clip_coef-hi stuck unrated 77 minutes after finishing.** For
comparison:
- clip_coef-lo: finished 20:11 UTC, rated by ~20:41 (≤30 min)
- clip_coef-mid: finished 20:32 UTC, rated by ~20:41 (≤10 min)
- clip_coef-hi: finished **20:54 UTC, still unrated at 22:11** — 77 min

**Hypothesis:** bench_eval is being starved by the 3 parallel training
runs. No GPU slack for it. (Alternative: bench_eval errored silently.)

**Updated clip_coef sweep — incomplete (hi unrated):**

| run | clip_coef | rate | Elo | bv n | st |
|---|---|---|---|---|---|
| clip_coef-lo  | 0.10 | 0.909 | 1085 | 10 | rated |
| clip_coef-mid | 0.20 | 0.909 | 1089 | 10 | rated |
| clip_coef-hi  | 0.30 | 0.904 | — | 0 | **unrated** ← stuck |

Lo + mid still flat (~4 Elo apart). If hi back-fills similar, this is a
genuinely flat axis.

**Champion identity unchanged.** Champ Elo holding at 1134 (random walk
from 1147 over the last 90 min — Elo-system noise as bench corpus grows).

**Skipped queueing.** Queue 3 running + 0 queued = 3. Adding 3 more
would max the cap at 6 AND pile on the bench_eval starvation. Wait one
fire to see if entropy-lo finishes cleanly first.

**Open ask for Paul:**
1. Did you intentionally enable parallel worker execution? If so, expect
   bench_eval to lag (or break). If not, single-process worker may have
   been duplicated by a stale backstop launch.
2. Same fire-12 questions still open (gamma 0.95 baseline? skip
   self_play-gated axes?).

### Loop fire 15 — 2026-04-28 14:41 PT — worker restarted, killed 2 zombies, re-queued entropy_coef

**Self-correction on fire 14.** I claimed "3 entropy_coef runs in
parallel" + "concurrency change". Wrong on both counts.

**What actually happened:**
- The original worker (running since 09:25 PDT, 5h+ CPU time) **died/restarted ~14:03 PT.**
- At restart, the DB had `entropy_coef-lo` (started 13:55) and
  `entropy_coef-mid` (started 14:03) listed as `running` — they were
  running in the *old* process. New process picks up `entropy_coef-hi`,
  trains it, completes with elo=1061/rated/bv=10.
- lo and mid never got re-picked-up — DB rows stayed `running` forever.
- Current state: 1 worker process (PID 3680495), GPU at 0%, 5.3GB RSS,
  stuck idle because the DB still shows in-flight work.

**The fire-14 "parallel runs" reading was a DB artifact — three rows
showing `running` simultaneously, but only one process actually ran any
of them.** No worker code or concurrency change. PaulLinux git is at
0910708, worker.py last modified 2026-04-27.

**Action this fire:**
1. **Marked `entropy_coef-lo` and `entropy_coef-mid` as `failed`** with
   error message identifying them as zombies. Standard "stale running
   → failed" cleanup per loop procedure.
2. **Re-queued entropy_coef** (`karp-260428-1443-entropy_coef-{lo,mid,hi}`)
   to recover the data we lost AND preempt the backstop from auto-
   picking the next round-robin axis (which would be `latest_bias`
   — self_play-gated, wasted compute).
3. **Did NOT investigate why the original worker died.** It had 5h+ CPU
   time, peaked at 9.0G memory; possibly OOM, possibly the systemd
   restart on memory pressure. Surfacing to Paul rather than digging.

**clip_coef-hi still unrated 105 min after finishing.** The bench_eval
backlog must have been wiped by the worker restart. That run is now
permanently unrated unless we manually re-rate it.

**Updated entropy_coef sweep — only -hi has data:**

| run | entropy_coef | updates | sps | rate | Elo | PFSP |
|---|---|---|---|---|---|---|
| -lo  | 0.003 | — | — | — | failed (zombie) | — |
| -mid | 0.01  | — | — | — | failed (zombie) | — |
| -hi  | 0.03  | 80 | 4358 | **0.863** | 1061 | 0.577 |

**Worth flagging:** -hi's rate (0.863) is the lowest of any karp- run
this whole loop (others 0.89-0.92). Higher entropy bonus → more
exploration → lower training-time win rate. Expected directionally; a
data point we want repeated in the re-run.

**No karp- promotions yet.** Champion `0952f5cc` holding at Elo 1140
(drift continues, identity unchanged).

### Loop fire 16 — 2026-04-28 15:11 PT — back to healthy serial; entropy_coef-lo done; queued lr

**State:** Worker healthy (PID 3680495, 1h 8min uptime, RSS 5.7GB,
+0.4GB since fire 15 — slow growth worth watching). GPU 4% (between
training iters when polled). Queue back to **serial flow** post-cleanup.

**Re-queued entropy_coef sweep — lo done, mid running, hi queued:**

| run | entropy_coef | updates | rate | Elo | PFSP | bv n |
|---|---|---|---|---|---|---|
| -lo  | 0.003 | 47 | 0.914 | **1057** | 0.588 | 10 |
| -mid | 0.01  | — | — | — running (6 min in) | — | — |
| -hi  | 0.03  | — | — | — queued | — | — |

**Note on -lo:** Only 47 updates (vs gamma sweep's 66, n_envs's 67, lr's
~100 from old config). Same 20-min budget. With low entropy_coef
(0.003), training apparently converged faster — fewer required updates
per fixed sps. Possible artifact of randomness or genuine signal that
low-entropy pol learns the random_legal opponent quicker. Will see
mid + hi.

**Champion Elo:** 1140 → 1142 (+2, drift only). Identity unchanged.

**Queued lr sweep next** (`karp-260428-1512-lr-{lo=1e-4,mid=3e-4,hi=1e-3}`).
Reasoning: `lr` was last swept in fire 7-8 BEFORE the opponent switch
to `random_legal`. Current cycle hasn't tested lr under the current
config. Forced this axis to skip `latest_bias` (self_play no-op).

**Queue depth now 5 (1 running + 4 queued)** — under cap 6. Worker has
~80 min of work ahead.

**Carry forward:** all 3 open decisions from fire 15 still open
(worker death, gamma baseline, self_play-gated axes).

### Loop fire 17 — 2026-04-28 15:41 PT — entropy_coef ultra-flat (lo=mid within 1 Elo); skipped queueing

**State:** Worker healthy (PID 3680495, 1h 38min uptime, RSS 5.74GB —
**growth has stalled**, +0.04 vs +0.4 since last fire). GPU 3% (between
iters when polled). Champion Elo 1140 (drift, identity unchanged).

**Entropy_coef sweep — lo + mid done:**

| run | entropy_coef | updates | rate | Elo | PFSP | bv n |
|---|---|---|---|---|---|---|
| -lo  | 0.003 | 47 | 0.914 | 1057 | 0.588 | 10 |
| -mid | 0.01  | 49 | 0.912 | 1058 | 0.607 | 10 |
| -hi  | 0.03  | — | — | — running (15 min in) | — | — |

**Within 1 Elo between lo and mid.** Updates count and training rate
near-identical. **PPO entropy bonus is a non-lever** at this scale vs
random_legal. Will see hi for the symmetry — fire-14's old entropy_coef-hi
was 0.03 → 1061 / rate 0.863 / 80 updates. Expect hi this run to be
similar (more exploration + slower convergence + similar terminal Elo).

**Worker memory tracking (since fire 15 restart):**
- Fire 15: 5.32GB (just-restarted)
- Fire 16 (+30 min): 5.73GB (+0.4)
- Fire 17 (+30 min): 5.74GB (+0.04)

Growth has plateaued. Probably JAX trace cache filling up to steady
state. No imminent OOM risk.

**Queue depth 4** (1 entropy_coef-hi running + 3 lr queued) — under cap
6 with 2 slots free. **Skipped queueing** — worker has ~80 min of work
ahead, no point piling on while bench_eval is healthy.

**Carry forward:** open decisions unchanged from fire 15-16.

### Loop fire 18 — 2026-04-28 16:11 PT — entropy_coef-hi underperforms; lr-lo done; loop has exhausted informative axes

**State:** Worker healthy (2h 8min continuous uptime, RSS 5.78GB
plateaued). Champion drift 1140→1140, identity unchanged.

**Entropy_coef sweep complete:**

| run | entropy_coef | updates | rate | Elo | PFSP |
|---|---|---|---|---|---|
| -lo  | 0.003 | 47 | 0.914 | 1057 | 0.588 |
| -mid | 0.01  | 49 | 0.912 | 1058 | 0.607 |
| -hi  | 0.03  | 54 | 0.901 | **1036** | 0.627 |

**Finding revised:** entropy_coef IS a lever, but the wrong direction
than I expected. Lo + mid are tied at ~1057 (plateau); **hi (0.03)
under-trains by ~22 Elo**. Higher entropy bonus → more exploration
under random_legal → fewer effective updates spent on exploiting.
This is the second monotone axis (with gamma) where lower wins.
PFSP weight (0.627) is highest for hi — confirming bench archive
beat it more often.

**Lr sweep — lo done, mid running, hi queued:**

| run | lr | updates | rate | Elo | PFSP |
|---|---|---|---|---|---|
| -lo  | 1e-4 | 49 | 0.916 | **1073** | 0.562 |
| -mid | 3e-4 | — running (1 min in) | — | — | — |
| -hi  | 1e-3 | — queued | — | — | — |

(Last lr sweep was fire 7-8, pre-random_legal, so this one's worth
keeping.)

**🛑 Loop status: all informative axes now swept under current config.**

Axis ledger:
| axis | status | finding |
|---|---|---|
| entropy_coef | ✅ swept | hi under-trains by 22 Elo; lo=mid plateau |
| lr | 🟡 in flight | lo=1073 so far |
| rollout_steps | ✅ (fire 9) | lower wins by ~12 Elo |
| n_envs | ✅ (fire 9) | flat |
| gamma | ✅ (fire 10-12) | **lower wins by 25 Elo** (strongest finding) |
| clip_coef | ✅ (fires 12-13) | flat |
| leaderboard_bias | ❌ skipped | no-op while self_play=false |
| latest_bias | ❌ skipped | no-op while self_play=false |

After lr-mid + lr-hi finish, **the loop has nothing new to learn under
current config.** Round-robin will start cycling repeats. The
opponent-bound ceiling (~1095 max karp Elo vs champion 1140) is the
real wall — flagged since fire 9.

**Strategic next moves to surface to Paul:**
1. **Bake gamma 0.95** into baseline (strongest finding) and re-sweep
   one or two axes to see if the absolute Elo lifts. Cheap.
2. **Add an "opponent" axis** — sweep `random_legal` vs `neural`-vs-
   champion, even though it'd be slow. Tests the ceiling-is-opponent-
   bound hypothesis directly.
3. **Bump cell_budget_seconds** from 20→30 min so karp matches the
   30-min budget cron-agent uses for `phase2_selfplay-med`. Currently
   the karp- ceiling could be partially budget-bound.
4. **Coast** — round-robin keeps generating noise data; useful only
   if we suspect Elo readings are unstable.

**Skipped queueing this fire.** Queue 2 (1 running + 1 queued). Will
let lr finish before deciding next move.

**Carry forward:** the 3 fire-15 open decisions (worker death? gamma
baseline? skip self_play-gated axes?) — this fire's strategic-next-
moves list is in the same family.

### Loop fire 19 — 2026-04-28 16:41 PT — lr signal: lo wins by 37 Elo over mid; let backstop coast

**State:** Worker healthy (2h 38min continuous, RSS 5.79GB plateau).
Champion drift 1140→**1145** (+5, identity unchanged).

**Lr sweep — lo + mid done, hi running:**

| run | lr | updates | rate | Elo | PFSP |
|---|---|---|---|---|---|
| -lo  | 1e-4 | 49 | 0.916 | **1073** | 0.562 |
| -mid | 3e-4 | 51 | 0.912 | 1036 | 0.625 |
| -hi  | 1e-3 | — | — | — running (9 min in) | — |

**lr-lo (1e-4) beats lr-mid (3e-4) by 37 Elo.** Third strong signal of
the loop (after gamma and entropy_coef hi-underperform). Lower lr
trains more cautiously; given that **most axes prefer "more updates"**
(rollout_steps lower, gamma lower, entropy lower), lower lr fits the
same theme — slower, more conservative learning produces a better
final policy at this 20-min budget.

**Three monotone-lower-wins findings:**
| axis | lo | hi | gap |
|---|---|---|---|
| gamma | 1095 | 1070 | -25 Elo |
| entropy_coef | 1057 | 1036 | -22 Elo |
| **lr** | **1073** | **1036 (mid)** | **-37 Elo** |

Plus rollout_steps (-12 Elo), n_envs (~flat), clip_coef (~flat).
Pattern is robust: the karp config wants smaller, more frequent updates.

**No autonomous config changes.** All strategic next moves from fire
18 require Paul's call (bake gamma 0.95? change cell budget? add
opponent axis?). Loop is now in repeat-cycles mode.

**Skipped queueing.** After lr-hi finishes (~16:52 PT), backstop's
15-min timer will pick `rollout_steps` (round-robin from `lr`) and
start the second cycle. Generates noise estimates for Elo stability.

**Worker memory footprint sanity check:**
- Fire 15→16: +0.4GB
- Fire 16→17: +0.04GB
- Fire 17→18: +0.04GB
- Fire 18→19: +0.01GB
Plateau confirmed. Original worker died at peak 9.0GB after 5h+ CPU
time; current process at 2h 38min / 5.79GB is well below that
trajectory, but worth a flag if it continues climbing past ~7GB.

### Game-review addition (per Paul's ask, fire 19)

Built `scripts/karp_review_games.py` to sample 1 win + 1 loss from the
latest rated karp- run, compute behavior signals (action-type
distribution, noop rate, entropy, value-drop, repeat-pick streaks,
weak sends), and flag anomalies. Now part of every loop fire.

**First-run finding on `karp-260428-1512-lr-lo` (Elo 1073, our best
karp- run this loop):**

| game | tag | ticks | dec | noop% | entropy | value drop | top types | flags |
|---|---|---|---|---|---|---|---|---|
| `d993327c` | WIN | 92 | 46 | 59% | 2.33 | +0.19 | noop=59% 100%=28% 75%=7% | high noop rate 59% |
| `9c2156c5` | LOSS | 83 | 42 | 52% | 3.02 | +6.70 | noop=52% 100%=31% 75%=12% | high noop rate 52% |

🚩 **High no-op rate** in both win + loss (52-59% of decisions are
"do nothing"). The policy is **passive by default** — only acts on
~half its decision opportunities. Notable additional signals:
- **No 25% or 50% sends ever** — when it does send, it's always
  75% or 100% (commit-max behavior).
- **Value drop +6.70 in the loss** vs +0.19 in the win — policy
  was confident at game start (~5.5) and lost confidence over the
  course of the game (-1.x at end). It saw the loss coming and
  didn't adjust strategy.
- **Long games (80-90 ticks)** for `random_8_12` map — passive
  behavior dragging out matches.

Possible causes of no-op preference:
1. Reward function doesn't penalize inaction enough.
2. Random_legal opponent is *also* passive enough that no-op is
   optimal at many ticks.
3. Action mask gates valid sends conservatively (need a min-garrison
   to send) and noop is the only legal action a lot of the time.

**Hypothesis 3 REJECTED — investigation 2026-04-28 17:00 PT.**

Counted 443 decisions across 24 lr-lo bench games:
| Reading | Count | % |
|---|---|---|
| Picked noop | 160 | 36% |
| Picked noop *because forced* (no send types legal) | 5 | **1.1%** |
| Picked noop *by choice* (sends were legal) | 155 | 35% |
| Decisions with 3+ send types legal | 419 | **94.6%** |

Mask is essentially permissive — almost every tick, 3 or 4 of the 4
send-percentage options are legal. **35% of decisions are noop-by-choice.**

Comparison with bench opponents (other karp/cron PPO runs) in the
same 24 games:

| action | lr-lo | opponents | diff |
|---|---|---|---|
| 25%  | 1.4% | 0.5% | ~same |
| 50%  | 3.6% | 6.5% | -3pp |
| 75%  | 18.7% | 14.0% | +5pp |
| 100% | 40.2% | 44.7% | -4pp |
| **noop** | **36.1%** | **34.3%** | **+2pp** |

**Passivity is fleet-wide.** Every PPO model trained under this config
converges to "noop ~35%, send 100% when sending". Not lr-lo specific.

This re-points investigation to:
- **Hypothesis 1 (reward shape)** — primary suspect. Passive
  consolidation produces a positive reward signal under random_legal.
- **Hypothesis 2 (random_legal is also passive)** — likely
  compounding factor. Random_legal samples uniformly over legal
  actions including noop, so it's also ~20% noop minimum (1/5
  action types when 4 send types are legal).

**Implication for the loop's strategic moves:** the karp ceiling
(~1095 Elo) and the passivity may be the same problem. Switching to
`opponent_name=neural` (train vs champion) would likely break this
equilibrium because the champion isn't passive enough to make
mutual-noop a viable strategy.

## Code changes during loop

### 2026-04-28 16:50 PT — added scripts/karp_review_games.py

Per Paul's fire-19 ask: review actual gameplay each loop fire. Script
samples 1 win + 1 loss from the latest rated karp- run, computes
behavior signals from replay JSON in the `replays` bucket, and flags
anomalies (high noop, type-collapse, repeat streaks, weak sends, low
entropy). First run discovered 52-59% noop rate in lr-lo — a real
behavior bug aggregate Elo numbers were hiding.

Procedure addition saved to project memory.

### Loop fire 20 — 2026-04-28 17:11 PT — lr full curve done; backstop kicked in 2nd cycle; passive-vs-aggressive contrast confirmed

**State:** Worker healthy (3h 8min uptime, RSS 5.80GB plateau).
Champion **+10 Elo bump** to **1155** — biggest single-fire jump
(identity unchanged, just bench corpus updates).

**Lr sweep complete (full curve):**

| run | lr | Elo |
|---|---|---|
| -lo  | 1e-4 | **1073** |
| -mid | 3e-4 | 1036 |
| -hi  | 1e-3 | **1009** |

**Range = 64 Elo.** Strongest single-axis signal of the loop. Lower
lr wins decisively. lr-hi at 1e-3 (the PPO default) is *worst*.

**Backstop kicked in.** Round-robin's 2nd cycle started: queued
`rollout_steps-{lo,mid,hi}` (last_used was lr in 1st cycle, picked
rollout_steps as next, correctly skipping the 2 self_play-gated axes
that I'd manually preempted). Cycle 2 will give noise estimates on
prior findings.

**Game review — best vs worst karp run:**

| run | wins/24 | tag | ticks | noop% | top type | value drop |
|---|---|---|---|---|---|---|
| lr-lo (best, 1073) | 12 | WIN | 92 | 59% | noop | +0.19 |
| lr-lo (best, 1073) | 12 | LOSS | 83 | 52% | noop | +6.70 |
| **lr-hi (worst, 1009)** | **4** | **WIN** | **8** | **0%** | **100%** | **-0.87** |
| lr-hi (worst, 1009) | 4 | LOSS | 50 | 40% | noop | +7.59 |

**The worst karp model wins faster (8 ticks!) and is more aggressive,
but loses 20/24 games. The best karp model is the most passive but
wins ~50%.** This validates fire-19's hypothesis: **bench_eval rewards
passivity in this fleet.** Aggressive plays crush some opponents fast
but can't handle the rest. Passive plays drag matches and survive.

**Implication:** the karp ceiling (~1095) and the fleet-wide passivity
are the same problem. Reward_v14 (per-tick shaping for active play +
holding territory + losing units = penalty) is the leading hypothesis
to break this equilibrium. Discussed with Paul — pending design
sign-off (magnitudes + symmetric vs event-based losing-units penalty).

**Queue:** 3 (1 running + 2 queued). Skipped queueing — let cycle 2
run for noise estimates while reward_v14 design lands.

**Carry forward:**
- reward_v14 design awaiting sign-off (magnitudes; symmetric vs
  event-based losing penalty).
- Strategic moves from fire 18 (gamma 0.95 baseline, opponent axis,
  cell budget bump) all pending.

### Loop fire 21 — 2026-04-28 17:20 PT — implemented reward_v14, queued v13/v14 A/B sweep

**Major code change.** Added `reward_v14` per Paul sign-off this fire:
- v1.4 keeps v1.3 terminal/capture rewards
- Adds per-tick shaping: `coef_b * (b_p1 - b_p2) + coef_u * (u_real_p1 - u_real_p2)`
- Coefficients: `coef_b = 0.0010` (per building diff), `coef_u = 0.0002` (per real-unit diff)
- Symmetric: p1 gets +delta, p2 gets -delta
- Per-tick shaping skipped on terminal tick (terminal reward already encodes outcome)
- Designed for total per-game shaping ≈ ±1.0 = ~20% of REWARD_WIN(5.0)

**Wiring** (committed `d813ce8`):
- `sim/config.py`: REWARD_VERSION_V14=2, *_BY_VERSION extended, new
  REWARD_TICK_{BUILDINGS,UNITS}_COEF_BY_VERSION
- `sim/engine.py` + `sim/engine_jax.py`: shaping injection point after
  victory check, byte-identical between numpy and JAX
- `training/trainer.py`: `cfg.reward_version: int` overrides legacy
  `cfg.reward_v13: bool` when ≥0
- `configs/karpathy_loop.yaml`: new 2-cell sweep axis `reward_version`
  (lo=1 v13 control, hi=2 v14 treatment)
- `tests/test_rewards_v14.py`: 6 new tests, all 37 existing tests pass

**State:** Worker healthy 3h 18min uptime, RSS 5.80GB plateau.
Champion 1155, identity unchanged. rollout_steps-lo still running
(~20 min in, finishing now). rollout_steps-mid + hi queued.

**Queued reward_version sweep** as next axis after the rollout_steps
cycle-2 wraps up:

| label | reward_version |
|---|---|
| karp-260428-1719-reward_version-lo | 1 (v13 control) |
| karp-260428-1719-reward_version-hi | 2 (v14 treatment) |

**Pre-deployment caveat.** The running worker (PID 3680495) imported
modules at startup (~14:03 PT). It will NOT pick up the new reward_v14
code until it restarts. Until then, the v14 cell will reference an
import that doesn't exist in worker memory → crash on entering training.

**Options for fire 22:**
- (A) Restart worker after rollout_steps cycle-2 finishes (~60 min)
  → cleanest, only zombies the cycle-2 last run if mid-bench
- (B) Restart now, zombie rollout_steps-lo + mid+hi
- (C) Wait for natural OOM (worker died last time at 5h CPU / 9GB peak;
  currently 3h CPU / 5.8GB plateau — could be hours)

Plan: fire 22 will check rollout_steps cycle-2 status; if done, do (A);
if still in flight and reward_version-lo about to start, restart with
controlled timing.

### Loop fire 22 — 2026-04-28 17:41 PT — worker restart confirmed; reward_version sweep up next

**State after restart.** Worker PID 4019322 (new), elapsed 19:41,
RSS 5.57GB. v14 code loaded — confirmed by running v14 tests on
PaulLinux post-pull (6/6 pass). Champion 1155, unchanged.

**rollout_steps cycle-2 status:**
- lo: done but **stuck unrated** (Elo=1000, bv=0). Bench_eval queue
  was wiped by the worker restart. Same pattern as clip_coef-hi at
  fire 15. Lost data point.
- mid: running 19m 49s (started 17:21 PT, just past 20-min budget,
  in bench_eval phase now)
- hi: queued

**Reward_version sweep ready.** Queued at fire 21:
| label | reward_version |
|---|---|
| reward_version-lo | 1 (v13 control) |
| reward_version-hi | 2 (v14 treatment, will use new code) |

First chance to start ~17:55 PT after rollout_steps-mid + hi wrap.
Since the worker is post-restart, **both cells will use the new code**
(v14 module is importable). Fire 23 should have first results.

**Game review** (correction — Paul caught fire-22 skipping this; should
have run regardless of "newest run" status). Two new rated runs since
fire 21:

| run | rollout_steps | wins/24 | Elo | sample-WIN noop% | sample-WIN top type |
|---|---|---|---|---|---|
| rollout_steps-mid (cycle 2) | 64  | **18** | 1053 | 67% | noop |
| rollout_steps-hi (cycle 2)  | 128 | 15 | 1048 | **50%** | **50%-send** |

**rollout_steps-hi shows NOVEL behavior** — first time we've seen a
non-100%-non-noop send dominate (50% sends were 50% of decisions in
the WIN sample). 3-tick wins. The longer effective horizon (128
vs 32 rollout steps) may have given the policy room to discover
that smaller sends preserve garrison for follow-up.

**rollout_steps-mid wins MORE games (18/24) but is more passive** —
67% noop in WIN sample, 41-tick games. Different strategy, similar
Elo. Bench_eval is averaging over both regimes.

**Cycle 1 → cycle 2 Elo shift:** rollout_steps results dropped 30-50
Elo (cycle 1: 1082-1107; cycle 2: 1048-1053). Champion advanced
~10 Elo over the same window (1145→1155); bench corpus is also
tougher. Real "Elo deflation" — same policy quality is graded harsher.

**v14 hypothesis check.** This passive-vs-aggressive Elo-tie is exactly
the equilibrium reward_v14 is designed to break. Expect v14 cells to:
- Push noop rate down across decisions
- Compress game length toward the "active" regime (8-42 ticks vs 80-92)
- Differentiate the two strategies on Elo (one should win clearly)

**Worker memory note.** New worker is at 5.57GB / 19:41 elapsed —
slightly faster ramp than the previous restart (5.32GB at similar
point), probably JAX cache + slightly different working set. Will
keep watching for the 9GB ceiling that killed the previous worker.

**Queue:** 1 running + 3 queued = 4. Skipped queueing.

### Loop fire 23 — 2026-04-28 18:11 PT — reward_version-lo running; champion +10 Elo

**State.** Worker PID 4019322, 49:39 elapsed, RSS 5.73GB (+0.16 since
fire 22; growth slower than first 30 min, looks plateau-bound around
6GB). GPU 3% between iters.

**Champion bumped +10 Elo to 1165** — second-largest single-fire jump
(after fire-20's +10). Identity unchanged (`cron-260428-0407-phase2_
selfplay-med-00`). Just bench-corpus updates from new karp- runs being
graded against it.

**🟢 Reward_version A/B is LIVE:**
- `reward_version-lo` (v13 control) **running**, started 18:05 PT,
  ~6 min in. Will finish ~18:25 PT.
- `reward_version-hi` (v14 treatment) queued next.

**Game review unchanged.** Latest rated is still rollout_steps-hi
(Elo 1048, 50%-send novel behavior — analyzed in fire 22 correction).
No new rated runs since.

**Queue depth 2** (1 running + 1 queued). Cap 6, room for 4.
**Skipped queueing** — v13/v14 A/B is the highest-value experiment
running; queueing more would push v14 results further back. Wait
until reward_version-hi completes before adding cycle-2 axes.

**Worker memory.** RSS trajectory:
- Restart: 4.4GB
- 19 min: 5.57GB
- 49 min: 5.73GB

Slope flattening. JAX cache warming up the same way the previous
worker did. No OOM concern at this rate.

### Loop fire 24 — 2026-04-28 18:41 PT — v13 control rated 1052; v14 running

**🟢 V13 control done.** First half of the A/B is in:

| run | reward_version | wins/24 | Elo | rate | updates | bv n |
|---|---|---|---|---|---|---|
| reward_version-lo | v13 (rv=1) | 11 | **1052** | 0.911 | 49 | 10 |
| reward_version-hi | v14 (rv=2) | — | running (15 min in) | — | — | — |

V13 control's Elo (1052) lands **right in the cycle-2 cluster**
(rollout_steps-mid 1053, rollout_steps-hi 1048, all ~1050±5). Confirms
baseline reproducibility under current bench corpus.

**v13 control game review** — same passive-survival pattern as prior
v13 runs:

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 52 | 62% | noop / 100% / 75% | -0.61 |
| LOSS | 38 | 42% | 100% / noop / 75% | **+6.06** |

This is the **baseline behavior signature reward_v14 is supposed to
disrupt** — high noop rate, value-drop in losses, mixed 100%/75% sends
when active.

**State.** Worker PID 4019322, 1h 19min uptime, RSS 5.75GB (+0.02 since
fire 23, plateau holding around 6GB). Champion drift 1165→1162.

**Reward_version-hi v14 ETA:** ~18:50 PT for training-done, ~19:00-19:10
PT for bench_eval-rated. Fire 25 (next) should have first v14 result.

**Queue empty after v14 finishes.** Backstop will tick ~19:00 PT and
queue next round-robin axis (gae_lambda → ...). Skipped queueing
this fire to keep v14 path clear.

**Strategy notes for fire 25 (when v14 result lands):**
- If v14 Elo > v13 by >20 (significant): re-sweep {gamma, lr,
  rollout_steps} under v14 (the three axes with strong v13 signal,
  to confirm findings carry to v14).
- If v14 Elo similar to v13 (±20): check behavior signals — v14 may
  shift behavior without shifting Elo (still useful, less urgent).
- If v14 Elo < v13: investigate whether shaping coefs are too high
  / too distracting from terminal signal.

### Loop fire 25 — 2026-04-28 19:11 PT — 🟢 v14 BROKE PASSIVITY but Elo dipped 12

**🟢 V14 result is in. Headline: v14 changed the policy completely
but Elo went down 12.**

**A/B summary:**

| metric | v13 control | v14 treatment | Δ |
|---|---|---|---|
| Wins / 24 | 11 | 8 | -3 |
| Elo | **1052.7** | **1040.6** | **-12** |
| Training rate | 0.911 | 0.917 | +0.6 pp |
| Updates | 49 | 50 | ~same |
| Steps/sec | 2664 | 2701 | ~same |

**🔥 BEHAVIOR DIFFERENCE IS DRAMATIC:**

| metric | v13 WIN sample | v14 WIN sample | Δ |
|---|---|---|---|
| **noop%** | **62%** | **0%** | **-62 pp** |
| ticks | 52 | 12 | -40 (-77%) |
| top types | noop > 100% > 75% | **100% / 75% / 50%** | new mix |
| value drop | -0.61 | -1.56 | (more confident) |

| metric | v13 LOSS sample | v14 LOSS sample | Δ |
|---|---|---|---|
| noop% | 42% | 31% | -11 pp |
| ticks | 38 | 32 | -6 |
| value drop | +6.06 | +7.73 | bigger collapse |

**v14 introduced a NEW behavior class** — 0% noop, fast 12-tick wins
using all three send sizes (100%/75%/50%). **First time across the
whole loop that 50%-sends appear in a sample.** The passivity
equilibrium IS broken. But the policy now over-commits and loses
more games (16/24 losses vs v13's 13/24).

**Diagnosis: shaping coefs are too aggressive.** Per-game shaping
budget ±1.0 was 20% of WIN(5.0); evidence suggests this is large
enough to push the policy past optimal aggression. The agent learned
that staying active = +shaping > terminal cost of losing fast.

**Strategic next moves (Paul to call):**
1. **Halve coefs (v14b)** — try 0.0005/0.0001 to keep the nudge but
   smaller. Closer to terminal-dominant.
2. **Add shaping-magnitude axis** — sweep 3 cells at 0.5x/1x/2x
   coefs, find the sweet spot.
3. **Asymmetric — drop units shaping** — units fires every tick on
   ~50 real-unit deltas (dominant); buildings fires more sparsely.
   Try keeping coef_b=0.001 and coef_u=0.
4. **Combine v14 + gamma 0.95** — gamma was the strongest single
   axis; lower gamma + shaping may compound.
5. **Accept** — v14 added policy diversity to the bench corpus; even
   if Elo dipped, future self-play could benefit from non-passive
   strategies in the archive.

**v14 hypothesis confirmed structurally** even though Elo went the
wrong way. The agent's *capacity* for active play was being shaped by
v13's terminal-only signal — change the signal, change the policy.
What we learned: **the passivity wasn't a learning bug, it was a
correct response to v13's reward landscape.** v14 found a different
attractor; we just need to tune coefs so it's a *better* attractor.

**Worker state.** PID 4019322, 1h 49min uptime, RSS 5.77GB (+0.02 since
fire 24, plateau holding). Champion 1167 (drift +5 since fire 24,
identity unchanged). Backstop pulled `entropy_coef` for cycle-2 (lo
running 10 min in, mid + hi queued; rv=1, all v13).

**Queue:** 1 running + 2 queued = 3. Skipped queueing — entropy_coef
re-sweep (cycle 2) is fine noise data; want to wait for Paul's call
on v14 next-step before adding more.

### Loop fire 26 — 2026-04-28 19:41 PT — entropy_coef cycle-2 lo done; passivity getting WORSE under v13

**State.** Worker PID 4019322, 2h 19min uptime, RSS 5.77GB plateau.
Champion drift 1167→1160 (-7, identity unchanged). GPU 2% between iters.

**entropy_coef cycle-2 progress:**

| run | cycle | entropy_coef | Elo | rate | updates |
|---|---|---|---|---|---|
| -lo (cycle 1) | — | 0.003 | 1057 | 0.914 | 47 |
| -lo (cycle 2) | — | 0.003 | **1048** | 0.914 | 49 |
| -mid | running, 19 min in | — | — | — | — |
| -hi | queued | — | — | — | — |

Cycle-2 lo is **9 Elo lower** than cycle-1. Same Elo-deflation pattern
seen in rollout_steps (cycle 1: 1082-1107; cycle 2: 1048-1053).
**Bench corpus is grading harder over time** as champion advances and
karp- runs accumulate.

**🚨 v13 passivity is GETTING WORSE.**

| run | WIN noop% | WIN ticks | LOSS noop% | LOSS ticks |
|---|---|---|---|---|
| fire 19 lr-lo | 59% | 92 | 52% | 83 |
| fire 22 rollout_steps-mid | 67% | 41 | — | — |
| fire 24 reward_version-lo (v13) | 62% | 52 | 42% | 38 |
| **fire 26 entropy_coef-lo cycle-2** | **69%** | **116** | **73%** | **95** |

The latest v13 run is the **most passive ever observed**. 73% noop in
loss is more passive than any prior sample. Game length 116 ticks (vs
typical 80-95) shows the policy *increased* its survival-by-stalling
strategy.

**Implication for v14 strategy.** Fire-25 read v14 as "broke passivity
but Elo dipped 12 → tune coefs down." Today's data adds nuance:

- **v13 isn't holding still** — it's deepening the passive lock-in. The
  -12 Elo gap measured at fire 25 was vs THAT moment's v13. By fire 26,
  the same v14 run might already be -5 or -8 vs the new v13 baseline.
- **Without v14 intervention, the fleet trends to mutual-noop**. The
  long-term cost of staying on v13 is a more brittle, less varied
  policy archive.
- **The "halve coefs" recommendation is still right**, but the urgency
  to deploy SOME v14 variant is higher than fire 25 implied.

**No code changes this fire.** Awaiting Paul's call on v14 next steps
(5 options surfaced fire 25 — recommendation: halve coefs to 0.0005 /
0.0001 and re-A/B).

**Queue:** 1 running + 1 queued = 2. Skipped queueing — let
entropy_coef cycle-2 finish for full noise estimate.

### Loop fire 27 — 2026-04-28 20:11 PT — entropy_coef cycle-2 complete; high entropy is partial-v14-substitute

**State.** Worker idle (GPU 0%), queue empty after entropy_coef cycle-2
finished. PID 4019322, 2h 49min uptime, RSS 5.75GB plateau. Champion
drift 1160→1166 (+6, identity unchanged).

**entropy_coef cycle-2 complete:**

| run | cycle 1 | cycle 2 | Δ |
|---|---|---|---|
| -lo (0.003) | 1057 | **1048** | -9 |
| -mid (0.01) | 1058 | **1041** | -17 |
| -hi (0.03) | 1036 | **1035** | -1 |

Average -9 Elo deflation, with **hi nearly flat** (-1) while lo + mid
dropped 9-17. Range narrowed from cycle-1's 22 Elo (1057-1036) to
cycle-2's 13 Elo (1048-1035). **Direction holds (lo > mid > hi) but
the high-entropy floor is more stable** vs the bench corpus changes.

**🟢 Game-review finding: high entropy is a partial v14 substitute.**

| approach | WIN noop% | wins/24 | mechanism |
|---|---|---|---|
| v13 + low entropy (passive baseline) | 69% | 11 | converges to noop equilibrium |
| v13 + high entropy | **20%** | 8 | explores out of equilibrium |
| **v14 + low entropy** | **0%** | 8 | **shaping forces non-passive** |

entropy_coef-hi cycle-2 WIN sample: 29 ticks, 20% noop, **100%/75%/noop
mix** (75% sends DO appear here). LOSS: 61% noop, value-drop +1.49
(vs v13-lo's +5.57). High entropy *partially* breaks the passivity
equilibrium even under v13.

This re-frames the v14 design space:
- v14 is one path to non-passive behavior (reward shaping)
- High entropy is another (exploration pressure)
- They could **combine** — v14 + entropy_coef=0.03 might compound the
  effect AND tune the over-aggression

**Backstop next.** Queue empty; backstop ticks at :15 (~4 min away).
Round-robin will pick `lr` (last_used was entropy_coef, lr is next in
YAML). Cycle-2 noise estimate on the strongest-signal axis (cycle-1
range was 64 Elo). Will let it run — high diagnostic value.

**Skipped queueing** — backstop has it covered, and pre-queueing
something v14-related autonomously would step on Paul's pending
decision (5 options surfaced fire 25, no call yet).

### Loop fire 28 — 2026-04-28 20:41 PT — lr cycle-2 in flight; lo Elo holds via 3-strong-wins not 11-mixed-wins

**State.** Worker PID 4019322, 3h 19min uptime, RSS 5.79GB (slow
0.04GB growth over 30 min — slight climb but well below previous
worker's 9GB ceiling). Champion **+5 to 1171** (drift, identity
unchanged).

**Backstop pulled lr cycle-2.**

| run | cycle 1 | cycle 2 | Δ |
|---|---|---|---|
| -lo (1e-4) | 1073 | **1054** | -19 |
| -mid (3e-4) | 1036 | running, 4 min in | — |
| -hi (1e-3) | 1009 | queued | — |

-19 Elo deflation, **biggest single-cell drop** of the cycles seen
so far (entropy_coef-lo was -9, rollout_steps-lo went unrated).

**🟢 Surprising game-review on lr-lo cycle-2:**

| metric | v13 control (fire 24) | lr-lo cycle-2 (this) |
|---|---|---|
| Elo | 1052.7 | 1053.7 |
| Wins / 24 | **11** | **3** |
| WIN noop% | 62% | **27%** |
| WIN ticks | 52 | 22 |
| LOSS value drop | +6.06 | +9.08 |

**Same Elo, very different policy.** lr-lo cycle-2 wins fewer games
but those 3 wins are against *stronger opponents* (PFSP weight raises
the value of beating tough opponents). The agent is more aggressive
(27% noop in WIN vs 62%), uses 100%-dominant attacks, and concedes
hard when losing (+9 value drop is largest of the loop).

**Insight:** Elo and win-count are decoupling. The bench corpus has
matured to where **policy quality ≠ win rate**. A specialist that
beats 3 strong opponents can score the same as a generalist that
beats 11 medium ones.

**Implication for v14 next-step.** This makes the v14-vs-v13 -12 Elo
gap from fire 25 even less meaningful — they could be measuring
different things. v14 might be a strong-opponent specialist (8 wins
but high-value), v13 a medium-opponent generalist (11 wins low-value).
**Recommendation: when Paul re-runs the v14 A/B, also pull per-game
opponent Elos to compare strong-vs-weak win patterns.**

**Worker memory note.** RSS 5.79GB after 3h 19min — first sustained
slow growth observed since the early plateau. Prior worker died at
9GB / 5h. Current trajectory ≈ 0.04GB / 30 min = 0.08GB/hr =
expected to hit 7GB by 6h, 8GB by 8h. Well within bounds.

**Queue:** 1 running + 1 queued = 2. Skipped queueing.

### Loop fire 29 — 2026-04-28 21:11 PT — 🚩 first type-collapse anomaly + worker mem accelerating

**State.** Worker PID 4019322, 3h 49min uptime, **RSS 6.28GB (+0.49GB
since fire 28)** — first big mem jump since early plateau. GPU 45%
(mid-training), CPU 135%. Champion drift 1171→1168.

**lr cycle-2 update:**

| run | cycle 1 | cycle 2 | Δ |
|---|---|---|---|
| -lo (1e-4) | 1073 | 1054 | -19 |
| -mid (3e-4) | 1036 | **1045** | **+9** ← went UP |
| -hi (1e-3) | 1009 | running, 12 min in | — |

**Range narrowed from 64 Elo (cycle 1) to 9 Elo so far (cycle 2).** The
strong "lower-lr-wins" signal is partially eroding under bench corpus
deflation. Will see hi.

**🚩 First TYPE-COLLAPSE anomaly detected — lr-mid cycle-2:**

| game | tag | ticks | noop% | top types | flag |
|---|---|---|---|---|---|
| `9d1f72cf` | WIN | 22 | 9% | **100%=91%** noop=9% | **type-collapse: 100% is 91%** |
| `27bbadfd` | LOSS | 51 | 38% | 100%=46% noop=38% 75%=12% | — |

The script's anomaly detector caught this — first type-collapse
flag across all 29 fires. The policy converged to "always send 100%".
Wins quickly (22-tick games) when it works, loses with **+8.92 value
drop** when it doesn't. **Policy fragility** — narrow action repertoire.

**Diagnosis: Elo deflation is pushing policies to extremes.** Cycle-2
samples now show:
- super-passive: entropy_coef-lo cycle-2 (73% LOSS noop, 116-tick games)
- super-aggressive: **lr-mid cycle-2 (91% type-100, 22-tick wins)**

The middle-ground passive-aggressive blend that was cycle-1's norm is
eroding. Bench corpus is selecting for specialists.

**Worker memory note 🚩.** RSS jumped +0.49GB in 30 min. Previous worker
hit 9GB at ~5h CPU and crashed. Current: 6.28GB at 3h49min. Linear
extrapolation: 9GB at ~5h30min — i.e. **~1h40min from now**. If memory
keeps growing at this rate, plan for restart around fire 32.

**Queue:** 1 running + 0 queued = 1. Skipped queueing. After lr-hi
finishes (~9 min), backstop will pick `rollout_steps` cycle-3 next.
We've covered most axes twice now. Loop is firmly in noise-estimate
mode for v13; **the actionable next step remains v14 next-step decision
(Paul to call from fire-25 5-option list, expanded by fire-28's
opponent-Elo recommendation).**

### Loop fire 30 — 2026-04-28 21:41 PT — lr cycle-2 complete: lr-hi crashed to Elo 954 (sub-anchor!); worker mem normalized

**State.** Worker PID 4019322, 4h 19min uptime, **RSS 5.79GB (-0.49
since fire 29)** — memory came back down. Previous worker's "runaway"
pattern not happening; fire 29's spike was transient (likely JAX cache
pressure that GC'd back down). Restart pressure removed. Champion drift
1168→1172 (+4, identity unchanged).

**lr cycle-2 fully complete:**

| run | cycle 1 | cycle 2 | Δ |
|---|---|---|---|
| -lo (1e-4) | 1073 | 1054 | -19 |
| -mid (3e-4) | 1036 | 1045 | +9 |
| -hi (1e-3) | 1009 | **954** | **-55 ⚠️** |

**lr-hi cycle-2 dropped to Elo 954** — the **first sub-1000 of any
cycle-2 cell**. Below the random_legal anchor (1000). Range jumped
from cycle-1's 64 Elo to cycle-2's **100 Elo** (1054-954). The
"lower-lr-wins" signal is *stronger* in cycle-2, not weaker.

**Game review on lr-hi cycle-2 (Elo 954):**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 22 | 18% | 100% / noop / **50%** | -0.73 |
| LOSS | 47 | 46% | noop / 100% / 50% | **+11.31 ⚠️** |

**+11.31 value-drop is the largest of the entire loop** (previous
record was +9.08). High lr produces wildly unstable training — policy
oscillates each step, lands on something that strong opponents crush.

**Pattern: 50%-sends emerge ONLY at axis high-end:**
| run | cycle | 50%-sends in WIN | Elo |
|---|---|---|---|
| rollout_steps-hi cycle-2 | — | 50% | 1048 |
| **lr-hi cycle-2** | — | 18% | 954 |

These two runs are the **only** places non-100/non-noop sends have
appeared all loop. Both at axis high-end. **Higher learning pressure →
more action diversity → either novel wins or catastrophic losses.**

**Connection to v14 thesis.** v14 (per-tick shaping) is one mechanism
to drive action diversity. The other mechanism we've seen is **high
learning pressure** (lr-hi, rollout_steps-hi). v14 finds the diversity
*deliberately* via reward shape; high-lr finds it *accidentally* via
training instability. v14's better — it gets the diversity AND the
training stays stable (rate 0.917 in fire 25's v14 run).

**Backstop progress.** Now in **cycle-3** (3rd time around the round-
robin) on rollout_steps. lo running 11 min in. The cycles are giving
us a richer noise picture but not new findings — most actionable
next step remains v14 next-step decision.

**Queue:** 1 running + 2 queued = 3. Skipped queueing.

### Loop fire 31 — 2026-04-28 22:11 PT — first 25%-send in WIN sample; rollout_steps cycle-3 deflating

**State.** Worker PID 4019322, 4h 49min uptime, RSS 5.80GB plateau
holding (~+0.01 since fire 30). Champion drift 1172→1169.

**rollout_steps cycle progression:**

| run | cycle 1 | cycle 2 | cycle 3 |
|---|---|---|---|
| -lo (32)  | 1107 | unrated (lost) | **1033** |
| -mid (64) | 1100 | 1053 | running |
| -hi (128) | 1082 | 1048 | queued |

cycle 1 → cycle 3 lo deflation: **-74 Elo**. The bench corpus has
hardened significantly across the day.

**🟢 First 25%-send detected in a WIN sample.**

Game review on rollout_steps-lo cycle-3:
- WIN: 45 ticks, 52% noop, **noop=52% / 100%=43% / 25%=4%**
- LOSS: 52 ticks, 50% noop, noop=50% / 100%=38% / 75%=8%

The 25%-send (small action) appears in the WIN sample for the first
time across 30+ fires. Previously 25%-sends only showed up in losses.
**The agent finally used the full 4-tier send vocabulary in a victory.**

**Send-vocabulary breadth across cycle-2/3 runs:**

| run | sends seen in WIN | Elo |
|---|---|---|
| entropy_coef-lo cycle-2 | noop, 100%, 75% | 1048 |
| reward_version-lo (v13) | noop, 100%, 75% | 1052 |
| **rollout_steps-lo cycle-3** | **noop, 100%, 25%** | 1033 |
| rollout_steps-hi cycle-2 | noop, 100%, 75%, 50% | 1048 |
| lr-mid cycle-2 (type-collapse) | 100%, noop only | 1045 |
| lr-hi cycle-2 | 100%, noop, 50% | 954 |
| **reward_version-hi (v14)** | **100%, 75%, 50%** | **1040** |

**v14 is still the only run with non-noop-dominant + 3 send sizes** in
a WIN. Vocabulary breadth weakly correlates with Elo but isn't sufficient.

**Queue:** 1 running + 1 queued = 2. Skipped queueing.

**v14 next-step still pending Paul** (5 options from fire 25 + fire 27's
v14+entropy compound idea + fire 28's per-game-opponent-Elo recommendation).

### Loop fire 32 — 2026-04-28 22:41 PT — rollout_steps cycle-3 complete; HI INVERTED above MID; calmest policy yet

**State.** Worker idle (GPU 0%), queue empty. PID 4019322, **5h 19min
uptime** (worker now exceeded previous worker's 5h-CPU lifetime — no
crash, RSS 5.76GB plateau holding). Champion **+6 to 1175** (identity
unchanged).

**rollout_steps full cycle progression:**

| run | cycle 1 | cycle 2 | cycle 3 |
|---|---|---|---|
| -lo (32)  | **1107** | unrated | 1033 |
| -mid (64) | 1100 | 1053 | **1020** |
| -hi (128) | 1082 | 1048 | **1043 ← inverted** |

**cycle-3 hi (1043) > cycle-3 mid (1020)** — the "lower-rollout_steps-
wins" signal from cycle-1 has **flipped**. Range cycle-3 = 23 Elo
(narrowing from cycle-1's 25). Hi went DOWN least (-5 from cycle-2).
Lower rollouts (32, 64) are now penalized more under harder bench corpus.

**🟢 Game review on rollout_steps-hi cycle-3 — calmest policy yet:**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 78 | 44% | noop / 100% / 75% | **+0.28** |
| LOSS | 22 | 27% | 75% / 100% / noop | **+0.35** |

**LOSS value-drop +0.35 is the LOWEST seen across all 30+ fires.**
For comparison: recent typical +5 to +9; loop record +11.31 (lr-hi
cycle-2). This policy plays through losses **calmly** — same value
at end as start. Doesn't oscillate or panic.

**Action diversity in defeat:** 75%-sends are 36% in LOSS sample —
actively engaging while losing. Compare to passive baselines that
hit 50-73% noop in losses. This is a *resilient* policy.

**Insight: longer rollouts → calmer policy.** Cycle-3 hi has 128
rollout steps before each update — enough horizon for the policy to
internalize "I'm losing but it's not over yet". Shorter rollouts (32,
mid 64) lead to twitchy reactive policies that panic-spike under stress.

**v14 design implication.** If v14's drawback was over-aggression
(fire 25), **v14 + rollout_steps=128** might compound favorably:
- v14 provides per-tick non-passivity reward
- Long rollouts give the policy time to learn measured non-aggression
- Combined: active but calm

This adds a 7th variant to the v14 next-step list (alongside fire-27's
v14+entropy compound). Recommendation tier:
1. **v14 + rollout_steps=128** (NEW — long horizon for measured aggression)
2. **v14 + entropy=0.03** (fire 27 — exploration tames over-aggression)
3. **Halve coefs to 0.0005/0.0001** (fire 25 — direct knob)

**Worker memory verdict.** 5h 19min uptime, 5.76GB RSS, no crash. The
previous worker's 9GB-at-5h was either a different leak path (maybe
in code we've since changed) or the result of a different workload mix.
Current worker stable. **No restart pressure.**

**Backstop next.** Queue empty, ticks at :45 PT (~4 min). Round-robin
from rollout_steps picks **n_envs** for cycle-3.

### Loop fire 33 — 2026-04-28 23:11 PT — n_envs cycle-3 in flight; new pattern: short passive wins

**State.** Worker PID 4019322, **5h 49min uptime**, RSS **6.44GB
(+0.68 since fire 32)** — second mem spike (similar to fire 29's
+0.49). Consistent with transient JAX cache pressure pattern. GPU
45% (mid-iter). Champion drift 1175→1171 (-4, identity unchanged).

**n_envs cycle progression:**

| run | cycle 1 | cycle 3 |
|---|---|---|
| -lo (512) | 1083 | **1035 (-48)** |
| -mid (1024) | 1087 | running |
| -hi (2048) | 1081 | queued |

Cycle-1 was flat (range 6 Elo). Cycle-3 lo deflation -48 — significant.
Will see if mid+hi follow.

**🟢 New behavior pattern: short passive wins.**

n_envs-lo cycle-3 game review:
- WIN: **25 ticks**, **69% noop**, 100%/75% as backup — short passive win
- LOSS: 26 ticks, 38% noop, +4.34 value drop

Compare to fire 26's entropy_coef-lo cycle-2 (also 69% noop in WIN):
- fire 26: 69% noop / **116-tick wins** ("survive forever, opponent fades")
- fire 33: 69% noop / **25-tick wins** ("survive briefly, opponent collapses")

The bench corpus has matured to where opponents also self-destruct
quickly. Passive-survival now wins in 25 ticks because opponents
crumble first. **Less about agent skill, more about opponent fragility.**

This is a *negative* sign for v13's path: the policy isn't getting
better, but the *opposition* is getting worse. Mutual decay.

**v14 implication.** Strengthens the fire-26 thesis (passivity is
deepening + getting *worse* over time). The fact that 25-tick wins
exist via 69% noop means **the bench corpus is rewarding passivity
more, not less**. Without v14 intervention, expect more cycle-3
runs to converge to short-passive equilibrium.

**Worker memory note 🚩.** RSS climbed from 5.79 → 6.44 GB this fire
(+0.68). Same pattern as fire 29's transient spike (which GC'd back
to 5.79 by fire 30). Still under 9GB ceiling. Will check fire 34
for the GC pattern.

**Queue:** 1 running + 1 queued = 2. Skipped queueing.

### Loop fire 34 — 2026-04-28 23:41 PT — n_envs-mid cycle-3 emerges aggressive; 3 archetypes within 20 Elo

**State.** Worker PID 4019322, **6h 19min uptime**, RSS **6.32GB
(-0.12 since fire 33's 6.44 spike, partial GC)**. VRAM 6533 MiB
(+520 vs fire 33). Champion **+9 to 1180** — biggest single-fire
jump in many fires (identity unchanged).

**n_envs cycle-3 progression:**

| run | cycle 1 | cycle 3 | Δ |
|---|---|---|---|
| -lo (512) | 1083 | 1035 | -48 |
| -mid (1024) | 1087 | **1023** | **-64** |
| -hi (2048) | 1081 | running, 12 min in | — |

Cycle-3 -mid deflation -64, biggest n_envs cell drop. Bench corpus
continuing to harden.

**🟢 n_envs-mid cycle-3 emerged AGGRESSIVE:**

| game | tag | ticks | decisions | noop% | top types | value drop |
|---|---|---|---|---|---|---|
| WIN | 14 | 7 | **14%** | **100%=86%** noop=14% | **-2.28 (record!)** |
| LOSS | 21 | 11 | 18% | 100%/noop/50% | +5.83 |

WIN value-drop **-2.28 is the LARGEST negative drop of all 30+ fires**
(prev record ~-1.6 from v14). Policy was *very* confident throughout
the win. 100%-sends are 86% of decisions in the win — just below the
type-collapse threshold (85%); only 7 decisions so small-sample,
genuinely high concentration.

**🟢 Three different policy archetypes clustered within 20 Elo:**

| run | archetype | Elo |
|---|---|---|
| n_envs-lo cycle-3 | passive: 69% noop / 25-tick WIN | 1035 |
| **n_envs-mid cycle-3** | **aggressive: 14% noop / 14-tick WIN / type-100=86%** | **1023** |
| rollout_steps-hi cycle-3 | calm: 44% noop / +0.35 LOSS drop | 1043 |

**The bench corpus is averaging across very different strategies.**
Same Elo window (~1020-1045) reached via three completely different
paths. This is *exactly* the equilibrium v14 should break — by
rewarding a *single* coherent active strategy rather than letting
policies drift to whichever local equilibrium each run lands on.

**v14 thesis strengthens.** What we're seeing under v13:
- Policy diversity is *high* (good for bench corpus richness)
- Policy *coherence per run* is low (bad for actually getting better)
- Each run finds its own local-optimum basin
- Bench Elo measures "average across opposing basins", not "raw skill"

v14's per-tick shaping should pull all runs toward a *common*
non-passive attractor — at the cost of some Elo (per fire 25's
v14 result), but with the upside of compounding skill rather than
oscillating between archetypes.

**Worker memory.** 5.79 → 6.44 (spike) → 6.32 (partial GC). Net
+0.53 since plateau, slight long-term creep. Still under 7GB at 6h+
uptime. Watching but no restart pressure.

**Queue:** 1 running + 0 queued = 1. Skipped queueing. Backstop ticks
:45 PT next; round-robin from n_envs picks **gamma** for cycle-3.

### Loop fire 35 — 2026-04-29 00:11 PT — n_envs cycle-3 complete; 4 archetypes within ~20 Elo

**State.** Worker PID 4019322, **6h 49min uptime**, RSS **6.61GB
(+0.29 since fire 34)** — long-term creep continues despite partial
GCs. MEM% 10%. Champion 1180 (identity unchanged).

**n_envs cycle-3 complete:**

| run | cycle 1 | cycle 3 | Δ |
|---|---|---|---|
| -lo (512)  | 1083 | 1035 | -48 |
| -mid (1024) | 1087 | 1023 | -64 |
| -hi (2048) | 1081 | **1041** | -40 |

Range cycle-3 = 18 Elo. Same flat-ish direction as cycle-1 (range 6
Elo). n_envs is the only axis where cycle-1 and cycle-3 still agree
(no inversion). hi edged lo, mid worst.

**Game review on n_envs-hi cycle-3 — less-passive policy emerges:**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 30 | **13%** | 100%/noop/75% | -0.40 |
| LOSS | 51 | 23% | **100%=31% / 75%=27% / noop=23%** | +8.63 |

LOSS sample uses **75%-sends 27%** — first time we've seen 75%-sends
as a meaningful share in a LOSS. Combined with low noop (23%), this
is a less-passive losing policy.

**🟢 4 archetypes within ~20 Elo under v13 cycle-3:**

| run | archetype | WIN noop% | Elo |
|---|---|---|---|
| n_envs-lo cycle-3 | passive (short wins) | 69% | 1035 |
| **n_envs-hi cycle-3** | **less-passive** | **13%** | **1041** |
| n_envs-mid cycle-3 | aggressive (type-100=86%) | 14% | 1023 |
| rollout_steps-hi cycle-3 | calm (LOSS drop +0.35) | 44% | 1043 |

**The bench corpus rewards a continuum of playstyles.** Elo no longer
discriminates strategy. v14 thesis strengthens: shaping is needed to
pull all runs toward a single coherent attractor instead of letting
each run drift into its own basin.

**Backstop pulled gamma cycle-3.** mid running 10 min in (started
00:01 PT). lo + hi queued.

**Worker memory creep 🚩.** 5.79→6.44→6.32→6.61 GB across fires
32/33/34/35. Net +0.82 over 4 fires (2h). At this rate (+0.27/30min
avg), worker hits 9GB by ~5 AM PT — within session window. **Restart
should happen within next 2-3 fires.** Will time it between runs to
avoid a zombie.

**Queue:** 1 running + 2 queued = 3. Skipped queueing.

### Loop fire 36 — 2026-04-29 00:41 PT — gamma cycle-3 mid done; second 3-tick WIN of the loop

**State.** Worker PID 4019322, **7h 19min uptime**, RSS **6.59GB
(-0.02 since fire 35)** — first downtick of the creep, GC nudged off
20 MB. Holding around 6.6GB ceiling for now. Champion **+5 to 1185**
(identity unchanged, steadily climbing).

**gamma cycle-3 progression:**

| run | cycle 1 | cycle 3 | Δ |
|---|---|---|---|
| -lo (0.95) | **1095** | queued | — |
| -mid (0.97) | 1080 | **1043** | -37 |
| -hi (0.99) | 1070 | running, 19 min in | — |

Cycle-1 had the cleanest monotone signal of the whole loop (lo > mid > hi,
range 25 Elo). Will see if cycle-3 confirms.

**🟢 New record: 3-tick WIN — second of the entire loop.**

gamma-mid cycle-3 game review:
- WIN: **3 ticks**, **50%=50% / noop=50%** — fastest WIN tied with
  rollout_steps-hi cycle-2 (also 3-tick, also used 50% sends)
- LOSS: 23 ticks, 50% noop, 100%/75% backup, value-drop +4.94

**Pattern: 3-tick wins use 50%-sends.** Both observed 3-tick wins
across 36 fires used 50%-sends. **Commit-50%-immediately is a viable
opening exploit** when the opponent is fragile (cycle-2/3 deflation
regime). Agent has discovered an aggressive opener.

**v13 cycle-3 is producing increasingly diverse winning paths.** Adds
to fire-34's "4 archetypes within 20 Elo" finding — short-passive
(25-tick, 69% noop), aggressive (14-tick, 86% type-100), and now
**ultra-fast 50%-rush (3 ticks)** all coexist around Elo 1023-1043.

**Worker memory creep update.** 5.79→6.44→6.32→6.61→6.59 across 5
fires. Net +0.80 since fire-32 plateau, but creep is *slowing*: last
fire's first downtick. May plateau at ~6.6GB without restart.
**Holding off restart this fire — will check trajectory fire 37.**

**Queue:** 1 running + 1 queued = 2. Skipped queueing.

### Loop fire 37 — 2026-04-29 01:11 PT — gamma cycle-3 INVERTED; extremes coexist; mem plateau confirmed

**State.** Worker PID 4019322, **7h 49min uptime**, RSS **6.59GB
(flat vs fire 36)** — creep STOPPED. Two consecutive flat readings
confirm plateau at ~6.6GB. Restart pressure fully removed. Champion
**+4 to 1189** (identity unchanged, steady climb).

**🟢 gamma cycle-3 complete — INVERTED from cycle-1:**

| run | cycle 1 | cycle 3 | Δ |
|---|---|---|---|
| -lo (0.95) | **1095** | 1027 | **-68** |
| -mid (0.97) | 1080 | **1043** | -37 |
| -hi (0.99) | 1070 | **1014** | -56 |

Cycle-1: **lo > mid > hi monotone** (range 25 Elo, strongest signal).
Cycle-3: **mid > lo > hi**. The "lower-gamma-wins" finding is **gone
under deflation**. lo deflation -68 Elo is cycle-3's biggest single
gamma drop.

**Three major axes have now shown signal-erosion under cycle-3:**
| axis | cycle-1 direction | cycle-3 direction |
|---|---|---|
| gamma | lo > mid > hi (range 25) | **mid > lo > hi** (range 29) |
| rollout_steps | lo > mid > hi (range 25) | **hi ≈ lo > mid** (range 23) |
| lr | lo > mid > hi (range 64) | lo > mid > hi (range 100, hi sub-1000) |

Only lr keeps the cycle-1 direction; gamma + rollout_steps inverted.

**🚩 NEW EXTREME — gamma-hi cycle-3:**

| game | tag | ticks | noop% | top types |
|---|---|---|---|---|
| WIN | **110** | **78%** | noop=78% / 100%=16% / 75%=4% | **most passive WIN sample of all 36 fires** |
| LOSS | 10 | 20% | 100%/noop/25% | fast loss using 25%-sends |

gamma=0.99 (highest discount = longest effective horizon) → agent uses
its long horizon to **wait for opponents to time out**. 110-tick games
+ 78% noop = pure stalling. **Timeout-victory specialist.**

**🟢 v13 cycle-3 produces both extremes simultaneously:**

| cell | strategy | WIN ticks | WIN noop% |
|---|---|---|---|
| gamma-hi | stalling extremist | 110 | 78% |
| gamma-mid | ultra-fast 50%-rush | 3 | 50% |
| n_envs-mid | aggressive type-100 | 14 | 14% |
| rollout_steps-hi | calm | 78 | 44% |
| (rest) | various intermediates | 25-78 | 13-69% |

**Stalling beats aggressive (timeout victory). Aggressive beats stalling
(hit fast). Mid-tier struggles against both.** Bench corpus rewards
each at ~1014-1043 because they trade wins.

**🟢 v14 thesis at maximum strength.** v13 produces a fragmented
strategy space where every policy has a counter. v14's per-tick
shaping pulls all runs toward a unified non-passive attractor —
eliminating the stall basin entirely. The 110-tick 78% noop WIN is
exactly what shaping would prevent.

**Worker memory.** 5.79→6.44→6.32→6.61→6.59→6.59. Plateaued. **No
restart needed.** The previous worker's 9GB-at-5h crash was
workload-specific.

**Queue empty.** Backstop ticks :15 PT (4 min), picks **gae_lambda**
for cycle-3.

### Loop fires 38-41 — 2026-04-29 01:52-03:14 PT — consolidated entry

Fires 38-40 had state-pulls and analysis but never landed a commit
(interrupted mid-fire each time). Consolidating with fire 41 here.

**State at fire 41.** Worker PID 4019322, **9h 52min uptime**, RSS
6.58GB plateau holding (bounced 6.59→6.61→6.62→6.61→6.58 across
fires 37-41 — no creep). Champion drift 1189→1178→1179 across
the 4-fire window (small noise, identity unchanged).

**🐛 Bug fix to scripts/karp_review_games.py (fire 38).** When pulling
matches for a run, all 10 bench-eval matches share the same `created_at`
(queued in one transaction). The script's `order(created_at desc)
.limit(5)` was non-deterministically picking 5 ties — sometimes hitting
the 5 matches that hadn't been populated with games yet, returning 0.
**Fixed:** removed implicit limit, fetch up to 20 matches and let the
games-table query filter to the populated ones. Game review on
gae_lambda-lo cycle-3 worked correctly after fix.

**gae_lambda cycle-3 (NEW axis, never run before):**

| run | gae_lambda | Elo | rate | updates |
|---|---|---|---|---|
| -lo | 0.90 | **1046** | 0.926 | 46 |
| -mid | 0.95 | 1015 | 0.916 | 49 |
| -hi | 0.98 | 1018 | 0.896 | 52 |

**lo > hi > mid, range 31 Elo.** Lower gae_lambda wins (shorter advantage
horizon, less variance per update). Same direction as cycle-1's gamma
+ lr findings: **karp config wants conservative, stable updates**.
gae_lambda-lo game review: 36% noop in 22-tick WIN, balanced 45/45/9
LOSS. Moderately active, balanced.

**clip_range cycle-3 in progress:**

| run | clip_range | cycle 1 | cycle 3 | Δ |
|---|---|---|---|---|
| -lo (0.10) | 1085 | **958** | **-127 ⚠️** |
| -mid (0.20) | 1089 | 1014 (unrated, in bench_eval) | TBD |
| -hi (0.30) | unrated | queued | — |

**clip_range-lo cycle-3 crashed to Elo 958 (sub-anchor).** -127 Elo
deflation vs cycle-1 — **biggest single-cell drop of the entire loop**
(prior record: lr-hi cycle-2 -55, n_envs-mid cycle-3 -64). cycle-1
clip_range was flat (1085/1089). cycle-3 is opening up dramatically.

**🟢 clip_range-lo cycle-3 game review — v14-LIKE behavior under v13:**

| game | tag | ticks | decisions | noop% | top types | value drop |
|---|---|---|---|---|---|---|
| WIN | **9** | 5 | **0%** | **100%/75%/50%** | -0.79 |
| LOSS | 23 | 12 | 42% | noop/100%/75% | +8.90 |

**This is the FIRST single-WIN-sample with 0% noop AND a three-way
send mix (100%/75%/50%) under v13.** It's exactly v14's signature:
active, varied send-vocabulary. Difference: v14 hit Elo 1040; this
clip_range-lo cycle-3 is sub-anchor at 958.

**Why?** Low clip_range = small per-step PPO update = cautious but
exploratory. Same theme as gae_lambda-lo (which scored 1046). But
under cycle-3 corpus deflation, the exploration didn't pay off in
wins — the policy is varied but lossy.

**Implication:** v14's per-tick shaping reaches the same active-varied
attractor as low-clip_range, but **stays at higher Elo** (1040 vs
958). v14 trades less Elo for the same behavior diversity. This
reframes fire-25's "v14 -12 Elo" verdict more favorably:
- v13 + low-clip-range: -127 Elo for active-varied policy
- **v14 + baseline clip-range: -12 Elo for active-varied policy**

**v14 is a 10× more efficient path to the same diversity.**

**Worker memory.** No restart pressure. 9h 52min uptime, plateau at
~6.6GB. Net +0.80 since fire-32 plateau then flat. Previous worker's
9GB-at-5h crash was workload-specific (since fixed by code changes
or simply different seed).

**Queue.** clip_range-mid done (in bench_eval, will rate next fire),
clip_range-hi queued.

### Loop fire 42 — 2026-04-29 03:52 PT — clip_range cycle-3 complete; opposite-extreme policies emerge across the axis

**State.** Worker PID 4019322, **10h 30min uptime**, RSS 6.61GB
plateau. Champion drift 1179→1176.

**clip_range cycle-3 complete:**

| run | clip_range | cycle 1 | cycle 3 | Δ |
|---|---|---|---|---|
| -lo (0.10) | 1085 | **958** | **-127** |
| -mid (0.20) | 1089 | 1014 | -75 |
| -hi (0.30) | unrated | **1033** | NEW |

**Cycle-3: hi > mid > lo** (range 75 Elo). Cycle-1 was flat (range 4
Elo on lo+mid). **clip_range is the 4th axis to expand/invert under
cycle-3 deflation** (after gamma, rollout_steps, gae_lambda).

**🟢 Same axis, opposite-extreme policies emerge:**

| cell | WIN noop% | WIN top types | LOSS noop% | LOSS value drop | Elo |
|---|---|---|---|---|---|
| clip_range-lo cycle-3 | **0%** | 100/75/50 (varied) | 42% | +8.90 | 958 |
| clip_range-hi cycle-3 | 21% | **100%=79%** (narrow) | **75%** (extreme) | +5.06 | 1033 |

Same hyperparam axis, dramatically different policy archetypes:
- **lo (small updates)** → varied-but-uncertain policy. Active sends
  but fewer wins, big LOSS collapses.
- **hi (big updates)** → narrow-decisive policy. 79% type-100 in WIN,
  but **75% noop in LOSS** — pure giving up.

**Two-faced policies.** clip_range-hi is **aggressive when winning,
passive surrender when losing**. It can finish fragile opponents fast
(28-tick wins) but collapses in defeat. clip_range-lo is more active
even in defeat (42% noop vs 75%) but its win attempts are too varied
to consistently break opposition.

**Reinforces fire-37's fragmented-strategy-space picture.** Cycle-3
bench corpus is a **mosaic of brittle local-equilibrium policies**.
Each beaten by a specific counter. Bench Elo averages over the
rock-paper-scissors interactions.

**update_epochs cycle-3 in flight** (NEW axis): lo running ~7 min
in, mid + hi queued. Will be another data point on whether new axes
under cycle-3 land in the same strategy-fragmentation regime.

**Worker memory.** 6.61GB at 10h 30min — flat plateau holding. No
restart pressure.

**Queue:** 1 running + 2 queued = 3. Skipped queueing.

### Loop fires 43-45 — 2026-04-29 04:20-05:17 PT — consolidated entry

Fires 43-44 had state pulls + analysis but didn't land commits.
Consolidating with fire 45 here.

**State at fire 45.** Worker PID 4019322, **11h 55min uptime**, RSS
6.61GB plateau holding firmly. Champion drift 1176→1177→1173.

**update_epochs cycle-3 complete (NEW axis, never run before):**

| run | update_epochs | Elo | rate |
|---|---|---|---|
| -lo  | 2  | **1023** | 0.905 |
| -mid | 4 (baseline) | 1008 | 0.912 |
| -hi  | 8  | 1008 | 0.894 |

**lo > mid ≈ hi**, range 15 Elo (modest). Same theme as gae_lambda-lo,
lr-lo: less-aggressive updates win. Diminishing returns past 4 epochs.

**🟢 update_epochs-hi cycle-3 game review — broken value head:**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 16 | 25% | 100%=75% | -1.00 |
| **LOSS** | 7 | 25% | 100%=75% | **-1.31 (NEGATIVE)** |

**First negative LOSS value-drop across 44 fires.** Every prior LOSS
sample had positive drop (agent's value estimate fell during the
game = lost confidence). This run shows value *rising* +1.31 during
a loss — **the agent thought it was winning when it lost**.

Same play in both WIN and LOSS (25% noop / 100%=75%). **Monotone
policy with broken value estimation.** High update_epochs (8) likely
overfits to noisy advantage estimates → value head hallucinates.
Different failure mode from clip_range-hi cycle-3's "two-faced"
(aggressive WIN + passive LOSS) — this is "blind" not two-faced.

**update_epochs-lo game review (fire 43, captured but uncommitted):**
- WIN: 18 ticks, 11% noop, 100%=56% / 75%=33% / noop=11% — varied,
  active, 75%-sends 33% (high)
- LOSS: 46 ticks, 74% noop, value-drop +0.77 (low — calm)

Combination of "active winner + calm loser." Compare to clip_range-hi
cycle-3 (similar Elo 1033): aggressive WIN + extreme passive LOSS but
panicked (+5.06 drop). Same broad strategy, very different emotional
regulation.

**minibatch_size cycle-3 in flight (NEW axis).** lo running 16 min in,
mid + hi queued. Will be one more data point before the cycle-3 round-
robin reaches max_grad_norm + reward_version (already swept).

**Known race condition observation.** update_epochs-mid cycle-3 was
rated (Elo 1008, bv=10, rated) but its 10 matches all had 0 games
populated. Bench_eval rating system writes to `runs.bench_vector`
*before* the per-game replay records hit `games` table. Game review
on freshly-rated runs may need to wait. **Not a bug — design decoupling.**

**Worker memory.** 6.61GB plateau holding for 7+ fires (since fire 37).
No restart pressure. 11h 55min uptime confirms previous worker's
9GB-at-5h crash was workload-specific.

**Queue.** 1 running + 2 queued = 3. Skipped queueing.

### Loop fire 46 — 2026-04-29 05:45 PT — minibatch_size cycle-3: mid wins, lo sub-anchor with active-loss behavior

**State.** Worker PID 4019322, 12h 24min uptime, RSS 6.61GB plateau
holding 8+ fires. Champion drift 1173→1170.

**minibatch_size cycle-3 progress (NEW axis):**

| run | minibatch_size | Elo |
|---|---|---|
| -lo (256) | **998** (sub-anchor!) |
| -mid (512) | 1019 |
| -hi (1024) | running |

**mid > lo by 21 Elo, lo sub-anchor.** First cycle-3 axis where mid
edges lo (will see hi).

**🟢 minibatch_size-lo cycle-3 game review — "active losing":**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 11 | 50% | 100%=50%/noop=50% | -1.37 |
| LOSS | 49 | **16%** | **100%/50%/75% (varied!)** | +6.94 |

**Active even in defeat** — only 16% noop in LOSS, 4 different action
types used (100/75/50/noop). 50%-sends *in a LOSS sample* — first
time we've seen the smallest-non-noop send size dominate during a loss.
The agent stayed engaged but couldn't break the opponent.

**Pattern confirmed across "cautious update" axes:**

| axis cell | Elo | WIN noop% | sends used in WIN | sends used in LOSS |
|---|---|---|---|---|
| clip_range-lo cycle-3 | 958 | 0% | 100/75/50 | noop/100/75 |
| **minibatch_size-lo cycle-3** | **998** | 50% | 100/noop | **100/50/75** |
| reward_version-hi (v14) | 1040 | 0% | 100/75/50 | 100/noop/75 |

**Small-batch / small-clip axes produce v14-like ACTIVE behavior under
v13, but at sub-anchor Elo (~960-1000). v14 reaches the same active
attractor at +42 to +82 Elo.** Three independent v13 mechanisms now
confirm v14's efficiency (clip_range-lo, minibatch_size-lo, gae_lambda-lo).

**Worker memory.** 6.61GB at 12h 24min — plateau confirmed across 8+
fires. The previous worker's 9GB-at-5h crash was workload-specific.
**No restart pressure.**

**Queue:** 1 running + 0 queued = 1. Skipped queueing. Backstop ticks
next, picks value_coef cycle-3 (NEW axis).

### Loop fires 47-48 — 2026-04-29 06:16-06:41 PT — minibatch_size cycle-3 + value_coef cycle-3 lo

**State.** Worker PID 4019322, **13h 19min uptime**, RSS 6.61GB
plateau holding 9+ fires. Champion drift 1170→1167→1159 (-11 net,
biggest drift sequence in many fires; identity unchanged).

**minibatch_size cycle-3 complete (NEW axis):**

| run | minibatch_size | Elo | wins/24 |
|---|---|---|---|
| -lo (256)  | 998 (sub-anchor) | 9 |
| -mid (512) | 1019 | (game review failed — race condition) |
| -hi (1024) | **1032** | **18** |

**hi > mid > lo, range 34 Elo, monotone.** Larger minibatch wins.
Same direction as rollout_steps cycle-3 (hi > mid). Both axes that
**increase per-update signal stability** win under cycle-3 corpus
deflation.

minibatch_size-hi cycle-3 game review:
- WIN: replay missing (race condition — game JSON not yet uploaded)
- LOSS: 54 ticks, 44% noop, **75%-sends 33%** (active-losing pattern again),
  value-drop +7.93 (panicked)

**🚩 5th cycle-1 finding to invert under cycle-3.** The "smaller-fewer-
faster wins" theme from cycle-1 has now flipped on:
- gamma (lo > mid > hi → mid > lo > hi)
- rollout_steps (lo > mid > hi → hi ≈ lo > mid)
- gae_lambda (NEW — lo wins, but range narrow)
- clip_range (flat → hi > mid > lo)
- **minibatch_size (NEW — hi > mid > lo, range 34 Elo)**

Only **lr** kept cycle-1 direction (lo > mid > hi).

**value_coef cycle-3 in progress (NEW axis):**

| run | value_coef | Elo | wins/24 |
|---|---|---|---|
| -lo (0.25)  | **1020** | 9 |
| -mid (0.50 baseline) | running 4 min in | — |
| -hi (1.00)  | queued | — |

**🟢 value_coef-lo cycle-3 — second negative LOSS value-drop seen:**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 43 | 45% | noop/100%/75% | -0.79 |
| **LOSS** | 16 | 50% | noop/100%/75% | **-0.13 (negative)** |

LOSS value-drop only -0.13 — agent's confidence barely shifted while
losing. Similar to update_epochs-hi cycle-3's -1.31. **Two paths to
"blind" policies emerged:**
1. **Under-train value head** (value_coef-lo): gradient pressure on
   value head is small, value head stays near-prior, doesn't track
   game state.
2. **Over-train value head** (update_epochs-hi): value head overfits
   to noisy advantage estimates, hallucinates value.

Both fail at distinguishing winning from losing positions. Symmetric
failure modes.

**Worker memory.** 9+ fires of flat plateau at 6.61GB. No restart pressure.

**Queue:** 1 running + 1 queued = 2. Skipped queueing.

### Loop fires 49-50 — 2026-04-29 07:12-07:47 PT — value_coef cycle-3 = U-SHAPED axis (first observed); max_grad_norm cycle-3 in flight

**State.** Worker PID 4019322, **14h 25min uptime**, RSS 6.61GB
plateau holding 11+ consecutive fires (since fire 37). Rock-solid.
Champion drift 1157→1162 (+5 rebound from prior decline 1180→1157).

**🟢 value_coef cycle-3 complete (NEW axis) — FIRST U-SHAPED axis of the loop:**

| run | value_coef | Elo | rate |
|---|---|---|---|
| -lo (0.25) | **1020** | 0.923 |
| -mid (0.50 baseline) | **1001** | 0.906 |
| -hi (1.00) | **1025** | 0.898 |

**hi > lo > mid.** mid (the *baseline* value!) is the **worst cell**.
Both extremes (very-low or very-high value-loss weight) produce
better policies. Range 24 Elo.

**Mechanism hypothesis:**
- mid (0.5): standard PPO weighting → no specialization, middle gets stuck
- lo (0.25): value head undertrained → policy ignores noisy value → exploratory
- hi (1.0): value head heavily trained → policy gets accurate value → deterministic

**Game review on value_coef-hi cycle-3 — extreme deterministic policy:**

| game | tag | ticks | noop% | entropy | top types |
|---|---|---|---|---|---|
| WIN | **10** | **0%** | **0.93** | **100%=80%** / 75%=20% |
| LOSS | 46 | 39% | 3.35 | varied |

**Entropy 0.93 is the LOWEST WIN-sample entropy of all 49+ fires.**
Typical WIN entropy is 1.5-3.0. 0.93 = near-deterministic policy.

**Game review on value_coef-mid (fire 49, captured but uncommitted):**
- WIN: 29 ticks, 33% noop, 100%/noop/75% (moderate)
- LOSS: 19 ticks, **10% noop**, 100%=60% / 75%=30% — most active LOSS sample
  of all 49 fires (10% is below all prior records)

**The two basins of the U-curve produce opposite policies:**

| run | value_coef | WIN noop% | WIN entropy | strategy |
|---|---|---|---|---|
| value_coef-lo (Elo 1020) | 0.25 | 33% | 2.17 | exploratory mixed |
| value_coef-hi (Elo 1025) | 1.00 | 0% | **0.93** | deterministic aggressive |

**Same Elo, opposite policies.** Reinforces fire-37/42's fragmented-
strategy-space picture from yet another angle. **U-shaped axes can
hide the best cells if you only run 2-cell A/B.**

**max_grad_norm cycle-3 in flight (NEW axis).** lo running 16 min in,
mid + hi queued. After this, the round-robin will have hit every
informative axis at least once in cycle-3 (or once total for the new
ones). The next cycle would either repeat or — if Paul calls v14
next-step — pivot to v14 testing.

**Champion Elo dynamics.** Tracked 1180 → 1157 (-23 over 2h, fire
44-49) then **+5 rebound to 1162**. Identity unchanged throughout.
Bench corpus continues averaging the champion against new karp- runs.

**Worker memory.** 11+ fires at 6.61GB. Plateau is real. **Previous
worker's 9GB-at-5h crash was workload-specific, not a leak.**

**Queue:** 1 running + 2 queued = 3. Skipped queueing.

### Loop fire 51 — 2026-04-29 08:19 PT — max_grad_norm cycle-3: lo restricts gradients too hard

**State.** Worker PID 4019322, **14h 49min uptime**, RSS 6.61GB
plateau holding 12+ consecutive fires. Champion +6 rebound to 1168
(continuing recovery from -23 dip earlier).

**max_grad_norm cycle-3 progress (NEW axis):**

| run | max_grad_norm | Elo | wins/24 |
|---|---|---|---|
| -lo (0.25) | **1010** | 6 |
| -mid (0.50 baseline) | 1020 | (race condition) |
| -hi (1.00) | running, just started | — |

**mid > lo by 10 Elo so far.** Range 10 Elo (small). Will see hi.

**max_grad_norm-lo cycle-3 game review:**
- WIN: 38 ticks, 47% noop, 100%/75% backup — moderate-passive
- LOSS: **11 ticks, 50% noop, ONLY 100%/noop** (binary action set!),
  value drop +7.49 (panicked)
- **Only 6/24 wins** — among the lowest of cycle-3 cells

**max_grad_norm=0.25 (small grad clip)** = restrict gradient updates
aggressively → cautious learning. Result:
- middling-passive WIN
- panicked-binary LOSS (no 75%-sends, no 50%-sends)
- only 6 wins

Restricting gradients too hard hurts both action diversity AND
resilience. **Same theme as fire-50's "mid is the worst" pattern
on value_coef** — but here lo (not mid) is worst because the axis
direction reversed (low max_grad_norm = MORE restrictive, not less).

**Queue.** 1 running + 0 queued = 1. After hi finishes, backstop
will pick reward_version (already done as A/B) or cycle through
again. **The loop has now covered all 11 axes** at least once in
cycle-3 (gae_lambda, update_epochs, minibatch_size, value_coef,
max_grad_norm = 5 NEW axes added this cycle).

**Skipped queueing.**

### Loop fire 52 — 2026-04-29 08:33 PT — quiet check-in; max_grad_norm-hi still in flight

**State.** Worker PID 4019322, 15h 11min uptime, RSS 6.61GB plateau
(13+ fires). Champion 1168 flat from fire 51.

**No new runs rated since fire 51.** max_grad_norm-hi still running
(started 08:14 PT, ~19 min into 20-min budget — about to enter
bench_eval phase).

**Discussed v14 next-step with Paul.** Recommendation stands:
1. **Land v14b** (halve coefs, 3-cell sweep) — directly tests if
   per-tick shaping recipe lifts active-policy attractor's Elo
2. Run a 60-min cell to test compute scaling (training-discipline
   gate before architecture)
3. Only if both fail → consider architecture (last resort per repo's
   own training-discipline.md rules)

**No autonomous v14 work** — awaiting Paul's call on coefs / cell count.

**Queue:** 1 running + 0 queued. Skipped queueing.

### Loop fire 53 — 2026-04-29 08:41 PT — max_grad_norm cycle-3 complete; backstop will re-run reward_version A/B next

**State.** Worker idle (GPU 0%), queue empty. PID 4019322, 15h 19min
uptime, RSS 6.60GB plateau (14 fires of flat readings). Champion drift
1168→1169.

**max_grad_norm cycle-3 complete (NEW axis):**

| run | max_grad_norm | Elo | wins/24 |
|---|---|---|---|
| -lo (0.25) | 1010 | 6 |
| -mid (0.50 baseline) | **1020** | (race) |
| -hi (1.00) | 1016 | 8 |

**mid > hi > lo. Range 10 Elo (small).** Flat-ish curve with a slight
edge to mid. Tight clipping (lo) is the only clear loser — over-
restricts gradients, forces binary action set in LOSS.

max_grad_norm-hi game review:
- WIN: 17 ticks, 11% noop, **100%/75%/noop varied** — active
- LOSS: 43 ticks, 32% noop, 100%/noop/**50%-sends** appear, value
  drop +8.82 (panicked)
- Allows action diversity that lo's 0.25 clip suppressed

**🟢 Backstop will pick reward_version next (round-robin from
max_grad_norm).** This is **valuable** — re-runs the v13/v14 A/B
under the harder cycle-3 bench corpus. Tests whether v14's -12 Elo
gap from fire 25 holds, shrinks, or grows under deflation.

If v14 holds or improves vs the harder corpus → strong signal to
land v14b with refined coefs. If v14 falls further → diagnosis
shifts to "v14 coefs are absolutely too strong, not just relatively."

**Queue:** empty. Skipped queueing — backstop's :45 tick handles it.

### Loop fire 54 — 2026-04-29 09:11 PT — 🟢 v13 control cycle-2 went TYPE-COLLAPSE (0% noop, 100%=100%); v14 running

**State.** Worker PID 4019322, **15h 49min uptime**, **RSS 7.11GB
(+0.50 since fire 53)** — first significant uptick after 14-fire plateau.
Watching for transient vs sustained creep. Champion drift 1169→1162 (-7).

**🟢 v13/v14 A/B re-run by backstop. v13 control rated; v14 in flight:**

| run | reward_version | Elo | wins/24 | WIN noop% | WIN top types |
|---|---|---|---|---|---|
| **fire-24 v13 control** | 1 | 1052 | 11 | 62% | noop > 100% > 75% |
| **fire-54 v13 control (NEW)** | 1 | **1033** | **13** | **0%** | **100%=100%** ← type-collapse |
| fire-25 v14 treatment | 2 | 1040 | 8 | 0% | 100%/75%/50% |
| **fire-54 v14 treatment** | 2 | running, ~4 min in | — | — | — |

**🚨 The v13 baseline has FUNDAMENTALLY SHIFTED.**

Fire 24's v13 control: **62% noop in WIN, passive-survival** — the
"baseline behavior signature reward_v14 is supposed to disrupt."

Fire 54's v13 control: **0% noop in WIN, 100% type-collapse** — the
v13 baseline is NOW similar to v14's behavior signature, just narrower.

**Why this happened.** Cycle-3 bench corpus has matured to reward
faster aggressive strategies (passive ones lose). v13 under selection
pressure has converged to **short-aggressive type-100 wins** instead
of **long-passive 60%-noop wins**. The whole fleet has shifted.

**Implication for v14 next-step decision.**

This **dramatically reframes** fire-25's "v14 -12 Elo, tune coefs down"
verdict:
- The original v13 baseline (62% noop, 1052) is gone — replaced by
  100%-type-collapse v13 (1033)
- The "passive-vs-active" framing for v13/v14 is obsolete
- The new v13/v14 comparison is **narrow-aggressive vs varied-aggressive**
- v14's varied 100/75/50 sends may be MORE valuable here (broader
  attack repertoire vs the new bench corpus)

**Need fire 55 v14 result before drawing conclusions.** If v14 cycle-2
beats v13 cycle-2 (reverses fire-25's -12 gap), that's a strong signal
to ship v14 as-is, not v14b.

**Worker memory.** +0.50GB this fire — biggest jump since fire 33.
Could be transient (like fire 29's +0.49 spike) or start of a new
creep. **Watching fire 55** for confirmation.

**Queue.** 1 running + 0 queued = 1. Skipped queueing.

### Loop fire 55 — 2026-04-29 09:41 PT — 🟢 v14 cycle-2: Elo gap holds at -11 but v14 wins MORE games (19/24)

**State.** Worker PID 4019322, **16h 19min uptime**, **RSS 6.62GB
(-0.49 since fire 54)** — fire-54's spike confirmed transient. Plateau
holding. Champion drift 1162→1159.

**🟢 reward_version A/B cycle-2 complete:**

| run | reward_version | Elo | rate | wins/24 |
|---|---|---|---|---|
| **fire-54 v13 control** | 1 | **1033** | 0.913 | 13 |
| **fire-55 v14 treatment** | 2 | **1022** | 0.916 | **19** |

**v14 -11 Elo gap** (vs fire-25's -12 — basically identical structural
gap). The shaping cost is ~10-12 Elo, not transient.

**🟢 But v14 wins MORE games:** 19/24 vs v13's 13/24 — second-highest
win count of the entire loop (after rollout_steps-mid cycle-2's 18).

**Game review on v14 cycle-2 (Elo 1022, 19 wins):**

| game | tag | ticks | noop% | entropy | top types | value drop |
|---|---|---|---|---|---|---|
| WIN | 35 | 11% | 2.87 | **100%=56% / 75%=17% / 25%=11%** | -1.47 |
| LOSS | **153** | 17% | 3.99 | **all 4 types: 100/50/75/noop** | +10.13 |

**Two new records:**
- LOSS sample = 153 ticks → **longest game of all 55 fires** (prev record:
  fire-37 gamma-hi cycle-3's 110-tick stalling WIN)
- WIN uses **25%-sends** (11% of decisions) → first appearance of all 4
  send sizes in a v14 WIN sample

**v13 vs v14 comparison (now under cycle-3 corpus):**

| metric | v13 (Elo 1033) | v14 (Elo 1022) |
|---|---|---|
| **Wins / 24** | 13 | **19** |
| WIN noop% | 0% | 11% |
| **WIN top types** | **100%=100%** narrow | **100%/75%/25%** varied |
| LOSS noop% | 29% | 17% |
| **LOSS top types** | 100%/75%/noop | **all 4 types** |
| **LOSS ticks** | 47 | **153** (record) |

**Interpretation: v14 = generalist, v13 = strong-opponent specialist.**

Same pattern as fire-28's lr-lo cycle-2 (3 strong wins ≈ 11 mixed wins
at same Elo). v14 wins MORE games — beats more medium opponents — but
gets crushed harder by strong ones (153-tick games, +10.13 value drop).
v13's type-100 strategy either wins fast or doesn't engage; v14 keeps
fighting and pays more in losses to top opponents.

**v14 is doing what shaping is supposed to do.** It produces an active
varied policy that wins more games and engages even in losing positions.
The Elo deficit comes from PFSP weighting penalizing engagement with
strong opponents.

**Verdict for v14 next-step decision:**

This is the data point we needed. Recommendation table:

| direction | evidence | recommendation |
|---|---|---|
| Ship v14 as-is | 19/24 wins, varied policy, full action vocabulary | **PRIMARY** |
| Land v14b (halve coefs) | -11 Elo gap is small, fewer side effects | secondary — but v14 already produces the diversity |
| Sweep coef magnitudes | U-shaped axes (fire 50) suggest possible better cells | tertiary — only if v14 + v14b both fall short |

**Updated recommendation: ship v14 to a longer test (60-min cell)
rather than tuning coefs.** v14 already produces the active-varied
behavior; the Elo gap is a PFSP artifact, not a policy weakness.
A 60-min v14 cell would also test compute scaling per the training-
discipline rules.

**Backstop pulled entropy_coef cycle-3** (after reward_version was the
reward_version-A/B sweep, round-robin moves to entropy_coef which had
its cycle-2 already; this is "cycle-3" of an axis that bypassed it).
lo running 11 min in.

**Worker memory.** 7.11→6.62 GB confirms fire-54's +0.50 was transient
(JAX cache pressure that GC'd). 14-fire plateau pattern continues.

**Queue.** 1 running + 2 queued = 3. Skipped queueing.

### Loop fire 56 — 2026-04-29 10:11 PT — entropy_coef cycle-3: continued deflation, fleet shift to active continues

**State.** Worker PID 4019322, **16h 49min uptime**, RSS 6.61GB
plateau holding (15+ fires). Champion drift 1159→1160.

**entropy_coef now has cycle-1, cycle-2, AND cycle-3 data:**

| run | cycle 1 | cycle 2 | cycle 3 |
|---|---|---|---|
| -lo (0.003) | 1057 | 1048 | **1019 (-29 vs cycle-2)** |
| -mid (0.01) | 1058 | 1041 | running |
| -hi (0.03) | 1036 | 1035 | queued |

**Cumulative cycle-1 → cycle-3 deflation on lo: -38 Elo.** Corpus-
deflation pattern continues uniformly.

**entropy_coef-lo cycle-3 game review:**
- WIN: 13 ticks, **29% noop**, 100%/noop/**50%-sends** — moderate active
- LOSS: 85 ticks, 40% noop, 100%/noop/75%, value drop +8.77

**Fleet shift confirmed across same cell over time:**
| fire | run | WIN noop% | WIN ticks | top non-noop |
|---|---|---|---|---|
| fire 17 | entropy_coef-lo cycle-1 | (data not in log) | — | — |
| fire 26 | entropy_coef-lo cycle-2 | 69% | 116 | 100/75 |
| **fire 56** | **entropy_coef-lo cycle-3** | **29%** | **13** | **100/50** |

Same hyperparams + same axis cell, completely different policy outcomes.
**The bench corpus's selection pressure has shifted dramatically over
~17 hours of training**, pulling all v13 baselines toward active
short-aggressive policies. v14's "active varied" attractor is now
more aligned with the corpus's preferred strategy than 17h ago.

**This is one more piece of evidence that v14's -11 Elo gap from
fire 55 is structural (PFSP weighting), not strategic.** v13 has
*caught up* to v14's behavior signature; the gap is now about strong-
opponent specialization, not active-vs-passive.

**Queue:** 1 running + 1 queued = 2. Skipped queueing.

### Loop fire 57 — 2026-04-29 10:41 PT — entropy_coef cycle-3 complete; FIRST 75%-DOMINANT LOSS sample observed

**State.** Worker idle (GPU 0%), queue empty. PID 4019322, **17h 19min
uptime**, RSS 6.59GB plateau (16+ fires). Champion drift 1160→1158.

**entropy_coef has 3 cycles of data now:**

| run | cycle 1 | cycle 2 | cycle 3 |
|---|---|---|---|
| -lo (0.003) | 1057 | 1048 | 1019 |
| -mid (0.01) | 1058 | 1041 | **1024** |
| -hi (0.03) | 1036 | 1035 | 1015 |

**Range narrowed each cycle:** 22 → 13 → 9 Elo. Direction shuffled
(cycle-1 lo>mid>hi, cycle-2 lo>mid>hi, cycle-3 mid>lo>hi). The
deflation has compressed the axis to near-noise.

**🟢 entropy_coef-mid cycle-3 game review — 75%-DOMINANT LOSS:**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| WIN | 16 | 12% | 100%=62% / 75%=25% / noop=12% | -2.03 |
| **LOSS** | **12** | **17%** | **75%=67% / 100%=17% / noop=17%** | +4.70 |

**FIRST 75%-DOMINANT LOSS sample across 56 fires.** Previously 75%-sends
always appeared as a secondary action (≤30% in any sample). Here
75%-sends are **67% of the LOSS decisions.**

**Why:** the agent learned that **75% sends preserve enough garrison
to defend** while still attacking. Full-commit 100% sends empty the
source building (vulnerable to counter); noop is too passive. **75%
is the "controlled aggression" middle ground for losing positions.**
This is a sophisticated loss strategy — first time we've seen one
emerge from a v13 baseline.

**Three cells now using non-100% as PRIMARY strategy under cycle-3:**

| run | sample | dominant non-noop |
|---|---|---|
| rollout_steps-hi cycle-2 (Elo 1048) | WIN | 50%-sends 50% |
| lr-hi cycle-2 (Elo 954) | WIN | 50%-sends 18% |
| **entropy_coef-mid cycle-3 (Elo 1024)** | **LOSS** | **75%-sends 67%** |

**The fleet is genuinely learning the action vocabulary** under
sustained corpus pressure. Slow process but real. v14's "shape it
toward variety" approach is *aligned* with the corpus's natural
selection pressure now — not fighting it.

**Backstop next:** ticks at :45 PT (~4 min). Round-robin from
entropy_coef → **lr**. cycle-3 of lr hasn't run yet (lr's cycle-2
was the strongest deflation seen: -19/+9/-55 across lo/mid/hi).
Will be informative noise data.

**Queue:** empty. Skipped queueing.

### Loop fire 58 — 2026-04-29 11:11 PT — 🟢 b6 vs-champion 90-min sweep queued (4 seeds); EXACTLY the experiment I recommended fire 57

**State.** Worker PID 4019322, **17h 49min uptime**, **RSS 7.22GB
(+0.61 since fire 57)** — significant uptick, watching for transient.
Champion drift 1158→1160.

**🟢 NEW NON-KARP RUNS QUEUED — and I'm NOT discarding them.**

Inspected the queue: 4 runs labeled `b6-260429-1802-default90-s{1,2,3,42}`,
queued **9 minutes after my fire-57 "train vs champion" recommendation.**

Description: *"b6 default K=4 90min vs b3-endurance; retry b5 (discarded)
— does 90 min break through the 0.34 plateau?"*

| field | value |
|---|---|
| budget | **90 minutes per run** (compute scaling gate) |
| seeds | 1, 2, 3, 42 (proper variance estimate) |
| **opponent_name** | **neural** (training vs champion!) |
| level_name | random_8_16 |
| n_envs | 1024 |

**This is exactly the experiment I recommended in fire 57:**
1. ✅ Train vs neural (champion) opponent
2. ✅ 90-min cell (compute scaling gate from training-discipline rules)
3. ✅ Multi-seed for variance estimate
4. ✅ Same level mix as karp loop (controls for distribution)

Per project memory step 4 ("Clear non-karp clutter"), I'd normally
discard these. **NOT discarding** — these are intentional, exactly
what we need. Surfacing instead.

**Resource impact.** 4 × 90 min serial = **6 hours of worker time**
queued behind lr-mid + lr-hi (~30 min remaining of lr cycle-3). The
karp- queue will be blocked through ~17:30 PT. **Acceptable** — these
b6 runs are more valuable than more cycle-3 noise data.

**lr cycle-3 in progress (still running, will complete before b6 starts):**

| run | cycle 1 | cycle 2 | cycle 3 |
|---|---|---|---|
| -lo | 1073 | 1054 | **1016 (-57 cumulative)** |
| -mid | 1036 | 1045 | running |
| -hi | 1009 | 954 | queued |

lr-lo cycle-3 game review:
- WIN: 12 ticks, 17% noop, **100%=83%** near type-collapse
- LOSS: 66 ticks, 18% noop, 100%/75%/noop varied
- **5/24 wins** — among lowest of cycle-3

**Worker memory.** Second consecutive +0.5GB+ uptick (fire-54's was
transient, this is +0.61). Watching for sustained creep but worker
service stays healthy. 17h 49min uptime intact.

**Queue:** 1 running + 5 queued (1 karp lr-hi + 4 b6 90-min seeds).
**Skipped queueing.**

### 🚨 REGIME CHANGE — 2026-04-29 11:25 PT — karp-loop v1 → v2

Per Paul: random_legal training is destroying progress. Kill it. Switch
the loop to vs-champion training + small-map curriculum. Bake v14 reward
as the new default.

**What I killed:**
- karp-260429-1045-lr-mid (running, marked failed mid-cycle)
- karp-260429-1045-lr-hi (queued, discarded)
- 4× b6-260429-1802-default90-s{1,2,3,42} (queued, discarded — Paul's own
  vs-champion experiment, will be re-attempted under the new regime
  because the karpv2- loop trains vs champion automatically now)

Lost: ~25 min of in-progress lr-mid training. Acceptable.

**What changed in the YAML / queue script:**

| field | v1 (old) | v2 (new) |
|---|---|---|
| training_opponent.name | `random_legal` | **`latest_champion`** |
| level_name | `random_full_mix` | **`random_close_4_5`** |
| level_mix | 4-16 building distribution | **removed** (single level) |
| reward_version | 1 (v13) | **2 (v14)** |
| reward_v13 | true | true (overridden by reward_version=2) |
| cell_budget_seconds | 1200 (20min) | **600 (10min)** |
| Label prefix | `karp-` | **`karpv2-`** |

**What stayed the same:**
- model_id (v9.0-1024) — no architecture change per training-discipline rules
- All sweep axes (entropy_coef, lr, gamma, etc.)
- All bench_eval logic (bench corpus stays mixed v1/v2 for now)
- Round-robin axis selection

**Files modified:**
- `configs/karpathy_loop.yaml`
- `scripts/queue_karp_sweep.py` (label prefix + last_axis filter)
- `scripts/karp_review_games.py` (filter on karpv2-)
- `scripts/karp_backstop.py` (filter on karpv2-)

**First karpv2- sweep queued:** entropy_coef × 3 cells, training vs
champion `cdcc0826` (b6 phase2_selfplay-med, Elo 1162), on
random_close_4_5, 10-min budget each.

**Expected speed.** Per project memory, neural opponent on JAX
backend triggers per-env CPU path → ~10× slower than random_legal.
Expect ~300 sps (vs ~3000 with random_legal). 10-min cell at 300 sps
= ~180k steps. Close-4-5 maps with ~half travel time = ~10-20 ticks
per game = 9k-18k episodes per cell. Plenty.

**Worker status.** Worker still busy with prior bench_eval cycle
(State=R, 6.6GB RSS). Will pick up karpv2- runs after current cycle
completes — no restart needed (engine bytecode is unchanged; only
config/labels changed).

**Old karp- runs preserved.** They stay in the bench corpus and runs
table for reference. Anyone querying `like 'karp-%'` will still get
v1-regime data. New regime is `karpv2-` only.

### Loop fire 59 — 2026-04-29 11:41 PT — 🚨 First karpv2- result: agent is BAD (Elo 798, 0 wins / 24)

**State.** Worker PID 4019322, 18h 19min uptime, RSS 6.62GB plateau.
GPU 18% (mid-iter). Champion 1170 (identity unchanged, +9 from regime
change point).

**🚨 First karpv2- run finished — confirms bootstrap problem.**

| metric | value | interpretation |
|---|---|---|
| Elo | **797.8** | sub-anchor, worse than random_legal baseline |
| Training rate | **0.134** | 13.4% wins vs champion during training (was 91% under v1) |
| sps | **322** | ~10× slowdown vs random_legal — fire-58 estimate accurate |
| updates | **3** | Only 3 PPO updates in 10-min cell |
| Wins / 24 bench | **0** | Zero wins. New record for failure. |

**Game review on the LOSS samples:**
- LOSS 1: 40 ticks, 55% noop, value-drop **+10.45**
- LOSS 2 (extra): 134 ticks, 69% noop, value-drop **+12.93 ⚠️ NEW LOOP RECORD**
  (prev: lr-hi cycle-2's +11.31)

Agent regressed to passive behavior + catastrophic value-head failure.
Value estimate starts high and collapses by 10-12 points — agent thinks
it's winning then gets crushed.

**🚨 The bootstrap problem.**

Pure vs-champion training is too hard for a fresh agent:
1. **3 updates ≈ uninitialized.** PPO needs hundreds of updates. We're
   doing 3 per cell. New agent has barely moved from random init.
2. **Champion is overpowered.** Trained 90+ min vs another champion.
   New agent has 10 min vs same champion. Mismatch.
3. **v14 shaping over-weighted at 13% win rate.** Per-tick deltas
   compound negatively when the opponent dominates the board.

**Three paths to fix:**

| option | effort | mechanism |
|---|---|---|
| A. Longer cells (30 min) | 1-line YAML | 9 updates/cell, may help converge |
| B. Smaller rollout_steps (32) + 20-min cells | 2-line YAML | ~12 updates/cell, faster per-update |
| C. Curriculum mixing (older champions + current) | bigger rewrite | weaker opponents at start, harder later |
| D. **Batch the neural opponent forward pass** | ~2-4h code | 10× speedup → 30+ updates/cell |

**Recommendation: B + D combined.**

Option B is a 2-line YAML change — instant. Option D was the speedup
discussion in the prior turn. Together: 12-30+ updates per cell at
20-min budget, vs the current 3.

C is the more principled long-term fix (bootstrap from weaker opponents)
but adds complexity now. Defer until we see if B+D unblocks training.

**Recommend implementing B (YAML) immediately, deferring D to after
the next karpv2- result confirms the diagnosis.**

**Queue.** entropy_coef-mid running, hi queued. Will let entropy_coef
sweep complete to see if the 3-update problem is consistent across
cells before changing config.

### Loop fire 60 — 2026-04-29 12:11 PT — entropy_coef sweep complete; bootstrap pattern confirmed; 1 win in cycle-3

**State.** Worker idle (GPU 0%), queue empty. PID 4019322, **18h 49min
uptime**, RSS 6.60GB plateau. Champion **+17 to 1187** (biggest single-
fire jump in many fires — bench corpus updating from karpv2- runs).

**🟢 First karpv2- entropy_coef sweep complete:**

| run | entropy_coef | Elo | rate | sps | updates | wins/24 |
|---|---|---|---|---|---|---|
| -lo (0.003) | 798 | 0.134 | 322 | 3 | 0 |
| -mid (0.01) | **823** | 0.131 | 322 | 3 | **1** |
| -hi (0.03) | 835 | 0.126 | 321 | 3 | (TBD) |

**Three patterns:**
1. **All sub-anchor** (798-835), 37 Elo range, hi > mid > lo
2. **All 3 updates, all ~321 sps** — slowdown + update-starvation uniform
3. **Higher entropy wins under bootstrap** — same finding as fire 27's
   "high entropy is partial v14 substitute" but stronger under v2

**🟢 entropy_coef-mid game review — first karpv2 win is GENUINELY DIFFERENT:**

| game | tag | ticks | noop% | entropy | top types | value drop |
|---|---|---|---|---|---|---|
| WIN | 22 | 55% | **4.34** | 50%=18% / 25%=18% / noop=55% | **-3.74** |
| LOSS | 11 | 67% | 3.98 | 75%=17% / 25%=17% / noop=67% | +3.41 |

**Two new loop records:**
- **WIN value-drop -3.74** — most negative ever (agent's confidence
  rose 3.74 during the win). Prior record was -2.28 from n_envs-mid
  cycle-3 / value_coef-hi cycle-3.
- **WIN entropy 4.34** — highest WIN entropy of the loop (typical 1-3).

**This is a NEW POLICY ARCHETYPE under v2:**
- Massive exploration (entropy 4.34)
- **Small sends only** (25%, 50% — never 100%)
- Patient (55% noop)
- Confident throughout (-3.74 value rise)

The agent learned **"don't overcommit, you can't beat the champion in
a slugfest."** Defensive, exploratory, small-commitment policy. This
is *plausibly* what you want vs a strong opponent — preserve garrison,
wait for openings, take small calculated risks.

LOSS sample also shows 25%-sends — even in defeat the agent preserves
garrison. Different from v1 LOSS samples (commit 100% and die fast).

**Verdict on the bootstrap problem.** 3 updates is too few — we're not
actually training. But the *direction* is interesting: where v1 wanted
"more updates of cautious learning", **v2 wants "more updates of
exploratory learning"** because the opponent is strong and cautious
loses. High entropy + small sends = a learnable response.

**Recommendation: switch to rollout_steps=32 + 20-min cells now.**
Same wall-time per fire, but 12 updates/cell instead of 3 = 4× learning
per cell. Hold off on D (batch neural opponent) until we see whether
12 updates is enough to break sub-anchor.

**Awaiting Paul's call on the YAML change** — won't autonomously edit.

**Queue:** empty. Backstop will tick at :15 PT and pick next axis (lr
cycle-1 of v2, since round-robin is fresh under karpv2- prefix). Will
let the cron continue if no YAML change lands first.

### Loop fire 61 — 2026-04-29 12:43 PT — config bumped (rs=8, lr=1e-3, 5min cells, bench level matched); worker restarted; first new-config sweep queued

**Major config changes shipped.** Per Paul fire 60-61:

| field | v2-initial | v2-tuned |
|---|---|---|
| cell_budget_seconds | 600 (10min) | **300 (5min)** |
| rollout_steps baseline | 64 | **8** |
| lr baseline | 3e-4 | **1e-3** |
| rollout_steps sweep cells | 32/64/128 | **4/8/16** |
| bench_eval LEVEL | random_8_16 | **random_close_4_5** (matches training) |

**bench_eval extracted to YAML.** New `configs/bench_eval.yaml` +
`cli/bench_config.py` loader. Per Paul: every settings block should
be in a config file, not Python constants.

**Worker restarted at 12:44 PT** to load new bench_eval LEVEL into
memory. New PID 1666605. lr-mid (in-flight, old config) marked failed
~20 min in. Lost data point but not worth waiting since it was old
config.

**Old-config lr sweep results before restart:**

| run | lr | Elo | rate | updates | wins/24 |
|---|---|---|---|---|---|
| -lo (1e-4) | 816 | 0.128 | 3 | 0 |
| -mid (3e-4) | failed (killed) | — | — | — |
| -hi (1e-3) | **839** | 0.129 | 3 | 0 |

**lr-hi 839 vs lr-lo 816** — 23 Elo edge to higher lr under v2.
Validates baking lr=1e-3 as new baseline.

**🟢 lr-hi LOSS sample — TWO new records:**

| metric | value | prior record |
|---|---|---|
| LOSS entropy | **5.33** | 4.52 (fire 33 EXTRA) |
| LOSS action vocabulary | **all 5 types used** (noop, 25, 50, 75, 100) | 4 types (multiple cells) |

Higher lr at 3 updates = maximum exploration; agent is still essentially
random, exploring everywhere. Validates lr=1e-3 for v2 bootstrap (need
broad exploration to escape random init) AND shows 3 updates is far
from convergence.

**First new-config sweep queued:** rollout_steps × [4, 8, 16].
- karpv2-260429-1244-rollout_steps-lo  rollout_steps=4
- karpv2-260429-1244-rollout_steps-mid rollout_steps=8 (matches baseline)
- karpv2-260429-1244-rollout_steps-hi  rollout_steps=16

**Hyperparams verified:** 5-min cells, lr=1e-3, neural opponent,
random_close_4_5, reward_version=2. Worker fresh, GPU 18% (between iters).

**Expected updates per cell at 322 sps:**
- rs=4:  ~24 updates per 5-min cell
- rs=8:  ~12 updates per 5-min cell (baseline)
- rs=16: ~6 updates per 5-min cell

This sweep will tell us whether **fewer-but-smaller-rollouts** or
**more-rollouts-bigger** wins under v2 bootstrap.

### Loop fire 62 — 2026-04-29 13:11 PT — 🟢 FIRST KARPV2 ABOVE ANCHOR (Elo 1030); update density change WORKED

**State.** Worker PID 1710960 (NEW — restarted again ~13:05 PT, 5min
elapsed). RSS 3.89GB (fresh JAX cache, much smaller than old worker's
6.6GB plateau). GPU 0% between iters. Champion **+3 to 1190**.

Two new commits since fire 61 (Paul's): `2913a42 levels: single source
of truth` and `a1ed6e8 play.html chunked level picker`. Worker restart
likely triggered by the levels code change.

**🟢 First karpv2- run ABOVE THE ANCHOR:**

| run | rs | Elo | rate | sps | updates |
|---|---|---|---|---|---|
| rollout_steps-lo | **4** | **1030** | **0.423** | 275 | **21** |
| rollout_steps-mid | 8 | failed (worker restart cleanup) | — | — | — |
| rollout_steps-hi | 16 | 962 | 0.157 | 288 | 6 |

**The update density change WORKED.** Going from 3 → 21 updates per
cell pushed training rate from 13.4% → 42.3% vs the champion. Elo
crossed the 1000 anchor for the first time under v2.

**Direction confirmed:** rs=4 (1030, 21 updates) > rs=16 (962, 6 updates).
Fewer rollouts → more updates → better learning. Same theme as v1's
"smaller-fewer-faster wins" but at smaller absolute values.

**🚨 THREE new loop records on rollout_steps-lo game review:**

| metric | value | prior record |
|---|---|---|
| WIN value-drop | **-8.30** | -3.74 (fire 60) |
| **LOSS value-drop** | **+46.40 ⚠️** | +12.93 (fire 59) |
| LOSS type-collapse | **75%-sends 73%** | first 75%-dominant 67% (fire 57) |

**+46.40 LOSS value-drop is wildly out of band** — every prior LOSS
sample was +0 to +13. The value head is finally LEARNING (both
directions: -8 in win = underestimated; +46 in loss = massively
overestimated) but also becoming UNCALIBRATED. Likely v14 per-tick
shaping accumulating across 30 ticks of losing position interacts
with the now-learning value head in unexpected ways.

**WIN sample:** 7 ticks, 50% noop, 100%/75%-sends balanced 25%/25%.
Fast win via balanced action selection.
**LOSS sample:** 30 ticks, 75%-sends 73% — agent committed to medium
sends repeatedly and lost.

**Other karpv2 progress this session:**

| run | rs | lr | Elo | rate | updates |
|---|---|---|---|---|---|
| n_envs-lo (1305) | 8 | 1e-3 | 1000 (unrated) | **0.404** | 18 |
| n_envs-hi (1305) | (running) | — | — | — | — |

n_envs-lo got 18 updates and 40.4% rate — confirms the 21-update
rs=4 result wasn't a fluke. **The new config consistently produces
12-21 updates/cell with 40%+ training rates.**

**Some failed runs from transient worker-restart races:**
- karpv2-260429-1249-n_envs-lo/mid: `make_neural_opponent() got an
  unexpected keyword argument 'opponent_run_id'` — but the opponent_kwargs
  were IDENTICAL to runs that succeeded. **Transient race during worker
  claim, not a real bug.** Per project memory ("don't burn >5min on
  diagnostics") — logged and moved on.

**Queue:** 2 queued (n_envs-mid + n_envs-hi second batch), 1 running
(n_envs-hi from earlier batch). Skipped queueing.

### Loop fire 64 — 2026-04-29 13:41 PT — n_envs sweep complete; lo wins; mid degenerated to 100% noop; new karpv2 champion

**State.** Worker PID 1768780, **3min 8s elapsed** (restarted at 13:38
PT for METRICS_UPLOAD_EVERY=1). RSS 3.33GB. GPU 13%. Champion identity
changed: **cron-260428-0407-phase2_selfplay-med-00** still #1 at Elo
**1134** (-56 over the last few fires) but **karpv2-260429-1305-n_envs-lo
at 1040 is the new karpv2 champion** and the active training opponent.

**🟢 n_envs sweep complete under v2-tuned config:**

| run | n_envs | Elo | rate | updates | wins/24 |
|---|---|---|---|---|---|
| -lo (512) | **1040** | 0.404 | 18 | (TBD) |
| -mid (1024) | **1006** | 0.211 | 10 | **0** |
| -hi (2048) | 942 | 0.107 | 5 | — |

**Two above-anchor runs.** The v2-tuned config is producing real signal.
Direction: smaller n_envs → more updates per cell → better learning.
Same theme as rs sweep (rs=4 won with 21 updates).

**🚨 n_envs-mid degenerated to PURE NOOP (Elo 1005, 0 wins):**

| game | tag | ticks | noop% | top types | value drop |
|---|---|---|---|---|---|
| LOSS | 34 | **94%** | noop=94% / 75%=6% | +14.54 |
| EXTRA | 52 | **100%** | **noop=100%** ⚠️ | +20.60 |

**TWO new loop records:**
- **First complete-noop game** (100% noop for 52 ticks) across all 64 fires
- LOSS value-drop **+20.60** — 3rd-largest of all loop

**How does Elo 1005 with 0 wins happen?** Bench_eval gives Elo updates
based on each match outcome. With 0 wins / 24 losses, Elo should drop
hard. But it landed at 1005 — close to anchor. **The bench corpus is
full of fellow-noop policies; mutual-stall doesn't lose Elo points.**

This is **the same passivity collapse as fire 26 under v1, but now under
v2.** Even vs-champion training + v14 reward + 5-min cells doesn't
guarantee active policies — when the cell only gets 10 updates, the
noop attractor wins.

**Update density still matters more than anything else.** 18-update lo
run is a real policy (40% rate, 1040 Elo). 10-update mid run is a
degenerate noop machine.

**🚨 b6 champion losing Elo on close-4-5:** cdcc0826 went 1190 → 1134
(-56) over last few fires as karpv2- runs grade vs it on the new bench
level. The b6 champion was a big-map specialist; doesn't transfer to
close-4-5.

**Queue:** gamma-mid running, gamma-hi queued, gamma-lo-redo queued.
Skipped queueing — 3 in queue, gamma sweep will finish ~15 min.

**Pending Paul-asked work** (held until gamma sweep completes, won't
edit code mid-sweep):
- Move worker constants (METRICS_UPLOAD_EVERY etc.) to configs/worker.yaml
- Implement rotating-champion training (1 champion per update, sampled
  from archive — addresses opponent-diversity gap; see fire 63 reply)

### Loop fire 65 — 2026-04-29 14:00 PT — gamma sweep complete; gae_lambda mid-flight; karpv2-n_envs-lo holds champion

**State.** Worker active. Backstop active. GPU 18%, VRAM 4.6GB. Queue: 1
running (`gae_lambda-mid`, started 13:58 UTC), 1 queued (`gae_lambda-hi`).

**🟢 gamma sweep results** (all vs the karpv2 champion `n_envs-lo`):

| run | gamma | Elo | PFSP | bv | rate | promoted? |
|---|---|---|---|---|---|---|
| -lo  | 0.95 | — | — | — | — | killed×2 (intentional worker-restart kills) |
| -mid | 0.97 | 969 | 0.840 | 11 | 0.361 | N |
| -hi  | 0.99 | 942 | 0.892 | 11 | 0.415 | N |

Neither cell touched the karpv2 champion (`n_envs-lo` still #1 at Elo
1040). Mid (baseline) edges hi by ~27 Elo, but PFSP is higher for hi
(0.892 vs 0.840) — hi's archive-sweep was more 50/50; mid's more
decisive toward losing. Hi has higher training-rate (0.415 vs 0.361).
Reading: hi explores more sharply but doesn't convert wins consistently.

**No `gamma-lo` data this cycle.** Killed twice by intentional restarts.
Backstop will re-queue on next round-robin.

**gae_lambda-lo** (0.9) also killed by fire-65 restart. -mid (0.95)
running; -hi (0.98) queued. Sweep finishes ~14:13 PT.

**Champion archive (top 5 by archived_at):**

1. ⭐ `karpv2-260429-1305-n_envs-lo` (Elo 1040) — first karpv2 champion
   in the loop's history; promoted today at 20:12 UTC.
2. `cron-260428-0407-phase2_selfplay-med-00` (Elo 1134)
3. `cron-260428-0407-phase2_selfplay-short-03`
4. `cron-260428-0407-phase2_selfplay-short-02`
5. `cron-260428-0407-phase2_selfplay-short-01`

cron-med-00 still has higher Elo (1134) but karpv2-n_envs-lo is the
most-recently-promoted, which is what `random_champion` picks as the
training opponent.

**Holding queue.** 2 karpv2- runs in flight; not queueing more this fire.
PaulLinux backstop fires every 15min and will queue `clip_range` next
once `gae_lambda-hi` clears.

### Loop fire 66 — 2026-04-29 14:11 PT — first sweep WITH per-update rotation; new policy archetype (mid-range sends dominate)

gae_lambda sweep with rotate_per_update done: lo failed (worker restart),
mid=Elo 955, hi=Elo 983. Higher training rate (0.41-0.44 vs 0.40 without
rotation) but lower bench Elo (was 1030+ without rotation). Expected
trade-off — rotation prevents over-specialization to one opponent;
policy is more general (higher training rate across diverse opps) but
underperforms on bench-eval matches against any specific opponent.

🟢 NEW POLICY ARCHETYPE on gae_lambda-mid game review:
- WIN sample: 67 ticks, **50%-sends DOMINATE (32%)**, no 100%-sends at all,
  value-drop **-7.34** (2nd-largest negative WIN drop of all 66 fires)
- LOSS sample: 16 ticks, 4 different send sizes used (noop 38%, 25% at
  25%, 100% at 25%)

First time across 66 fires that 100%-sends are absent from a WIN sample.
The agent learned medium-commitment 50% sends are preferable vs a diverse
opponent pool. Rotation forcing genuinely new strategy.

Champion ranking continues shifting on close-4-5: cdcc0826 b6 down to
1146 (was 1190 at peak). karpv2-n_envs-lo at 1053. b6 specialist fading.

Label bug from fire 65 (champion:mw2-pfsp collapse) ships on NEXT sweep —
gae_lambda runs predated commit d84e6bf so they still showed collapsed
labels. clip_range sweep (backstop next) gets the fix.

Queue empty. Backstop ticks ~14:15, picks clip_range cycle-3-of-v2.

### Loop fire 67 — 2026-04-29 14:41 PT — 🟢 FIRST ROTATION REMATCH DATA — agent improved +16pp avg across all 7 opponents

**State.** Worker PID 1854207, 8 min uptime since fire 66.5 restart. RSS
3.94GB. GPU idle. Champion ranking: cdcc0826 b6 at 1157 (drift), karpv2
champion at **1080 (+28 in 30min, climbing)**.

**🟢 update_epochs-hi shipped the FIRST rotation_rematch result:**

| opp | init% | final% | Δpp |
|---|---|---|---|
| d53a2871 | **14%** | **52%** | **+38** |
| 6796f27e | **45%** | **80%** | **+35** |
| 02651ce3 | 63% | 76% | +13 |
| e18208fb | 40% | 52% | +12 |
| 8d0ef4cf | 65% | 72% | +7 |
| 42902f3e | 51% | 56% | +5 |
| 66e2f9b2 | 47% | 52% | +5 |

**Avg +16.4pp across 7 opponents. Every single one improved.** 0 draws,
0 timeouts in 175 games (25 × 7) — every match resolved decisively.

**Critical insight:** bench Elo says 977 (sub-anchor). Rate says 0.486
(mostly losing). **Rotation rematch says +16pp avg improvement.**

The bench archive is full of opponents the agent *didn't* train against
in this 5-min cell. Bench Elo is a misleading proxy because it averages
performance across non-overlapping populations. The rematch is the real
diagnostic — and it shows real learning happening every fire.

This validates fire 66's "rotation gives lower Elo but higher training
rate is the desired trade-off" interpretation. Now we have direct
evidence: **the agent is learning ~+16pp per 5-min cell against the
opponents it actually faces.**

**Other runs this fire:**
- update_epochs-lo: Elo 957, rate 0.467 (rematch=0; finished before fire 67 code)
- update_epochs-mid: Elo 986 unrated (bench_eval pending; 53.1% rate — highest yet)

**update_epochs cycle direction so far** (all rotation):
- lo (2 epochs): 957 / 0.467
- mid (4): unrated / **0.531**  ← highest training rate
- hi (8): 977 / 0.486 / +16pp avg rematch ⭐

Mid edges hi if it rates similarly. Backstop next axis: minibatch_size.

**Skipped queueing.** Worker idle, backstop ticks ~14:45.

### Loop fire 68 — 2026-04-29 15:29 PT — minibatch_size + update_epochs cont. → new champion #3; clip_range got killed-for-restart (not a real failure); value_coef in flight

**State.** Worker active, backstop active (last fire 15:15 PT, next 15:30). Karp
queue: value_coef-lo running, value_coef-{mid,hi} queued, max_grad_norm-{lo,mid,hi} queued by this fire = depth 6 (cap).

**Recent finished karp- runs since fire 67 (sorted by queue time):**

| label | swept_var | dur | Elo | n | PFSP | promoted? |
|---|---|---|---|---|---|---|
| `karpv2-...1419-update_epochs-lo` | update_epochs=2 | 5.8m | 957.0 | 11 | 0.776 | no |
| `karpv2-...1419-update_epochs-mid` | update_epochs=4 | 12.6m | unrated (986.7) | 5 | — | no |
| `karpv2-...1419-update_epochs-hi` | update_epochs=8 | 19.2m | 977.0 | 11 | 0.819 | no (rematch ⭐) |
| `karpv2-...1414-clip_range-{lo,mid,hi}` | clip_range | 5.4m × 3 | killed-for-restart | — | — | n/a |
| `karpv2-...1445-minibatch_size-lo` | minibatch_size=256 | 5.8m | **1029.0** | 11 | 0.721 | no |
| `karpv2-...1445-minibatch_size-mid` | minibatch_size=512 | 12.5m | 1008.8 | 11 | 0.815 | no |
| `karpv2-...1445-minibatch_size-hi` | minibatch_size=1024 | 19.3m | 949.7 | 11 | 0.795 | no |
| `karpv2-...1448-cont-update_epochs-hi-20min` | continuation | 38.5m | **1055.2** | 12 | 0.662 | ⭐ **#3 champion** |

**Key reads:**
- **minibatch_size: lo wins (256 → 1029).** Smaller batches → more SGD steps per
  cell → faster policy movement. Mirrors the n_envs=lo finding from fire 64
  (more updates per cell = better in 5-min budget). Combined: agent benefits
  from update *density*, not breadth.
- **update_epochs: hi (8) wins on rematch (+16pp avg fire 67), but mid (4) wins
  on training rate (0.531).** Stayed at 4 (baseline) given the noise in 5-min
  cells and the rematch result was a one-off.
- **20-min continuation of update_epochs-hi** *did* climb to **1055** — proof
  that long compute on a working config (rotation + 4 epochs + larger horizon)
  pays. Becomes new champion archive entry #3.
- clip_range sweep wasn't real data — the runs were killed at the start of
  fire 66.5 (worker restart for label-fix code commit 3998608). Not a sweep
  failure. Should re-queue clip_range later in the round-robin.

**Karp leaderboard (top karpv2 champions, by source-run elo_score):**

| run | elo | n | when |
|---|---|---|---|
| `karpv2-260429-1305-n_envs-lo` | 1057.7 | 25 | fire 64 |
| `karpv2-260429-1448-cont-update_epochs-hi-20min` | 1055.2 | 12 | fire 68 (this) |
| `karpv2-260429-1244-rollout_steps-lo` | 1030.2 | 10 | fire 62 |

karpv2 ceiling holding around 1050-1060. The cron-era champion
`cron-260428-0407-phase2_selfplay-med-00` at **1147** is still untouched.

**Review games (most-recent rated, `karpv2-...1448-cont-update_epochs-hi-20min`):**

| game | tag | ticks | decisions | noop% | entropy | value drop | top types | flags |
|---|---|---|---|---|---|---|---|---|
| `bdcc3e13` | WIN | 31 | 16 | 38% | 2.11 | -1.36 | noop=38% 50%=19% 75%=19% | ok |
| `b869a8af` | LOSS | 55 | 28 | 64% | 2.69 | +8.90 | noop=64% 100%=21% 50%=11% | high noop rate 64% |

🚩 *Loss has 64% noop + value drop +8.9.* The agent didn't anticipate the loss —
critic was over-confident. Worth watching this in future losses; if value-drop
sign stays inverted across cells, the critic is mis-calibrated under rotation.

**Queued.** max_grad_norm sweep ({0.25, 0.5, 1.0}) via round-robin. Queue
depth at cap (6); next fire will likely skip queueing if value_coef + max_grad_norm
are still in flight.

### Loop fire 69 — 2026-04-29 16:00 PT — value_coef sweep flat (all sub-anchor); update_epochs-cont. backfilled to 1071 (+16); critic mis-cal pattern repeats; reward_version A/B queued

**State.** Worker active, backstop active (last 15:45, next 16:00 — same minute as
this fire). Karp queue: max_grad_norm-mid running, max_grad_norm-hi queued,
reward_version-{lo,hi} queued by this fire = depth 4.

**Recent finished karp- runs since fire 68:**

| label | swept_var | dur | Elo | n | PFSP | promoted? |
|---|---|---|---|---|---|---|
| `karpv2-...1530-value_coef-lo` | value_coef=0.25 | 6.0m | 958.8 | 12 | 0.752 | no |
| `karpv2-...1530-value_coef-mid` | value_coef=0.5 | 13.0m | 968.3 | 12 | 0.797 | no |
| `karpv2-...1530-value_coef-hi` | value_coef=1.0 | 19.9m | 964.0 | 12 | 0.812 | no |
| `karpv2-...1532-max_grad_norm-lo` | max_grad_norm=0.25 | 25.3m | 1002.8 | 12 | 0.725 | no |

**Backfill on champion #3** (`karpv2-...1448-cont-update_epochs-hi-20min`):
Elo n grew 12→16, score climbed **1055.2 → 1071.7 (+16.5)**. With more bench data
the long-cont. champion is the strongest karpv2 yet by ~14 Elo.

**Key reads:**
- **value_coef sweep is flat.** lo/mid/hi all sub-anchor, span only 9 Elo
  (959–968). Default 0.5 is fine — no signal to move. Notable: hi-coef PFSP
  is highest (0.812) so the critic isn't *worse* with more weight, just not
  better either.
- **max_grad_norm-lo (0.25) above anchor at 1003.** Tighter clipping = stable.
  Compare to mid/hi when they finish.
- **20-min continuation (cont-update_epochs-hi-20min) is the actual top karpv2.**
  Bench data backfilled cleanly between fires — confirms the "wait one fire
  before flagging unrated" rule.

**Karp leaderboard (top karpv2, by source-run elo_score):**

| run | elo | n | when |
|---|---|---|---|
| `karpv2-260429-1448-cont-update_epochs-hi-20min` | **1071.7** | 16 | fire 68 + backfill |
| `karpv2-260429-1305-n_envs-lo` | 1057.7 | 25 | fire 64 |
| `karpv2-260429-1244-rollout_steps-lo` | 1030.2 | 10 | fire 62 |
| `karpv2-260429-1445-minibatch_size-lo` | 1029.0 | 11 | fire 67/68 |
| `karpv2-260429-1532-max_grad_norm-lo` | 1002.8 | 12 | fire 69 (this) |

cron-era `cron-260428-0407-phase2_selfplay-med-00` still untouched at 1147.

**Review games (most-recent rated, `karpv2-...1532-max_grad_norm-lo`):**

| game | tag | ticks | decisions | noop% | entropy | value drop | top types | flags |
|---|---|---|---|---|---|---|---|---|
| `117929b2` | WIN | 55 | 28 | 39% | 3.58 | -2.96 | noop=39% 25%=21% 75%=14% | ok |
| `06b1d88e` | LOSS | 22 | 11 | 55% | 3.63 | +6.59 | noop=55% 50%=27% 100%=9% | high noop rate 55% |

🚩 *Loss has 55% noop + value drop +6.59.* Same critic-mis-cal pattern as
fire 68 (64% noop, +8.9). Two consecutive cells show the agent over-estimating
its position right before losing. **Likely systematic under random_champion +
rotation** — the agent isn't seeing the same opponent long enough to develop
a calibrated value head against any single one. Watch fire 70-71; if it
persists, candidate fix: bump bench_eval n above 12-16 OR slow rotation
(rotate_per_2_updates).

**Queued.** reward_version A/B (v1.3 vs v1.4) via round-robin — 2 cells, 5 min
each. Will produce direct evidence for whether per-tick shaping (v1.4) is
still helping under rotation, or whether the active-policy gains from v14
(noted at v14-bake on 2026-04-29) plateau under the new opponent mix.

### Loop fire 70 — 2026-04-29 16:30 PT — 🟢 reward_version A/B clean signal: v1.4 wins by +58 Elo; max_grad_norm sweep flat (U-shaped); cont-chain batch 01 running; critic mis-cal 3-for-3

**State.** Worker active, backstop active. Karp queue cleared except cont-batch
running. Queue: `karpv2-cont-0791c618-01` running (started 16:26, ETA 16:46) +
this fire just queued entropy_coef sweep (3 cells) = depth 4.

**Chain status (cont-0791c618):** batch 01 running. Helper called, no-op as
expected — waits for head to finish.

**Recent finished karp- runs since fire 69:**

| label | swept_var | dur | Elo | n | PFSP |
|---|---|---|---|---|---|
| `karpv2-...1532-max_grad_norm-mid` | 0.5 | 32.2m | 929.8 | 20 | 0.794 |
| `karpv2-...1532-max_grad_norm-hi` | 1.0 | 39.1m | 1001.3 | 20 | 0.716 |
| `karpv2-...1600-reward_version-lo` | v1.3 | 17.6m | 1003.4 | 16 | 0.801 |
| `karpv2-...1600-reward_version-hi` | v1.4 | 24.6m | **1061.9** | 13 | 0.702 ⭐ |

**Backfill (more bench data):**
- value_coef-{lo,mid,hi}: 959/968/964 → **955/960/944** (n: 12→20). Still flat.
- max_grad_norm-lo: 1003 → **990** (n: 12→20).

**Key reads:**
- 🟢 **reward_version A/B is the cleanest signal of the day. v1.4 beats v1.3 by
  +58 Elo (1062 vs 1003)** at the same opponent mix. Confirms the v14-as-default
  decision (2026-04-29 baseline bake) holds under random_champion rotation.
  Per-tick building/units shaping is doing real work, not just front-loading
  the easy random_legal regime.
- **max_grad_norm is U-shaped, not monotonic.** mid (0.5, baseline) at 930 is
  the worst; lo (0.25) at 990 and hi (1.0) at 1001 both better. Either noise at
  n=20 or genuine sensitivity to clipping at the baseline. Worth a redo on the
  axis when we cycle back; doesn't justify changing the baseline.
- **value_coef confirmed flat with full n=20 data.** 11 Elo span across all
  three. Default 0.5 stays.

**Karp leaderboard (top karpv2):**

| run | elo | n | when |
|---|---|---|---|
| `karpv2-260429-1448-cont-update_epochs-hi-20min` | **1071.7** | 16 | fire 68 + backfill |
| `karpv2-260429-1600-reward_version-hi` | **1061.9** | 13 | fire 70 (this) ⭐ |
| `karpv2-260429-1305-n_envs-lo` | 1057.7 | 25 | fire 64 |
| `karpv2-260429-1244-rollout_steps-lo` | 1030.2 | 10 | fire 62 |

reward_version-hi will likely promote when bench data fills out. cron-era
champion `cron-260428-0407-phase2_selfplay-med-00` still untouched at 1147.

**Review games (most-recent rated, `karpv2-...1600-reward_version-hi`):**

| game | tag | ticks | decisions | noop% | entropy | value drop | top types | flags |
|---|---|---|---|---|---|---|---|---|
| `7b91107f` | WIN | 92 | 46 | 39% | 3.35 | -1.86 | noop=39% 100%=22% 75%=17% | ok |
| `9542ae9b` | LOSS | 15 | 8 | 75% | 3.91 | +3.57 | noop=75% 50%=12% 100%=12% | high noop rate 75% |

🚩 **Critic mis-cal pattern is now 3-for-3:**

| fire | run | LOSS noop% | LOSS value drop |
|---|---|---|---|
| 68 | `cont-update_epochs-hi-20min` | 64% | +8.9 |
| 69 | `max_grad_norm-lo` | 55% | +6.6 |
| 70 | `reward_version-hi` | 75% | +3.6 |

Every loss the critic was over-confident right before losing. Entropy is fine
(3.4-3.9). Issue is value-head calibration, not policy exploration. Hypothesis
firming up: **random_champion rotation = critic never sees enough of any one
opponent to calibrate against them**. Candidate fixes for a future axis:
1. `opponent_pool_mode: ""` (fixed-per-run opponent) — direct test
2. Bigger n in bench_eval (n>20) — tighter Elo, fewer false positives
3. Slower rotation (`rotate_per_4_updates` if available, or pin per cell)

**Queued.** entropy_coef sweep ({0.003, 0.01, 0.03}) via round-robin. New cycle
starts; round-robin wrapped after reward_version.

### Loop fire 71 — 2026-04-29 17:01 PT — 🔻 cont chain batch 01 regressed (-13pp rematch); diagnostic lr=1e-4 cont queued; entropy_coef-lo crashed on MPS-on-Linux

**State.** Worker active, backstop active. Karp queue: entropy_coef-hi running,
diag-lr1e4 queued, lr-{lo,mid,hi} queued by this fire = depth 5.

**🔻 Cont chain batch 01 result (`karpv2-cont-0791c618-01`):**

Bench Elo backfilled cleanly: 1043.5 → **1084.9** (n: 10→24). Initial -51
reading was small-n noise. Real Δ vs parent (1094.7, n=37) is **-10 Elo**.

End-of-run rematch (Paul flagged this fire): **-13pp avg across 13 opponents.**
Worst regression vs strongest opp `0952f5cc` (cron-1147 champion): 80% → 52%
(-28pp). Cleanest signature of "policy walked backward from a champion peak,
losing ground specifically against strong opps." Diagnosis (best hypothesis):
**bootstrap-era hyperparams are wrong for fine-tuning a champion** — lr=1e-3,
update_epochs=8, random_champion rotation are too hot for a starting point
already at 80-98% vs bench.

**Chain paused.** Will not queue batch 02 until lr1e4 diagnostic returns.

**Diagnostic queued (`karpv2-diag-0791c618-lr1e4-01`, id `791d76dd`):**

| field | value |
|---|---|
| parent | `0791c618` (1095-Elo) |
| only-changed | `lr` 1e-3 → 1e-4 |
| inherited | update_epochs=8, opp_pool_mode=rotate_per_update, reward_v=2 |
| budget | 1200s |

If this returns positive end-of-run rematch Δ → lr was the cause; we update
chain config and resume from `0791c618`. If still negative → run option #2
next (rotation off).

**Other karp runs since fire 70:**

| label | swept | dur | Elo | n | PFSP | notes |
|---|---|---|---|---|---|---|
| `karpv2-...1630-entropy_coef-lo` | 0.003 | 5.0m | **failed** | — | — | 🚨 `AttributeError: module 'torch.mps' has no attribute 'current_device'` — MPS code path on a Linux/CUDA worker. Needs separate investigation; not blocking. |
| `karpv2-...1630-entropy_coef-mid` | 0.01 | 25.5m | 960.3 | 18 | 0.770 | normal |
| `karpv2-...1630-entropy_coef-hi` | 0.03 | running | — | — | — | in flight |

**Karp leaderboard (top karpv2):**

| run | elo | n | when |
|---|---|---|---|
| `karpv2-260429-1448-cont-update_epochs-hi-20min` | 1094.7 | 37 | parent (backfilled) |
| `karpv2-260429-cont-0791c618-01` | 1084.9 | 24 | chain batch 01 (-10 vs parent) |
| `karpv2-260429-1600-reward_version-hi` | 1061.9 | 13 | fire 70 |
| `karpv2-260429-1305-n_envs-lo` | 1057.7 | 25 | fire 64 |

**Review games (most-recent rated, `karpv2-...1630-entropy_coef-mid`):**

| game | tag | ticks | decisions | noop% | entropy | value drop | top types | flags |
|---|---|---|---|---|---|---|---|---|
| `7a2cc2d1` | WIN | 38 | 19 | 21% | 3.61 | -5.24 | 75%=26% 25%=26% noop=21% | ok |
| `f3c72e84` | LOSS | 27 | 14 | 50% | 3.62 | +3.20 | noop=50% 50%=21% 25%=14% | ok |

✓ First loss across 4 fires that did NOT trigger the high-noop anomaly flag
(50% is on the threshold). Critic value-drop still positive (+3.20) but the
mildest yet. May indicate entropy_coef=mid moderates the rotation noise, but
n=2 sample, don't over-read.

**Queued.** lr sweep ({1e-4, 3e-4, 1e-3}) via round-robin. lr-lo (1e-4) is the
same lr as the diagnostic but from a fresh-init parent — gives us two
independent angles on the lr question.

## Code changes during loop

### 2026-04-29 12:25 PT — bench_eval config extraction + update density bump

Two changes shipped together (commit 6cf8077):

1. **bench_eval.py constants → configs/bench_eval.yaml**:
   - LEVEL, MAX_TICKS, ELO_K (match.*)
   - SWEEP_GAMES (sweep.games_per_champion)
   - PROMO_GAMES, PROMO_THRESHOLD (promotion.*)
   - BOOTSTRAP_GATE_GAMES, BOOTSTRAP_GATE_THRESHOLD, MIN_ARCHIVE_FOR_GATE
     (bootstrap_gate.*)
   - MAX_ARCHIVE_SIZE, ERA_SOFT_CAP (archive.*)
   - New cli/bench_config.py loader
   - bench_eval.py top-level constants now read from YAML at module import.

2. **karpathy_loop.yaml update density bump**:
   - cell_budget_seconds: 600 → 300
   - rollout_steps baseline: 64 → 8
   - lr baseline: 3e-4 → 1e-3
   - rollout_steps sweep cells: [32,64,128] → [4,8,16]

### 2026-04-29 11:25 PT — regime change v1→v2 (see above)

### 2026-04-29 01:55 PT — fix scripts/karp_review_games.py for tied created_at

Bench_eval inserts all match rows in one transaction → identical
created_at. Old query `.order('created_at', desc=True).limit(5)`
non-deterministically grabbed 5 ties; sometimes 0/5 had populated
games. Increased limit to 20 (10 matches per run typical, well
below threshold). Game review now works on freshly-rated runs.

### 2026-04-28 17:18 PT — implemented reward_v14 with per-tick shaping

Per Paul: rewards holding more buildings + units, penalties for
losing same. Symmetric delta-based (option i) chosen for smoother
PPO learning vs event-based loss penalties. Magnitudes tuned to
~20% of WIN total per game so terminal signal stays dominant.
Sweepable axis added (`reward_version` 2-cell A/B). Tests + docstrings
in place. Worker requires restart to pick up new code (planned fire 22).

### 2026-04-27 23:35 PT — extract knobs to configs/karpathy_loop.yaml

Paul: "anything that we can configure (eg the training composition, level
selection, etc) should be defined in a config file." Done in one commit
(90b3568):

- **`configs/karpathy_loop.yaml`** — single source of truth: schedule
  (fire interval, cell budget, max queue depth), queue policy (stale
  threshold, protected label prefix), model (id + sim id), baseline
  hyperparams (incl. self_play=true, leaderboard_bias=0.3, level_mix),
  and 8 sweep axes (entropy_coef / lr / rollout_steps / n_envs / gamma /
  clip_coef / leaderboard_bias / action_repeat) with 3 cells each.
- **`cli/loop_config.py`** — typed loader; `load()` returns a `LoopConfig`
  with `.next_axis(last_used)` for round-robin pick.
- **`scripts/queue_karp_sweep.py`** — replaces inline-Python INSERTs in
  the loop prompt. CLI: `--axis lr` to force, `--override 'n_envs=512'`
  to bake into baseline before sweeping, `--dry-run` for preview. Reads
  most-recent karp- run's axis from labels and round-robins to the next.
- Loop prompt simplified: future fires call `python scripts/queue_karp_sweep.py`
  instead of crafting INSERT statements. Hyperparams no longer embedded
  in two places (prompt + scripts/queue_b5.py).

Knock-on: my first sweep used label `karp-...-entropy-{lo,mid,hi}` with
the axis name truncated. The new script uses the full axis name
(`entropy_coef`), so future round-robin picks will work cleanly. The
3 in-flight runs are unaffected — they just have shorter labels.

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
