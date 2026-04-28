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

## Code changes during loop

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
