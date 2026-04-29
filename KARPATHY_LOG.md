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

## Code changes during loop

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
