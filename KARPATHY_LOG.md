# Karpathy Loop — hyperparam sweep log

Format: one table per sweep. Baseline config: `v9.0-full`, `random_8_12`, self-play,
`n_envs=32`, `rollout_steps=128`, `snapshot_every=10`, `lr=3e-4`, `entropy_coef=0.01`,
budget 900s (15 min). Each run auto-admits vs top-5 + random-legal baseline.

## Sweep 1 — entropy_coef @ 15 min

| run | entropy_coef | vs baseline (20 games) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-ent-003 | 0.003 | 70% (14/20) | 2% |
| kar-ent-010 | 0.01 (baseline) | **90% (18/20)** | **8%** |
| kar-ent-030 | 0.03 | 85% (17/20) | 4% |

**Finding:** baseline `entropy_coef=0.01` wins on both axes. Too little
exploration (0.003) hurts the most; too much (0.03) is chaotic but not broken.
Caveat: 15 min is short; low entropy might catch up at longer budgets.

## Sweep 2 — lr @ 15 min

| run | lr | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-lr-1e4 | 1e-4 | 50% (10/20) | 4% |
| kar-lr-3e4 | 3e-4 (baseline) | 75% (15/20) | 8% |
| kar-lr-1e3 | 1e-3 | **85% (17/20)** | **10%** |

**Finding:** **higher lr (1e-3) wins** at 15-min budgets. Low lr (1e-4) barely
beats random — learning too slow for the budget. Seed noise is real: prior
kar-ent-010 at baseline config hit 90%, kar-lr-3e4 at same config hit 75%.

## Sweep 3 — rollout_steps @ 15 min

| run | rollout_steps | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-rs-64 | 64 | 85% (17/20) | 12% |
| kar-rs-128 | 128 (baseline) | 70% (14/20) | 12% |
| kar-rs-256 | 256 | **90% (18/20)** | 10% |

**Finding:** mixed / within noise. `rollout_steps` isn't a major knob at 15 min.
256 slightly better vs baseline; 64 ties on top-5. Baseline 128 unexpectedly
weakest vs baseline — likely seed variance (128 elsewhere hit 75-90%).

## Sweep 4 — snapshot_every @ 15 min

| run | snapshot_every | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-snap-5 | 5 (freshest) | 85% (17/20) | 4% |
| kar-snap-10 | 10 (baseline) | 55% (11/20) | 6% |
| kar-snap-20 | 20 (stalest) | **90% (18/20)** | **20%** |

**Finding:** **snap=20 wins decisively**, especially on vs-top-5 (20% — 3-5×
the other two). At short budgets, a stale self-play pool helps: fresh
snapshots of a half-trained agent add noise; infrequent updates give the
policy time to breathe. Would likely invert at long budgets where fresh
opponents matter more.

## Sweep 5 — entropy_coef @ 30 min (confirm)

Mac paused during training for clean CUDA-only comparison, re-enabled for match drain.

| run | entropy_coef | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-ent30-003 | 0.003 | **100% (20/20)** | 16% |
| kar-ent30-010 | 0.01 (baseline) | 80% (16/20) | 16% |
| kar-ent30-030 | 0.03 | 90% (18/20) | 16% |

**Finding:** **flip from 15-min result** — at 30 min, low entropy (0.003) wins
vs baseline. Confirms hypothesis: low exploration needs more time to exploit.
vs-top-5 coincidentally identical at 16% across all three (all faced same top-5
chain runs; per-opponent rates differ but sum to 0.8 in each case).
Overall skill has jumped vs 15-min runs (80–100% vs baseline, was 50–90%).

## Sweep 6 — level_name breadth @ 15 min

Admission evaluation is fixed at `random_8_12`, so this tests
train-breadth vs eval-narrow.

| run | level | vs baseline (20) | vs top-5 (10 each, avg) |
|---|---|---|---|
| kar-lvl-8-12 | random_8_12 (narrow) | **85% (17/20)** | 6% |
| kar-lvl-8-24 | random_8_24 (medium) | 70% (14/20) | **12%** |
| kar-lvl-8-32 | random_8_32 (wide) | 65% (13/20) | 4% |

**Finding:** vs-baseline rewards narrow training (specialist wins the
specialist test) but **vs-top-5 flips — medium (8-24) is the sweet spot**
(2-3× the others). Strong opponents expose the narrow specialist; wider
training spreads capacity too thin. Medium is the Goldilocks zone.
Suggests the main prod config (8-12) is too narrow for generalization.

## Sweep 7 — capacity @ varied budgets (in flight)

Testing Paul's hypothesis that trunk width is the ceiling. Sizes:
- `v9.0-full`  (BODY=128, ~170k params)      — 30 min budget
- `v9.0-256`   (BODY=256,  395k params)      — 30 min budget
- `v9.0-512`   (BODY=512,  915k params)      — 30 min budget
- `v9.0-1024`  (BODY=1024, 2.3M params)      — **3 hours** (long run)

Mac paused for clean CUDA-only training; unpaused for post-training match drain.
