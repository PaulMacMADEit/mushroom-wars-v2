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
