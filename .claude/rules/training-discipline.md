---
paths:
  - "training/**/*.py"
  - "workers/**/*.py"
  - "cli/**/*.py"
  - "configs/**/*.py"
  - "configs/**/*.yaml"
  - "scripts/cron_agent_pulse.py"
  - "JAX_PORT_PLAN.md"
  - "CURRICULUM_PLAN.md"
  - "FUSED_ROLLOUT_PLAN.md"
  - "PHASE_G_PLAN.md"
  - "KARPATHY_LOG.md"
---

# Mushroom Wars v2 — training discipline (path-scoped)

## One variable at a time

When benchmarking or A/B testing across infra changes (sim version, training framework, level distribution, model architecture), **change one variable at a time**. Multi-variable swings produce uninterpretable results.

- When sim/framework version bumps, the *first* run under new infra must use the **exact** baseline hyperparams of the prior infra. Establish the new reference point before changing anything else.
- Train and eval (admission) must use the **same** sim and **same** level distribution. Cross-distribution evaluation is meaningless.
- When proposing an A/B test, list the variables that differ. If >1, propose splitting into separate runs.

Origin: 2026-04-24 `kar-sim11-combo-a` had 3 simultaneous changes (sim v1.0 → v1.1, new hyperparams, new training level) and produced an uninterpretable 7.7% (vs 43% baseline). Wasted ~25 min of compute and a Karpathy round.

Codified in JAX_PORT_PLAN.md §13.3 and §13.5.

## Major version bump on I/O / topology changes

Any change that **invalidates existing checkpoints** is a major version bump (v5.x → v6.0, v6.x → v7.0). Not minor.

Major-bump triggers:
- New observation shape (encoder input)
- New action shape (decoder output)
- New network topology (head structure, body depth/width, head subnet split)
- New action-space semantics (decoder behaviour change even if shape stays)

Minor bumps (v6.0 → v6.1) are reserved for:
- Different sampling order
- Auxiliary losses that don't change shape
- Hyperparameter tuning
- Different decoder semantics within the same I/O contract

Procedure when planning a change:
1. Does this change the serialized model shape? Yes → major bump.
2. Does it change action-space semantics (decoder)? Yes → major.
3. Does it change observation features (encoder shape)? Yes → major.
4. Update `CURRENT_VERSION` in `ModelVersion.ts` (or equivalent).
5. Register the new version in the VERSIONS table.
6. Old checkpoints stay playable at their original version.
7. Commit message names the version change ("v6.0: multi-head ...").

Violating this causes silent checkpoint mismatches — old weights don't fit new shape, runs crash at load time or load with scrambled outputs.

## Don't switch architectures preemptively

When an RL agent under-performs or fails to generalise, the **default first suspect is curriculum / reward signal / feature engineering, NOT architecture**.

Cheap-to-expensive lever order:
1. **Reward shaping** — change what the agent gets credit for.
2. **Curriculum / opponent design** — change what distribution the agent sees.
3. **Encoder feature engineering** — surface invariants the agent isn't picking up.
4. **Architecture** — last resort. ~10× more compute and engineer time than curriculum.

Three good diagnostics before architecture:
- **Compute scaling**: train 3× longer; if still improving, you're not architecture-bound.
- **Permutation-invariance test**: shuffle slot indices in eval; if win rate drops <5pp, the network already learned approximate equivariance.
- **Transfer matrix**: eval across the full level distribution; flat = architecture fine; cliff = curriculum problem.

**Counter-rule:** if learning curves plateau under multiple curriculum/reward variations, that's the signal architecture is the bottleneck.

Origin: 2026-04-27. Suspected the slot-anonymous MLP encoder couldn't generalise across building counts (close-only champion got 5% on `random_16_24`). Almost prescribed v10 set-equivariant attention. Instead, changed curriculum (full-map mix instead of close-only) and the same MLP went from 5% to 100%. Architecture was never the bottleneck.

## Pointers

- Source memories: [feedback_one_variable_at_a_time.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/feedback_one_variable_at_a_time.md), [feedback_mushroom_wars_version_bump.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/feedback_mushroom_wars_version_bump.md), [feedback_dont_change_arch_preemptively.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/feedback_dont_change_arch_preemptively.md)
- Champion archive + bench_eval system: [project_mushroom_wars_bench_eval.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/project_mushroom_wars_bench_eval.md)
- Current champion (b6 phase1_full_mix): [project_mushroom_wars_b6.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/project_mushroom_wars_b6.md)
- Disproven approaches (don't revive): [project_mushroom_wars_phase1_close_invalid.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/project_mushroom_wars_phase1_close_invalid.md), [project_mushroom_wars_b3.md](/Users/paul/Documents/Projects/AI/ClaudeMemories/memory/project_mushroom_wars_b3.md)
