# v13 plan — chain reorder + head capacity (bundled)

Last updated: 2026-05-03

## Goal

Two concurrent changes to the actor-critic policy net, shipped as one major bump (v12 → v13):

1. **Chain reorder.** Sampling order changes from `src → type(incl. noop) → tgt` to **`src → tgt → pct(incl. noop)`**. This lets the pct head condition on both src and tgt — directly addressing the cap-overflow pathology where the policy commits to "100%" before knowing whether the destination is at capacity.
2. **Head capacity bump on 7A and 7C.** Wrap the source and target heads' projections in 2-layer MLPs (`d → 2d → d` with GELU). +16.3% forward-pass FLOPs, +~800k params, concentrated where action decisions are made.

Encoder is **unchanged**: same OBS_DIM=192, same observation shape, same env action space (129).

## Bundling note (one-variable-at-a-time waiver)

Per [.claude/rules/training-discipline.md](.claude/rules/training-discipline.md) §1, ideally these would ship as separate bumps. Bundled at user request because:
- The chain reorder is the structural fix for the observed cap-overflow pathology.
- The head capacity bump is a well-precedented capacity adjustment with low downside risk.
- Both ship in one major bump anyway since the state_dict keys change either way.

If results are ambiguous, fall back to two follow-up A/Bs:
- v13a (no head MLPs, just chain reorder)
- v13b (no chain reorder, just head MLPs)
trained with identical seeds, to disentangle which lever moved the needle.

## Backward compatibility — v13 must be able to play v12

The env's action space is **unchanged**. Both nets emit into the same flat action index 0..128 (= 2 types × 8 src × 8 tgt + 1 noop). v12's `(type, src, tgt) → flat` packing is preserved as the env's contract. v13's internal sampling chain maps `(src, tgt, pct)` to the SAME flat packing for env compatibility:

```
v13 pct ∈ {noop, 50%, 100%}
  pct=noop → action = NOOP_INDEX
  pct=50%  → action = encode(type_idx=0, src, tgt)
  pct=100% → action = encode(type_idx=1, src, tgt)
```

So the env doesn't know v12 from v13 — both produce valid flat actions and read the same `action_mask`. Cross-version play "just works" once the **net dispatch** is wired in (loader instantiates the right ActorCritic class per checkpoint).

### Backward-compat checklist

- ✅ Same encoder (v12 OBS_DIM=192, no obs feature changes)
- ✅ Same env action space (flat 129)
- ✅ Same action mask shape and semantics
- ✅ Net registry instantiates the right class per `net_version` stamp
- ✅ Legacy unstamped checkpoints default to `"v12"`
- ✅ Tournament and match_runner load each side independently

## Architectural changes

### v13 head structure (replaces current `training/net.py`)

```python
# 7A Source — wrapped in MLP (NEW)
source_q = MLP(global, d → 2d → d) → q_proj(d → d)
source_k = MLP(buildings, d → 2d → d) → k_proj(d → d)   # × 8 buildings
source_logits = q · k

# 7C Target — wrapped in MLP, conditions on src ONLY (CHAIN REORDERED)
target_q = MLP([global; src], 2d → 2d → d) → q_proj(d → d)
target_k = MLP(buildings, d → 2d → d) → k_proj(d → d)   # × 8 buildings
target_logits = q · k

# 7B Pct (renamed from "type") — conditions on src AND tgt (CHAIN REORDERED + RENAMED)
pct_logits: [global ; src_token ; tgt_token] (3d) → MLP(3d → d → 3) → 3 logits
            # 3 outcomes: {noop=0, 50%=1, 100%=2}

# 7D Value — UNCHANGED
value_head: linear(d → d) → GELU → linear(d → 1)
```

### Sampling chain (`training/agent.py`)

```python
src ~ Categorical(source_logits | mask_src)
tgt ~ Categorical(target_logits | mask_tgt(src))
pct ~ Categorical(pct_logits    | mask_pct(src, tgt))
flat_action = pct_to_flat(src, tgt, pct)   # see backward-compat section
```

### Param count and FLOPs

| Stage | v12 FLOPs | v13 FLOPs | Δ |
|---|---|---|---|
| Encoder (2 transformer layers) | 23.2 M | 23.2 M | — |
| Tokenizer + LN | 0.08 M | 0.08 M | — |
| 7A Source | 0.67 M | 2.66 M | +1.99 M |
| 7B Pct (replaces Type) | 0.15 M | 0.27 M | +0.12 M |
| 7C Target | 0.74 M | 2.81 M | +2.07 M |
| 7D Value | 0.07 M | 0.07 M | — |
| **Total** | **24.9 M** | **29.0 M** | **+16.6%** |

Params: ~1.2M → ~2.0M (mostly in the source/target MLP wrappers).

## File-by-file changes

| # | File | Change | Approx LOC |
|---|---|---|---|
| 1 | `training/nets/__init__.py` (new) | Net registry: `NET_BUILDERS = {"v12": ..., "v13": ...}`, `get_net_class(version)`, `CURRENT_NET_VERSION = "v13"`, `DEFAULT_NET_VERSION = "v12"`. Mirrors `training/encoders/__init__.py`. | ~80 |
| 2 | `training/nets/v12.py` (new) | Verbatim frozen copy of current `training/net.py`. Self-contained. | ~350 |
| 3 | `training/net.py` (rewrite) | New v13 architecture: chain reorder + head MLP wrappers. Public surface (`forward_body`, `value`, `source_logits`, **renamed** `target_logits`, `pct_logits`) updated. | ~400 |
| 4 | `training/agent.py` (edit) | New sampling chain `src→tgt→pct`. `_decompose_masks` rewired. New `_compose_action(src, tgt, pct)` that maps to v12-compatible flat action. **For v12 ckpts** the agent must still use the old chain — solution: dispatch on `net_version` in `PPOAgent.__init__` and pick the right `act_batch` impl. | ~120 |
| 5 | `training/checkpoint.py` (edit) | Add `net_version` to wrapper. `save_state_dict(..., net_version=None)` defaults to `CURRENT_NET_VERSION`. `load_state_dict_with_version()` returns `(state_dict, encoder_version, net_version)`. | ~20 |
| 6 | `workers/match_runner.py` (edit) | `_load_agent` reads `net_version` from wrapper, calls `get_net_class(net_version)` to instantiate the right ActorCritic. | ~10 |
| 7 | `scripts/tournament.py` (edit) | Same dispatch as match_runner. | ~5 |
| 8 | `tests/test_v13.py` (new) | Shape tests, cross-version load, v12-vs-v13 plays one full game, FLOP ratio assertion ∈ [1.14, 1.18]. | ~120 |
| 9 | `training/trainer.py` (edit) | `--net-version v13` flag (default v13 for fresh runs). Warm-start from v12: copy encoder + tokenizer + body weights into v13 net; reinit heads from scratch (orthogonal, low gain). | ~40 |

**Total: ~1100 lines across 9 files. About 1/3 is the v12 archive copy.**

## Backward-compat is the trickiest part — design detail

The PPO agent's sampling chain is structurally different between v12 and v13. Solution:

**`PPOAgent` becomes thin and dispatches** to a per-version `_sample` impl:

```python
class PPOAgent:
    def __init__(self, net, device, net_version):
        self.net = net
        self._sample_impl = {
            "v12": _sample_v12,  # src → type → tgt
            "v13": _sample_v13,  # src → tgt → pct
        }[net_version]
    def act_batch(self, ...): return self._sample_impl(self.net, ...)
```

The two `_sample_*` functions live in `training/agents/v12.py` and `training/agents/v13.py` (new dir mirroring the nets/ pattern). They share `_decompose_masks` and `_compose_action` helpers, version-specific.

**Why two impls instead of one parameterised function:** the chain order, conditioning, and action-packing differ enough that a single parameterised function would have N branches and become unreadable. Two ~80-line impls are cleaner. Long-term we can factor common helpers.

## Test plan

| Test | What it asserts |
|---|---|
| `test_v13_forward_shape` | v13 net forward returns expected token tensor shape |
| `test_v13_cross_version_load` | save fresh v12 ckpt, load it via new loader, verify `net_version=="v12"` and net is v12 ActorCritic |
| `test_v13_cross_version_play` | take live b6 v12 champion + freshly-init v13 net, play 1 full game on `crossroads_8`. No crashes, valid winner. |
| `test_v13_flop_ratio` | profile or count FLOPs of v12 vs v13 forward pass; assert ratio ∈ [1.14, 1.18] |
| `test_v13_action_packing_compat` | for every (src, tgt, pct) triple, verify v13's flat action index matches v12's `encode(type_idx, src, tgt)` for the equivalent semantic action |
| `test_v13_mask_decomp` | for a synthetic obs, v13's `_decompose_masks` produces masks that, when applied stage-by-stage, only allow legal (src, tgt, pct) triples |

## Roll-out plan

1. **Code lands and tests pass.** No training started.
2. **First v13 run uses identical hyperparams to current v12 b6 champion** (one variable at a time at the *training* level, even though architecture has two coupled changes). 50M training steps.
3. **Bench vs b6** at 50M steps. Decision rule:
   - v13 ≥ 55% → v13 becomes new champion.
   - 45–55% → spawn v13a/v13b ablations to disentangle which lever helped.
   - <45% → revert; the bundled change isn't a net win. Investigate before another bump.
4. **Cross-play validation:** v13-vs-v12 head-to-head on full level distribution as sanity that v13 isn't just overfit.

## Rollback plan

- v12 ActorCritic class archived at `training/nets/v12.py` — v12 ckpts loadable forever.
- All existing v12 stamped checkpoints stay playable.
- If v13 underperforms, set `CURRENT_NET_VERSION = "v12"` and continue v12 training. v13 archive stays in `training/nets/v13.py` for future revisit.

## Open questions to confirm before implementation

(All resolved in conversation 2026-05-03)

1. ✅ Bundle chain reorder + head capacity? Yes.
2. ✅ Head capacity scope = 7A + 7C only (not 7B/7D)? Yes.
3. ✅ pct stays 3-way `{noop, 50%, 100%}`? Yes (no menu change).
4. ✅ Default fresh runs to v13? Yes.
5. ✅ Backward compat: v13 plays v12 in same env? Yes — implemented via shared flat action space and net registry dispatch.

## v14 follow-up (not in scope)

If v13 lands, candidate v14 changes (separate bumps):
- Pre-action noop gate (AlphaStar-style "act-or-skip" Bernoulli before src is sampled)
- Pct menu expansion ({25%, 50%, 75%, 100%})
- Encoder feature: per-building "fraction of capacity" for the cap-overflow signal
