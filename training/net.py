"""v12 actor-critic — set-transformer encoder + pointer-style heads.

Architecture (set-transformer over slot-tokens):

  Tokenizer:
    GLOBAL_FEATS → d_model       (one GLOBAL token, position 0)
    BUILDING_FEATS → d_model     (N_BUILDINGS tokens)
    GROUP_FEATS → d_model        (N_GROUPS tokens)
    + type embedding (GLOBAL=0, BUILDING=1, GROUP=2)
    + owner embedding (NEUTRAL=0, MINE=1, ENEMY=2; GLOBAL uses NEUTRAL)

  Encoder:
    n_layers × (multi-head self-attention with key-padding mask + FFN)
    pre-LN, GELU, FFN width = ffn_mult × d_model

  Heads (factored to match the agent's sampling chain):
    source_logits:  q = source_q_proj(GLOBAL); k = source_k_proj(buildings)
                    → (B, N_BUILDINGS) scaled dot-product
    cond_logits:    given chosen src building token,
                    type_logits  = type_head([GLOBAL; src_token])
                                   shape (B, NUM_TYPE_CHOICES) — 2 send pcts + noop
                    tgt_logits   = target_q_proj([GLOBAL; src]) · target_k_proj(buildings)
                                   → (B, N_BUILDINGS) scaled dot-product
    value:          value_head(GLOBAL) → scalar

The body returned by forward_body is a tuple (tokens, key_padding_mask)
where tokens is (B, 1+N_B+N_G, d_model). Heads slice in.

Why this matches the agent's existing chain (training/agent.py): the
external interface — forward_body / value / source_logits / cond_logits —
is byte-identical to v10's. Only the internals change. The chained
(src → type | src → tgt | src) sampling logic, masking decomposition, and
PPO update path stay untouched.

v10 → v12 differences:
  - body is no longer a flat (B, body_dim) tensor — it's a tuple. The
    agent treats it as opaque, so this is fine.
  - source_logits is now permutation-equivariant by construction (pointer
    head over building tokens, no positional bias).
  - target_head conditions on the literal source token (richer than v10's
    16-dim src embedding).
  - NUM_TYPES = 2 (was 4); NUM_TYPE_CHOICES = 3 (was 5).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from sim import config as C
from training.encoder import (
    BUILDING_FEATS,
    GLOBAL_FEATS,
    GROUP_FEATS,
    N_BUILDINGS,
    N_GROUPS,
    OBS_DIM,
)


NUM_TYPES         = len(C.SEND_PERCENTAGES)       # 2
NUM_TYPE_CHOICES  = NUM_TYPES + 1                 # 3 (+1 for noop)
NUM_SRC           = C.MAX_BUILDING_SLOTS          # 8
NUM_TGT           = C.MAX_BUILDING_SLOTS          # 8


# v12 architecture defaults — chosen to land at ~1M params on a 13-token set.
# See research notes 2026-05-01 (set-transformer + AlphaStar precedent).
D_MODEL  = 192
N_LAYERS = 2
N_HEADS  = 4
FFN_MULT = 4

# Token-stream layout — these positions are referenced by the heads' slicing
# code. Order is (GLOBAL, buildings…, groups…) so building indices line up
# with the agent's source/target slot indices (0..N_BUILDINGS-1).
TOKEN_GLOBAL_IDX  = 0
TOKEN_BLDG_START  = 1
TOKEN_BLDG_END    = 1 + N_BUILDINGS
TOKEN_GRP_START   = TOKEN_BLDG_END
TOKEN_GRP_END     = TOKEN_GRP_START + N_GROUPS
N_TOKENS          = TOKEN_GRP_END

# Type embedding ids (must match the order above).
TYPE_EMB_GLOBAL   = 0
TYPE_EMB_BUILDING = 1
TYPE_EMB_GROUP    = 2
NUM_TYPE_EMB      = 3

# Owner embedding ids (must match the encoder's OWNER_ID_* constants).
NUM_OWNER_EMB     = 3


def infer_d_model(state_dict: dict, default: int = D_MODEL) -> int:
    """Peek the model width from a saved ActorCritic state_dict.

    `global_proj.weight` has shape (d_model, GLOBAL_FEATS). Lets the worker
    reconstruct the right-sized net without storing size in the runs row.
    """
    w = state_dict.get("global_proj.weight")
    return int(w.shape[0]) if w is not None else default


def infer_body_dim(state_dict: dict, default: int = D_MODEL) -> int:
    """Back-compat alias for infer_d_model. Old loaders call body_dim."""
    return infer_d_model(state_dict, default=default)


def infer_obs_dim(state_dict: dict, default: int = OBS_DIM) -> int:
    """Peek the obs size from a saved ActorCritic state_dict.

    `global_proj.weight` has shape (d_model, GLOBAL_FEATS). The full obs is
    GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS + N_GROUPS * GROUP_FEATS,
    so we recompute from constants — encoder version controls the contract.
    """
    return default


class TransformerLayer(nn.Module):
    """One pre-LN self-attention layer + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        ffn_dim = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


class ActorCritic(nn.Module):
    """Set-transformer actor-critic. Forward is split so the agent can sample
    source → (type, tgt | source) in two passes that share the encoded body."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        ffn_mult: int = FFN_MULT,
        # Back-compat: old workers/configs pass `body_dim`.
        body_dim: int | None = None,
        head_hidden: int | None = None,  # unused under v12; kept for kwarg compat
    ):
        super().__init__()
        if body_dim is not None:
            d_model = body_dim
        self.d_model  = d_model
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.ffn_mult = ffn_mult

        # Tokenizers — one linear projection per token type.
        self.global_proj = nn.Linear(GLOBAL_FEATS,  d_model)
        self.bldg_proj   = nn.Linear(BUILDING_FEATS, d_model)
        self.grp_proj    = nn.Linear(GROUP_FEATS,   d_model)

        # Type-of-token embedding (GLOBAL / BUILDING / GROUP).
        self.type_emb  = nn.Embedding(NUM_TYPE_EMB,  d_model)
        # Owner embedding (NEUTRAL / MINE / ENEMY). GLOBAL token uses NEUTRAL.
        self.owner_emb = nn.Embedding(NUM_OWNER_EMB, d_model)

        # Self-attention encoder.
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, ffn_mult)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Pointer head: source.
        self.source_q_proj = nn.Linear(d_model, d_model)
        self.source_k_proj = nn.Linear(d_model, d_model)

        # Pointer head: target. Query conditioned on [GLOBAL; src_token].
        self.target_q_proj = nn.Linear(d_model * 2, d_model)
        self.target_k_proj = nn.Linear(d_model, d_model)

        # Type head: 3-way classification from [GLOBAL; src_token].
        self.type_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, NUM_TYPE_CHOICES),
        )

        # Value head: scalar from GLOBAL.
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Body + tokenizers + intermediate heads at standard PPO gain.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2.0 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy-head outputs at small gain (keeps early entropy high).
        for last_layer in (self.source_q_proj, self.source_k_proj,
                           self.target_q_proj, self.target_k_proj):
            nn.init.orthogonal_(last_layer.weight, gain=0.1)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
        type_last = self.type_head[-1]
        nn.init.orthogonal_(type_last.weight, gain=0.01)
        nn.init.zeros_(type_last.bias)
        # Value head at gain 1 so it predicts near-zero initially.
        value_last = self.value_head[-1]
        nn.init.orthogonal_(value_last.weight, gain=1.0)
        nn.init.zeros_(value_last.bias)
        # Embeddings small uniform (so they barely perturb the per-token signal
        # at init; the network learns useful values from gradients).
        nn.init.uniform_(self.type_emb.weight,  -0.1, 0.1)
        nn.init.uniform_(self.owner_emb.weight, -0.1, 0.1)

    # ------------------------------------------------------------------
    # Token decode + encoder
    # ------------------------------------------------------------------

    def _decode_obs_to_tokens(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice flat obs into per-token feature blocks, project, return
        (tokens (B, N_TOKENS, d), key_padding_mask (B, N_TOKENS) bool)."""
        B = obs.shape[0]
        device = obs.device

        globals_flat = obs[:, :GLOBAL_FEATS]                                  # (B, GF)
        bldg_offset_end = GLOBAL_FEATS + N_BUILDINGS * BUILDING_FEATS
        bldg_flat = obs[:, GLOBAL_FEATS:bldg_offset_end].reshape(B, N_BUILDINGS, BUILDING_FEATS)
        grp_flat  = obs[:, bldg_offset_end:].reshape(B, N_GROUPS, GROUP_FEATS)

        # Alive flags from the encoder's known column positions.
        bldg_alive = bldg_flat[..., 0]  # BLDG_FEAT_ALIVE = 0
        grp_alive  = grp_flat[..., 0]   # GRP_FEAT_ALIVE = 0

        # Owner ids (long for embedding lookup). Clamp guards against weird
        # values during obs-norm corner cases (RunningNorm shouldn't move
        # categorical columns much, but defence in depth).
        bldg_owner_id = bldg_flat[..., 1].long().clamp(0, NUM_OWNER_EMB - 1)
        grp_owner_id  = grp_flat[..., 1].long().clamp(0, NUM_OWNER_EMB - 1)

        # Type embedding ids — broadcast scalar lookups so we can add to the
        # whole token block at once.
        type_global   = self.type_emb(torch.tensor(TYPE_EMB_GLOBAL,   device=device, dtype=torch.long))
        type_building = self.type_emb(torch.tensor(TYPE_EMB_BUILDING, device=device, dtype=torch.long))
        type_group    = self.type_emb(torch.tensor(TYPE_EMB_GROUP,    device=device, dtype=torch.long))

        # Project + add positional/type/owner signals.
        global_tok = self.global_proj(globals_flat) + type_global       # (B, d)
        bldg_tok   = self.bldg_proj(bldg_flat) + type_building          # (B, N_B, d)
        bldg_tok   = bldg_tok + self.owner_emb(bldg_owner_id)
        grp_tok    = self.grp_proj(grp_flat) + type_group               # (B, N_G, d)
        grp_tok    = grp_tok + self.owner_emb(grp_owner_id)

        tokens = torch.cat([global_tok.unsqueeze(1), bldg_tok, grp_tok], dim=1)  # (B, T, d)

        # key_padding_mask: True where the position should be IGNORED.
        global_alive = torch.ones(B, 1, device=device, dtype=bldg_alive.dtype)
        alive = torch.cat([global_alive, bldg_alive, grp_alive], dim=1)
        key_padding_mask = alive < 0.5

        return tokens, key_padding_mask

    def forward_body(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, key_padding_mask = self._decode_obs_to_tokens(obs)
        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask)
        tokens = self.final_ln(tokens)
        return (tokens, key_padding_mask)

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    def value(self, body) -> torch.Tensor:
        tokens, _ = body
        global_token = tokens[:, TOKEN_GLOBAL_IDX, :]
        return self.value_head(global_token).squeeze(-1)

    def source_logits(self, body) -> torch.Tensor:
        tokens, _ = body
        global_token = tokens[:, TOKEN_GLOBAL_IDX, :]
        bldg_tokens  = tokens[:, TOKEN_BLDG_START:TOKEN_BLDG_END, :]

        q = self.source_q_proj(global_token)              # (B, d)
        k = self.source_k_proj(bldg_tokens)               # (B, N_B, d)
        scale = 1.0 / math.sqrt(self.d_model)
        return torch.einsum("bd,bnd->bn", q, k) * scale   # (B, N_B)

    def cond_logits(
        self,
        body,
        src_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, _ = body
        global_token = tokens[:, TOKEN_GLOBAL_IDX, :]
        bldg_tokens  = tokens[:, TOKEN_BLDG_START:TOKEN_BLDG_END, :]

        B = tokens.shape[0]
        batch_idx = torch.arange(B, device=tokens.device)
        src_token = bldg_tokens[batch_idx, src_idx]       # (B, d)
        cat = torch.cat([global_token, src_token], dim=-1)  # (B, 2d)

        type_logits = self.type_head(cat)                 # (B, NUM_TYPE_CHOICES)

        q = self.target_q_proj(cat)                       # (B, d)
        k = self.target_k_proj(bldg_tokens)               # (B, N_B, d)
        scale = 1.0 / math.sqrt(self.d_model)
        tgt_logits = torch.einsum("bd,bnd->bn", q, k) * scale  # (B, N_B)

        return type_logits, tgt_logits


__all__ = [
    "ActorCritic",
    "TransformerLayer",
    "NUM_TYPES",
    "NUM_TYPE_CHOICES",
    "NUM_SRC",
    "NUM_TGT",
    "D_MODEL",
    "N_LAYERS",
    "N_HEADS",
    "FFN_MULT",
    "infer_d_model",
    "infer_body_dim",
    "infer_obs_dim",
]
