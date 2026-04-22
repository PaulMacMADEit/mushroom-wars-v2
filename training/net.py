"""v9.0 actor-critic with chained source → type → target heads.

ARCHITECTURE §9.4. The 4097-way action space is the product of (source slot,
send percentage or noop, target slot); a flat 4097-way head has to learn
every triple independently. Factoring the decision into three conditional
heads drops the policy head from ~524k params → ~32k and lets the body
learn "slot K is strong" info once instead of repeating it across 32 × 4
target×type combinations.

  body           OBS_DIM → 128 → 128           shared features
  source_head    128 → 64 → 32                 "which of my slots sends?"
  type_head     (128+16) → 64 → 5              "how much? (25/50/75/100/noop)"
  target_head   (128+16) → 64 → 32             "where to?"
  value_head     128 → 64 → 1
  src_embed     Embedding(32, 16)              condition type/tgt on source

Sampling is done in the agent: src → (type | src) → (tgt | src). When the
sampled type is the noop slot (index = NUM_TYPES = 4), the flat action
returned to the env is NOOP_INDEX regardless of src/tgt.
"""

from __future__ import annotations

import torch
from torch import nn

from sim import config as C
from training.encoder import OBS_DIM


NUM_TYPES         = len(C.SEND_PERCENTAGES)       # 4 send percentages
NUM_TYPE_CHOICES  = NUM_TYPES + 1                 # +1 for noop (type=4)
NUM_SRC           = C.MAX_BUILDING_SLOTS          # 32
NUM_TGT           = C.MAX_BUILDING_SLOTS          # 32

BODY_DIM     = 128
HEAD_HIDDEN  = 64
EMBED_DIM    = 16


class ActorCritic(nn.Module):
    """Chained-head actor-critic. Forward is done in pieces so the agent can
    sample source → (type, tgt | source) in two passes sharing the body."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, BODY_DIM), nn.ReLU(),
            nn.Linear(BODY_DIM, BODY_DIM), nn.ReLU(),
        )
        self.source_head = nn.Sequential(
            nn.Linear(BODY_DIM, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, NUM_SRC),
        )
        self.type_head = nn.Sequential(
            nn.Linear(BODY_DIM + EMBED_DIM, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, NUM_TYPE_CHOICES),
        )
        self.target_head = nn.Sequential(
            nn.Linear(BODY_DIM + EMBED_DIM, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, NUM_TGT),
        )
        self.value_head = nn.Sequential(
            nn.Linear(BODY_DIM, HEAD_HIDDEN), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN, 1),
        )
        self.src_embed = nn.Embedding(NUM_SRC, EMBED_DIM)

        # Init: body + heads orthogonal, small-gain on the policy logits
        # (standard PPO trick — keeps early entropy high). Value head at
        # gain=1 so it predicts near-zero initially.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2.0 ** 0.5)
                nn.init.zeros_(m.bias)
        for head in (self.source_head, self.type_head, self.target_head):
            last = head[-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.uniform_(self.src_embed.weight, -0.1, 0.1)

    # ------------------------------------------------------------------
    # Forward pieces — called separately by the agent
    # ------------------------------------------------------------------

    def forward_body(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def value(self, body: torch.Tensor) -> torch.Tensor:
        return self.value_head(body).squeeze(-1)

    def source_logits(self, body: torch.Tensor) -> torch.Tensor:
        return self.source_head(body)

    def cond_logits(
        self,
        body: torch.Tensor,
        src_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given trunk output and a chosen source, return (type_logits, target_logits)."""
        src_emb = self.src_embed(src_idx)
        cond = torch.cat([body, src_emb], dim=-1)
        return self.type_head(cond), self.target_head(cond)
