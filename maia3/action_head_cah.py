"""CF-CAH head: the exact CF-ACT constructor/scorer with one added context block."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .action_head import CFActPolicyHead
from .action_to_board import ActionToBoardCrossAttention


@dataclass
class CFCahHeadDetails:
    logits_legal: torch.Tensor
    policy_probs: torch.Tensor
    m0: torch.Tensor
    contextual_actions: torch.Tensor
    attention_weights: torch.Tensor | None


class CFCahPolicyHead(CFActPolicyHead):
    """Drop-in CF-ACT head replacement (default return stays a three-tuple).

    Set return_details=True for a named diagnostic object. The new attention
    block is the only added architecture; all existing scorer layers are reused.
    """

    def __init__(
        self, d_model: int, mlp_dim: int | None = None,
        num_promotions: int = 5, num_heads: int = 8,
        mlp_ratio: float = 2.0, dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, mlp_dim=mlp_dim, num_promotions=num_promotions)
        self.cross_attention = ActionToBoardCrossAttention(
            d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )

    def forward(
        self, board_embeddings: torch.Tensor,
        src_sq: torch.Tensor, dst_sq: torch.Tensor,
        promo_type: torch.Tensor, action_mask: torch.Tensor,
        *, return_details: bool = False, return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | CFCahHeadDetails:
        m0 = self.constructor(board_embeddings, src_sq, dst_sq, promo_type)
        c, weights = self.cross_attention(
            m0, board_embeddings, action_mask,
            return_attention=return_attention and return_details,
        )
        logits = self.fc2(self.act(self.fc1(self.ln(c)))).squeeze(-1)
        masked_logits = logits.masked_fill(~action_mask, float("-inf"))
        has_legal = action_mask.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_legal, masked_logits, torch.zeros_like(masked_logits))
        probs = F.softmax(safe_logits, dim=-1)
        probs = torch.where(has_legal, probs, torch.zeros_like(probs))
        if return_details:
            return CFCahHeadDetails(masked_logits, probs, m0, c, weights)
        return masked_logits, probs, m0
