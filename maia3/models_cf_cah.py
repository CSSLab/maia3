"""Chessformer CF-CAH wrapper built directly on the existing CF-ACT model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chess
import torch

from .action_head import CFActPolicyHead
from .action_head_cah import CFCahPolicyHead
from .actions import collate_legal_actions, legal_moves_to_actions
from .models_cf_act import MAIA3ActModel


@dataclass
class CFCahModelDetails:
    logits_legal: torch.Tensor
    policy_probs: torch.Tensor
    logits_value: torch.Tensor
    logits_ponder: torch.Tensor
    m0: torch.Tensor
    contextual_actions: torch.Tensor
    attention_weights: torch.Tensor | None


class MAIA3CahModel(MAIA3ActModel):
    """Keep encoder, rating conditioning, WDL and ponder exactly as CF-ACT.

    Only the policy head is replaced. The inherited forward_with_actions()
    still returns (logits, probs, WDL, ponder, m0) for drop-in compatibility.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.policy_head = CFCahPolicyHead(
            d_model=cfg.dim_vit,
            mlp_dim=getattr(cfg, "act_mlp_dim", cfg.dim_vit),
            num_heads=getattr(cfg, "cah_num_heads", cfg.num_heads),
            mlp_ratio=getattr(cfg, "cah_mlp_ratio", 2.0),
            dropout=getattr(cfg, "cah_dropout", cfg.dropout),
        )

    def forward_with_details(
        self, tokens: torch.Tensor, self_elos: torch.Tensor,
        oppo_elos: torch.Tensor, src_sq: torch.Tensor,
        dst_sq: torch.Tensor, promo_type: torch.Tensor,
        action_mask: torch.Tensor, *, return_attention: bool = False,
    ) -> CFCahModelDetails:
        board, value, ponder = self.encode_board(tokens, self_elos, oppo_elos)
        details = self.policy_head(
            board, src_sq, dst_sq, promo_type, action_mask,
            return_details=True, return_attention=return_attention,
        )
        return CFCahModelDetails(
            details.logits_legal, details.policy_probs, value, ponder,
            details.m0, details.contextual_actions, details.attention_weights,
        )

    def forward(
        self, tokens: torch.Tensor, self_elos: torch.Tensor,
        oppo_elos: torch.Tensor, boards: Optional[List[chess.Board]] = None,
        return_4352_vocab: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CF-ACT-compatible inference, with strict 4352 index validation."""
        if boards is None:
            raise ValueError("MAIA3CahModel requires chess.Board objects or explicit actions")
        if len(boards) != tokens.shape[0]:
            raise ValueError("number of boards must match input batch size")
        board, value, ponder = self.encode_board(tokens, self_elos, oppo_elos)
        actions = [legal_moves_to_actions(b) for b in boards]
        collated = collate_legal_actions(actions, device=tokens.device)
        logits, _, _ = self.policy_head(
            board, collated["src_sq"], collated["dst_sq"],
            collated["promo_type"], collated["action_mask"],
        )
        if return_4352_vocab:
            for n in range(len(boards)):
                idx = collated["vocab_idx"][n][collated["action_mask"][n]]
                if (idx < 0).any() or (idx >= 4352).any() or idx.unique().numel() != idx.numel():
                    raise ValueError("legal moves must map bijectively into the 4352 vocabulary")
            logits = CFActPolicyHead.scatter_to_vocab_4352(
                logits, collated["vocab_idx"], collated["action_mask"]
            )
        return logits, value, ponder
