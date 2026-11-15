from __future__ import annotations

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionTokenConstructor(nn.Module):
    r"""Constructs explicit legal-action tokens from Chessformer square representations.

    Implements Equation (6) of the proposal:
        m_i^(0) = W_s * b_{u_i} + W_d * b_{v_i} + e_{p_i} + b_m \in R^d
    """

    def __init__(
        self,
        d_model: int,
        out_dim: Optional[int] = None,
        num_promotions: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim or d_model

        # W_s and W_d: independent linear projections for source and destination squares
        self.proj_src = nn.Linear(d_model, self.out_dim, bias=False)
        self.proj_dst = nn.Linear(d_model, self.out_dim, bias=False)

        # e_p: learned embedding for promotion type (0: None, 1: Queen, 2: Rook, 3: Bishop, 4: Knight)
        self.promo_embedding = nn.Embedding(num_promotions, self.out_dim)

        # b_m: learned shared action bias
        self.action_bias = nn.Parameter(torch.zeros(self.out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_src.weight)
        nn.init.xavier_uniform_(self.proj_dst.weight)
        nn.init.normal_(self.promo_embedding.weight, std=0.02)
        nn.init.zeros_(self.action_bias)

    def forward(
        self,
        board_embeddings: torch.Tensor,
        src_sq: torch.Tensor,
        dst_sq: torch.Tensor,
        promo_type: torch.Tensor,
    ) -> torch.Tensor:
        """Construct initial action tokens M^(0).

        Args:
            board_embeddings: Final square tokens B of shape (B, 64, d_model).
            src_sq: Source square indices of shape (B, L_max) in [0, 63].
            dst_sq: Destination square indices of shape (B, L_max) in [0, 63].
            promo_type: Promotion category indices of shape (B, L_max) in [0, 4].

        Returns:
            Action tokens M^(0) of shape (B, L_max, out_dim).
        """
        B_size, _, d = board_embeddings.shape
        L_max = src_sq.shape[1]

        # Expand indices for torch.gather across square dimension (dim=1)
        src_index = src_sq.unsqueeze(-1).expand(-1, -1, d)  # (B, L_max, d)
        dst_index = dst_sq.unsqueeze(-1).expand(-1, -1, d)  # (B, L_max, d)

        b_u = torch.gather(board_embeddings, dim=1, index=src_index)  # (B, L_max, d)
        b_v = torch.gather(board_embeddings, dim=1, index=dst_index)  # (B, L_max, d)

        # Additive combination with learned projections and embeddings
        proj_u = self.proj_src(b_u)                          # (B, L_max, out_dim)
        proj_v = self.proj_dst(b_v)                          # (B, L_max, out_dim)
        e_p = self.promo_embedding(promo_type)               # (B, L_max, out_dim)

        m_0 = proj_u + proj_v + e_p + self.action_bias       # (B, L_max, out_dim)
        return m_0


class CFActPolicyHead(nn.Module):
    r"""CF-ACT Policy Head: Minimal explicit action representation scored directly via shared MLP.

    Implements Equations (8) & (9) of the proposal:
        l_i^(ACT) = w_2^T GELU(W_1 LN(m_i^(0)) + b_1) + b_2
        pi_ACT(a_i | s) = exp(l_i^(ACT)) / sum_j exp(l_j^(ACT))
    """

    def __init__(
        self,
        d_model: int,
        mlp_dim: Optional[int] = None,
        num_promotions: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_dim = mlp_dim or d_model

        self.constructor = ActionTokenConstructor(
            d_model=d_model,
            out_dim=d_model,
            num_promotions=num_promotions,
        )

        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, self.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.mlp_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.constructor.reset_parameters()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        board_embeddings: torch.Tensor,
        src_sq: torch.Tensor,
        dst_sq: torch.Tensor,
        promo_type: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute legal action logits and probabilities.

        Args:
            board_embeddings: Final square tokens B of shape (B, 64, d_model).
            src_sq: (B, L_max)
            dst_sq: (B, L_max)
            promo_type: (B, L_max)
            action_mask: (B, L_max) bool, True for legal actions, False for padding.

        Returns:
            logits_legal: (B, L_max) masked logits with -inf on invalid/padded slots.
            policy_probs: (B, L_max) normalized probability distribution over legal actions.
            m_0: (B, L_max, d_model) raw action tokens.
        """
        # 1. Action-token construction
        m_0 = self.constructor(board_embeddings, src_sq, dst_sq, promo_type)

        # 2. Shared MLP scoring
        h = self.ln(m_0)
        h = self.act(self.fc1(h))
        logits = self.fc2(h).squeeze(-1)  # (B, L_max)

        # 3. Mask invalid / padding slots with -inf
        masked_logits = logits.masked_fill(~action_mask, float("-inf"))

        # 4. Numerically stable softmax over legal actions
        # Handle terminal positions (where action_mask is all False for a batch item)
        has_legal = action_mask.any(dim=-1, keepdim=True)  # (B, 1)

        # For positions with at least one legal move, compute standard softmax
        # For positions with 0 legal moves, set output probabilities to 0.0 without NaN
        safe_logits = torch.where(has_legal, masked_logits, torch.zeros_like(masked_logits))
        policy_probs = F.softmax(safe_logits, dim=-1)
        policy_probs = torch.where(has_legal, policy_probs, torch.zeros_like(policy_probs))

        return masked_logits, policy_probs, m_0

    @staticmethod
    def scatter_to_vocab_4352(
        logits_legal: torch.Tensor,
        vocab_indices: torch.Tensor,
        action_mask: torch.Tensor,
        vocab_size: int = 4352,
    ) -> torch.Tensor:
        """Scatter legal action logits into the fixed 4352-vocabulary tensor for engine / UCI compatibility.

        Args:
            logits_legal: (B, L_max) logits for legal actions.
            vocab_indices: (B, L_max) vocabulary indices in [0, 4351] (-1 only for invalid/padded).
            action_mask: (B, L_max) bool mask.
            vocab_size: Total vocabulary size (default 4352).

        Returns:
            Tensor of shape (B, vocab_size) with logits for legal moves and -inf elsewhere.
        """
        if logits_legal.ndim != 2:
            raise ValueError("logits_legal must have shape (B, L_max)")
        if action_mask.shape != logits_legal.shape or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean with shape (B, L_max)")
        if vocab_indices.shape != logits_legal.shape or vocab_indices.dtype != torch.long:
            raise ValueError("vocab_indices must be int64 with shape (B, L_max)")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if any(x.device != logits_legal.device for x in (action_mask, vocab_indices)):
            raise ValueError("logits, mask and vocabulary indices must share a device")

        # The original 4352-vocabulary is a bijection on legal moves.
        # Never silently discard a legal action or overwrite another action.
        for row in range(logits_legal.shape[0]):
            valid_indices = vocab_indices[row, action_mask[row]]
            if ((valid_indices < 0) | (valid_indices >= vocab_size)).any().item():
                raise ValueError(f"unmapped or out-of-range legal move in batch row {row}")
            if valid_indices.unique().numel() != valid_indices.numel():
                raise ValueError(f"duplicate legal vocabulary index in batch row {row}")

        B_size = logits_legal.shape[0]
        device = logits_legal.device
        full_logits = torch.full((B_size, vocab_size), float("-inf"), device=device, dtype=logits_legal.dtype)

        for b in range(B_size):
            valid = action_mask[b]
            if valid.any():
                v_idx = vocab_indices[b][valid]
                l_val = logits_legal[b][valid]
                full_logits[b, v_idx] = l_val

        return full_logits
