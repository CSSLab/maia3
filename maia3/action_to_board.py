"""One proposal-faithful action-to-board attention block for CF-CAH.

Only action queries are normalized. Keys and values are projected directly
from the final Chessformer board tokens, as specified in the research proposal.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ActionToBoardCrossAttention(nn.Module):
    """Contextualize each legal move against exactly 64 contextual board squares.

    m0: (N, Lmax, d), board_tokens: (N, 64, d), action_mask: (N, Lmax).
    Returns C: (N, Lmax, d), and optionally maps (N, heads, Lmax, 64).
    Padded queries return zero features/maps. No action-to-action attention.
    """

    def __init__(
        self, d_model: int, num_heads: int = 8,
        mlp_ratio: float = 2.0, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if mlp_ratio <= 0 or not 0 <= dropout < 1:
            raise ValueError("mlp_ratio must be positive and dropout must be in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        hidden = int(d_model * mlp_ratio)
        if hidden <= 0:
            raise ValueError("d_model * mlp_ratio must be at least 1")
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(
        self, m0: torch.Tensor, board_tokens: torch.Tensor,
        action_mask: torch.Tensor, return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if m0.ndim != 3 or m0.shape[-1] != self.d_model:
            raise ValueError("m0 must be (N, Lmax, d_model)")
        n, lmax, d = m0.shape
        if board_tokens.shape != (n, 64, d):
            raise ValueError("board_tokens must be (N, 64, d_model)")
        if action_mask.shape != (n, lmax) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean (N, Lmax)")
        if m0.device != board_tokens.device or m0.device != action_mask.device:
            raise ValueError("all input tensors must use the same device")

        def split_heads(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(self.ln_q(m0)), lmax)
        k = split_heads(self.k_proj(board_tokens), 64)  # Intentionally NO board LN
        v = split_heads(self.v_proj(board_tokens), 64)  # Intentionally NO board LN
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)  # Normalize over 64 squares, not moves
        context = self.attn_dropout(attention) @ v
        context = context.transpose(1, 2).contiguous().reshape(n, lmax, d)
        u = m0 + self.out_proj(context)
        c = u + self.ffn(self.ln_ffn(u))
        c = c.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        if return_attention:
            # Report pre-dropout distributions; terminal/padded rows are zero.
            attention = attention.masked_fill(~action_mask[:, None, :, None], 0.0)
        else:
            attention = None
        return c, attention
