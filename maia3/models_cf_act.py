from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    AbsolutePE,
    CustomTransformerEncoder,
)
from .action_head import CFActPolicyHead
from .actions import legal_moves_to_actions, collate_legal_actions


class MAIA3ActModel(nn.Module):
    """Maia-3 / Chessformer model with the CF-ACT explicit action-token policy head.

    Retains the exact Chessformer encoder backbone, GAB attention, Elo conditioning,
    and WDL value head from MAIA3Model, replacing the bilinear source-destination
    policy head with CFActPolicyHead.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert (
            getattr(cfg, "use_gab", False)
            + getattr(cfg, "use_relative_bias", False)
            + getattr(cfg, "use_absolute_pe", False)
        ) >= 1, "At least one of use_gab, use_relative_bias, use_absolute_pe must be True"

        # 1. Player Elo conditioning embeddings (identical to CF-BASE)
        self.elo_embedding_low = nn.Embedding(1, cfg.dim_emb)
        self.elo_embedding_high = nn.Embedding(1, cfg.dim_emb)
        nn.init.xavier_normal_(self.elo_embedding_low.weight)
        nn.init.xavier_normal_(self.elo_embedding_high.weight)

        # 2. Token input projection (identical to CF-BASE)
        time_info_dims = 4 if getattr(cfg, "include_time_info", False) else 1
        self.token_projection = nn.Linear(
            12 * cfg.history + time_info_dims - 1 + 2 * cfg.dim_emb,
            cfg.dim_vit,
        )

        # 3. Geometric Attention Bias (GAB) shared weight
        if getattr(cfg, "use_gab", False):
            self.gab_shared_weight = nn.Parameter(torch.empty(64 * 64, cfg.gab_gen_size))
            nn.init.xavier_normal_(self.gab_shared_weight)
        else:
            self.gab_shared_weight = None

        # 4. Optional absolute PE
        if getattr(cfg, "use_absolute_pe", False):
            self.abs_pe = AbsolutePE(cfg.dim_vit)

        # 5. Transformer encoder backbone (identical to CF-BASE)
        self.transformer = CustomTransformerEncoder(
            cfg=cfg,
            dim=cfg.dim_vit,
            depth=cfg.num_blocks,
            heads=cfg.num_heads,
            mlp_dim=int(cfg.dim_vit * cfg.mlp_ratio),
            dropout=cfg.dropout,
            gab_weight=self.gab_shared_weight,
        )
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

        # 6. Value and Ponder prediction heads (identical to CF-BASE)
        self.fc_value_hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.fc_value = nn.Linear(cfg.head_hid_dim, 3)

        self.fc_ponder_hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.fc_ponder = nn.Linear(cfg.head_hid_dim, 1)

        # 7. CF-ACT Policy Head: Action Token Constructor + Shared MLP Scorer
        mlp_dim = getattr(cfg, "act_mlp_dim", cfg.dim_vit)
        self.policy_head = CFActPolicyHead(
            d_model=cfg.dim_vit,
            mlp_dim=mlp_dim,
        )

    def interpolate_elo(self, elos: torch.Tensor) -> torch.Tensor:
        upper = 5000
        elos = torch.clamp(elos, 0, upper)

        weight_low = elos / upper
        weight_high = 1 - weight_low

        elo_emb_low = self.elo_embedding_low(torch.zeros_like(elos, dtype=torch.long))
        elo_emb_high = self.elo_embedding_high(torch.zeros_like(elos, dtype=torch.long))

        return weight_low.unsqueeze(1) * elo_emb_low + weight_high.unsqueeze(1) * elo_emb_high

    def encode_board(
        self,
        tokens: torch.Tensor,
        self_elos: torch.Tensor,
        oppo_elos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the unmodified Chessformer encoder and value head.

        Returns:
            B_board: (B, 64, dim_vit) contextual square representations
            logits_value: (B, 3) WDL logits
            logits_ponder: (B,) Ponder clock prediction
        """
        if getattr(self.cfg, "include_time_info", False):
            tokens = tokens[:, :, : 12 * self.cfg.history + 4 - 1]
        else:
            tokens = tokens[:, :, : 12 * self.cfg.history]

        self_elo_embs = self.interpolate_elo(self_elos).unsqueeze(1).expand(-1, 64, -1)
        oppo_elo_embs = self.interpolate_elo(oppo_elos).unsqueeze(1).expand(-1, 64, -1)

        embs = torch.cat([tokens, self_elo_embs, oppo_elo_embs], dim=-1)
        x = self.token_projection(embs)

        if hasattr(self, "abs_pe"):
            x = self.abs_pe(x)

        x = self.transformer(x)  # (B, 64, dim_vit)

        # Compute Value and Ponder branches
        pooled = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(pooled)))
        logits_ponder = self.fc_ponder(F.relu(self.fc_ponder_hid(pooled))).squeeze(1)

        return x, logits_value, logits_ponder

    def forward_with_actions(
        self,
        tokens: torch.Tensor,
        self_elos: torch.Tensor,
        oppo_elos: torch.Tensor,
        src_sq: torch.Tensor,
        dst_sq: torch.Tensor,
        promo_type: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass explicitly conditioned on legal actions.

        Returns:
            logits_legal: (B, L_max)
            policy_probs: (B, L_max)
            logits_value: (B, 3)
            logits_ponder: (B,)
            m_0: (B, L_max, dim_vit) action tokens
        """
        B_board, logits_value, logits_ponder = self.encode_board(tokens, self_elos, oppo_elos)

        logits_legal, policy_probs, m_0 = self.policy_head(
            B_board, src_sq, dst_sq, promo_type, action_mask
        )

        return logits_legal, policy_probs, logits_value, logits_ponder, m_0

    def forward(
        self,
        tokens: torch.Tensor,
        self_elos: torch.Tensor,
        oppo_elos: torch.Tensor,
        boards: Optional[List[chess.Board]] = None,
        return_4352_vocab: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """General forward pass with optional UCI compatibility.

        If boards is provided:
            Legal actions are generated directly from the board states.
            If return_4352_vocab is True:
                Scatters the legal logits into (B, 4352) tensor (with -inf on illegal moves).
            Else:
                Returns padded legal logits (B, L_max).
        """
        B_board, logits_value, logits_ponder = self.encode_board(tokens, self_elos, oppo_elos)

        if boards is not None:
            if len(boards) != tokens.shape[0]:
                raise ValueError("number of chess.Board objects must equal input batch size")
            batch_actions = [legal_moves_to_actions(b) for b in boards]
            collated = collate_legal_actions(batch_actions, device=tokens.device)

            logits_legal, _probs, _ = self.policy_head(
                B_board,
                collated["src_sq"],
                collated["dst_sq"],
                collated["promo_type"],
                collated["action_mask"],
            )

            if return_4352_vocab:
                logits_move = CFActPolicyHead.scatter_to_vocab_4352(
                    logits_legal, collated["vocab_idx"], collated["action_mask"]
                )
            else:
                logits_move = logits_legal
        else:
            # When no board is given, we cannot enumerate legal actions without knowing the position.
            raise ValueError(
                "MAIA3ActModel requires legal actions or chess.Board objects to score moves."
            )

        return logits_move, logits_value, logits_ponder

    def load_baseline_checkpoint(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
        strict: bool = True,
        trust_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """Initialize the unchanged backbone/value/ponder from CF-BASE weights.

        A strict load is mandatory for controlled research comparisons: it
        rejects absent, unexpected, or shape-incompatible non-policy weights.
        Original source/destination/promotion-head weights are intentionally
        excluded, and the new action head retains its random initialization.
        Set strict=False only for explicitly documented exploratory transfers.
        Arbitrary pickle checkpoints require trust_checkpoint=True.
        """
        ckpt = torch.load(
            checkpoint_path, map_location=device, weights_only=not trust_checkpoint
        )
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        )
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint must contain a state dictionary")

        renamed = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}
        baseline_policy_keys = {
            "proj_sq_from.weight", "proj_sq_to.weight", "promo_bias_proj.weight",
        }
        expected = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("policy_head.")
        }
        missing = sorted(set(expected) - set(renamed))
        unexpected = sorted(set(renamed) - set(expected) - baseline_policy_keys)
        incompatible = sorted(
            k for k in set(expected) & set(renamed)
            if not isinstance(renamed[k], torch.Tensor)
            or renamed[k].shape != expected[k].shape
        )
        if strict and (missing or unexpected or incompatible):
            raise RuntimeError(
                "Baseline checkpoint parity failure: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"shape_mismatch={incompatible[:10]}"
            )

        transferable = {
            k: renamed[k] for k in expected.keys()
            if k in renamed and k not in incompatible
        }
        self.load_state_dict(transferable, strict=False)
        return {
            "loaded_keys": len(transferable),
            "expected_keys": len(expected),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": incompatible,
            "excluded_policy_keys": sorted(baseline_policy_keys & set(renamed)),
            "strict": strict,
        }
