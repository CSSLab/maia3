import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
import math
import chess
import numpy as np

def make_bias_map():
    # 15 * 15 in units for distance pairs to 64 * 64 pairs of squares
    out = np.zeros((225, 64*64), dtype=float)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    out[15 * (i-k+7) + (j - l + 7), 64 * (i*8+j) + k*8+l] = 1
    return out

rpe_map = make_bias_map()

class RelativeBias(nn.Module):
    def __init__(self, nheads, name=None):
        super().__init__()
        # chess is 8x8, we want relative positions so -7 to 7 or 15x15
        self.register_buffer("rpe_factorizer", torch.tensor(rpe_map, dtype=torch.float32))
        self.nheads = nheads
        self.gate = nn.Parameter(torch.zeros(nheads, 15 * 15))

    def forward(self):

        # first gather the relative positions. shape will be (h, 64, 64)
        relative_bias = self.gate @ self.rpe_factorizer
        relative_bias = relative_bias.view(self.nheads, 64, 64)
        return relative_bias

class AbsolutePE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(64, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class MHA(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        use_gab: bool = True,
        use_relative_bias: bool = False,
        gab_weight: nn.Parameter | None = None,
        *,
        gab_gen_size: int = 64,
        gab_per_square_dim: int = 16,
        gab_intermediate_dim: int = 128,
        omit_qkv_biases: bool = False,
    ):
        super().__init__()
        self.use_gab = use_gab
        self.use_relative_bias = use_relative_bias

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            bias=not omit_qkv_biases
        )

        self.num_heads = nhead
        self.gen_size = gab_gen_size
        if use_gab:
            self.gab_weight = gab_weight


            if gab_per_square_dim == 0:
                self.sm1 = None
                self.sm2 = nn.Linear(d_model, gab_intermediate_dim)
            else:
                self.sm1 = nn.Linear(d_model, gab_per_square_dim)
                self.sm2 = nn.Linear(64 * gab_per_square_dim, gab_intermediate_dim)
            self.ln1 = nn.LayerNorm(gab_intermediate_dim)
            self.sm3 = nn.Linear(
                gab_intermediate_dim, nhead * gab_gen_size
            )
            self.ln2 = nn.LayerNorm(nhead * gab_gen_size)
            self.sm_act = nn.GELU()
        elif use_relative_bias:
            self.relative_bias = RelativeBias(self.num_heads)


    def _sq_bias(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 64, d_model) – board‑square tokens only (ignore CLS, elo, …)
        returns (B, H, 64, 64) – square‑pair bias matrix
        """
        squares = x                         # (B, 64, d_model)
        B = squares.size(0)

        if self.sm1 is not None:
            y = self.sm1(squares)                                  # (B, 64, p)
            y = y.reshape(B, -1)                                   # (B, 64*p)
            y = self.sm_act(self.sm2(y))                           # (B, hdim)
        else:
            y = torch.mean(squares, dim=1)                           # (B, d_model)
            y = self.sm_act(self.sm2(y))                           # (B, hdim)


        y = self.ln1(y)
        y = self.sm_act(self.sm3(y))                           # (B, H*gen)
        y = self.ln2(y).view(B, self.num_heads, self.gen_size) # (B, H, gen)

        #  einsum → (B, H, 64*64)
        b = torch.einsum("bhi,oi->bho", y, self.gab_weight)
        return b.view(B, self.num_heads, 64, 64)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = query

        if not self.use_gab and not self.use_relative_bias:
            return self.mha(
                query, key, value,
                need_weights=need_weights,
                attn_mask=attn_mask,
                **kwargs,
            )


        if self.use_gab:
            bias = self._sq_bias(query)   # (B, H, 64, 64)
            bias = bias.reshape(-1, 64, 64)  # (B*H, 64, 64) or (H, 64, 64)


        elif self.use_relative_bias:
            bias = self.relative_bias()  # (H, 64, 64)
            B = query.size(0)
            bias = bias.unsqueeze(0).expand(B, -1, -1, -1)   # no copy
            # if your attention needs (B*H,64,64) and you accept one copy:
            bias = bias.contiguous().view(-1, 64, 64)


        if attn_mask is not None:
            bias = bias + attn_mask.to(dtype=query.dtype)

        # -------- regular attention call ------------------------------
        return self.mha(
            query, key, value,
            need_weights=need_weights,
            attn_mask=bias,
            **kwargs,
        )


class EncoderOnlyBlock(nn.Module):

    def __init__(
        self,
        cfg,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        gab_weight: nn.Parameter | None = None,
    ):
        super().__init__()

        self.self_attn = MHA(d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_gab=cfg.use_gab,
            use_relative_bias=cfg.use_relative_bias,
            gab_weight=gab_weight,
            gab_gen_size=cfg.gab_gen_size,
            gab_per_square_dim=cfg.gab_per_square_dim,
            gab_intermediate_dim=cfg.gab_intermediate_dim,
            omit_qkv_biases=cfg.omit_qkv_biases,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        Norm = RMSNorm if cfg.use_rms_norm else nn.LayerNorm
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        act_name = cfg.activation.lower()
        if act_name == "relu":
            self.activation = F.relu
        elif act_name == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation function: {act_name}")

    def forward(self, x, attn_mask=None):
        # -------------------------------------------------
        # 1) Self-attention sub-layer  (Post-LayerNorm)
        # -------------------------------------------------
        sa_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
        )
        x = self.norm1(x + self.dropout1(sa_out))        # residual → LN

        # -------------------------------------------------
        # 2) Feed-forward sub-layer  (Post-LayerNorm)
        # -------------------------------------------------
        ff_out = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )
        x = self.norm2(x + self.dropout2(ff_out))        # residual → LN

        return x


class CustomTransformerEncoder(nn.Module):

    def __init__(
        self,
        cfg,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        gab_weight: nn.Parameter | None = None
    ):
        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList(
            [
                EncoderOnlyBlock(
                    cfg=cfg,
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    gab_weight=gab_weight
                )
                for _ in range(depth)
            ]
        )

        for blk in self.layers:
            if blk.self_attn.use_gab:
                blk.self_attn.gab_weight = gab_weight

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        if attn_mask is not None and attn_mask.dim() == 3:
            b, t, _ = attn_mask.shape
            attn_mask = (
                attn_mask.unsqueeze(1)
                .expand(-1, self.heads, -1, -1)
                .reshape(b * self.heads, t, t)
            )
        for blk in self.layers:
            x = blk(x, attn_mask=attn_mask)
        return self.norm(x)


class MAIA3Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert (self.cfg.use_gab + self.cfg.use_relative_bias +
                self.cfg.use_absolute_pe) >= 1, "At least one of use_gab, use_relative_bias, use_absolute_pe must be True"

        self.elo_embedding_low = nn.Embedding(1, cfg.dim_emb)
        self.elo_embedding_high = nn.Embedding(1, cfg.dim_emb)
        nn.init.xavier_normal_(self.elo_embedding_low.weight)
        nn.init.xavier_normal_(self.elo_embedding_high.weight)

        # When include_time_info=False: 1 dim (only clk_ponder for label)
        time_info_dims = 4 if cfg.include_time_info else 1
        self.token_projection = nn.Linear(12 * cfg.history + time_info_dims - 1 + 2 * cfg.dim_emb, cfg.dim_vit)

        if cfg.use_gab:
            self.gab_shared_weight = nn.Parameter(torch.empty(64 * 64, cfg.gab_gen_size))
            nn.init.xavier_normal_(self.gab_shared_weight)
        else:
            self.gab_shared_weight = None


        if cfg.use_absolute_pe:
            self.abs_pe = AbsolutePE(cfg.dim_vit)

        self.transformer = CustomTransformerEncoder(
            cfg=cfg,
            dim=cfg.dim_vit,
            depth=cfg.num_blocks,
            heads=cfg.num_heads,
            mlp_dim=int(cfg.dim_vit * cfg.mlp_ratio),
            dropout=cfg.dropout,
            gab_weight=self.gab_shared_weight
        )
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

        self.fc_value_hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.fc_value = nn.Linear(cfg.head_hid_dim, 3)

        self.fc_ponder_hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)
        self.fc_ponder = nn.Linear(cfg.head_hid_dim, 1)

        self.proj_sq_from = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)
        self.proj_sq_to = nn.Linear(cfg.dim_vit, cfg.head_hid_dim, bias=False)

        self.promo_bias_proj = nn.Linear(cfg.head_hid_dim, 4, bias=False)  # 4 promotion types: q, r, b, n


    def interpolate_elo(self, elos):

        upper = 5000
        elos = torch.clamp(elos, 0, upper)

        weight_low = elos / upper
        weight_high = 1 - weight_low

        elo_emb_low = self.elo_embedding_low(torch.zeros_like(elos, dtype=torch.long))  # (B, dim_emb)
        elo_emb_high = self.elo_embedding_high(torch.zeros_like(elos, dtype=torch.long))  # (B, dim_emb)

        return weight_low.unsqueeze(1) * elo_emb_low + weight_high.unsqueeze(1) * elo_emb_high


    def forward(self, tokens, self_elos, oppo_elos):
        # Adjust token slicing based on whether time info is included
        # When include_time_info=True: keep 12*history + 3 (exclude last clk_ponder which is label)
        # When include_time_info=False: keep 12*history (no time dims except clk_ponder which is label)
        if self.cfg.include_time_info:
            tokens = tokens[:, :, :12 * self.cfg.history + 4 - 1]  # (B, 64, 12*history + 3), last dim is label
        else:
            tokens = tokens[:, :, :12 * self.cfg.history]  # (B, 64, 12*history), clk_ponder already excluded

        self_elo_embs = self.interpolate_elo(self_elos)      # (B, dim_emb)
        oppo_elo_embs = self.interpolate_elo(oppo_elos)      # (B, dim_emb)
        self_elo_embs = self_elo_embs.unsqueeze(1).expand(-1, 64, -1)  # (B, 64, dim_emb)
        oppo_elo_embs = oppo_elo_embs.unsqueeze(1).expand(-1, 64, -1)  # (B, 64, dim_emb)

        embs = torch.cat([tokens, self_elo_embs, oppo_elo_embs], dim=-1)  # (B, 64, tokens_dim + 2*dim_emb)
        x = self.token_projection(embs)                                    # (B, 64, dim_vit)

        if hasattr(self, "abs_pe"):
            x = self.abs_pe(x)

        x = self.transformer(x)                                            # (B, 64, dim_vit)

        sq_from = self.proj_sq_from(x[:, :64, :])
        sq_to = self.proj_sq_to(x[:, :64, :])
        scores_base = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(self.cfg.head_hid_dim)
        scores_flat = scores_base.reshape(x.size(0), 64 * 64)  # (B, 4096)

        rank7_indices = [chess.square(file, 6) for file in range(8)]  # squares 48-55
        rank8_indices = [chess.square(file, 7) for file in range(8)]  # squares 56-63

        rank8_features = sq_to[:, rank8_indices, :]  # (B, 8, head_hid_dim)
        promo_biases = self.promo_bias_proj(rank8_features) * math.sqrt(self.cfg.head_hid_dim)  # (B, 8, 4) for q,r,b,n

        promotion_logits = []
        for from_file in range(8):  # source file (a-h)
            from_sq = rank7_indices[from_file]
            for to_file in range(8):  # target file (a-h)
                to_sq = rank8_indices[to_file]
                base_score = scores_base[:, from_sq, to_sq]  # (B,)
                for piece_idx in range(4):  # q=0, r=1, b=2, n=3
                    bias = promo_biases[:, to_file, piece_idx]  # (B,)
                    promotion_logits.append((base_score + bias).unsqueeze(1))
        promotion_logits = torch.cat(promotion_logits, dim=1)  # (B, 256)
        logits_move = torch.cat([scores_flat, promotion_logits], dim=1)  # (B, 4352)

        x = self.last_ln(x.mean(dim=1))
        logits_value = self.fc_value(F.relu(self.fc_value_hid(x)))          # (B, 3)
        logits_ponder = self.fc_ponder(F.relu(self.fc_ponder_hid(x)))      # (B, 1)

        return logits_move, logits_value, logits_ponder.squeeze(1)  # (B, 4352), (B, 3), (B,)