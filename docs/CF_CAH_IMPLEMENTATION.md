# CF-CAH implementation

This branch extends CF-ACT with one **action-to-board** cross-attention/FFN block and preserves the Chessformer encoder, GAB, Elo conditioning, WDL and ponder pathways.

## Architecture

For each legal action \`a_i=(u_i,v_i,p_i)\`:

\`\`\`text
B = ChessformerEncoder(board)               (N,64,d)
M0 = Ws B[src] + Wd B[dst] + e[p] + bm      (N,Lmax,d)
Q = LN(M0) Wq
K = B Wk                                  # no extra K/V LayerNorm
V = B Wv
A = softmax(Q K^T / sqrt(d_head), dim=-1)  # (N,heads,Lmax,64)
U = M0 + concat(A V) Wo
C = U + FFN(LN(U))
logits = shared_CF_ACT_MLP(C)
policy = masked_softmax(logits, legal_actions)
\`\`\`

The CF-ACT action constructor and MLP scorer are reused **without changing their parameterization**. No action-to-action attention, search, or alterations to input encoding are introduced.

## New modules

- \`maia3/action_to_board.py\`: independent action-to-board block
- \`maia3/action_head_cah.py\`: \`CFCahPolicyHead\`, optionally returns attention/representations
- \`maia3/models_cf_cah.py\`: \`MAIA3CahModel\`, inherited board encoder, WDL and ponder, legal-move and 4352-vocabulary interfaces
- \`tests/test_cf_cah.py\`: unit and model integration tests

## Usage

\`\`\`python
from maia3.models_cf_cah import MAIA3CahModel
from maia3.model_registry import resolve_model_spec, apply_model_config
import types, chess, torch

cfg = types.SimpleNamespace()
apply_model_config(cfg, resolve_model_spec("maia3-5m"))
model = MAIA3CahModel(cfg).eval()
# Use the existing Maia-3 tokenization/history protocol (do not replace it).
# outputs = model(tokens, self_elos, oppo_elos, boards=[chess.Board()], return_4352_vocab=True)
\`\`\`

For attention probes call \`forward_with_details(..., return_attention=True)\`. The inherited \`forward_with_actions\` continues to return the 5-tuple from CF-ACT. The default \`forward\` returns legal or 4352-vocabulary logits, WDL logits, and ponder logits.

Terminals yield zero action probability mass as a sentinel, with no softmax over an empty action set; terminal policy rows must be excluded from cross entropy and move-matching accuracy. The 4352 adapter refuses missing/colliding move indices instead of silently dropping actions.

## Testing

\`\`\`bash
python -m unittest discover -s tests -p 'test_cf_*.py' -v
\`\`\`

Status: isolated local smoke tests of the attention/policy module passed, but this is **not** evidence that repository-wide tests, full-checkpoint parity, original benchmark reproduction, or training have passed. Run the above tests and establish the complete baseline parity and training/evaluation gates prior to reporting results. The full research documentation is maintained separately as \`CF_CAH_Complete_Technical_Documentation.md\` in the downloadable project bundle.

Source proposal: *Explicit Action Representations for Chessformer* (Muhammad Fathy, November 2026). This branch implements only the second controlled policy-head intervention, not a tree-search engine.
