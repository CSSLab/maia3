from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import chess
import torch
from .utils import get_all_possible_moves, mirror_move


# Promotion mapping: 0 = none, 1 = queen, 2 = rook, 3 = bishop, 4 = knight
PROMOTION_TO_IDX: Dict[Optional[str], int] = {
    None: 0,
    "q": 1,
    "r": 2,
    "b": 3,
    "n": 4,
}

IDX_TO_PROMOTION: Dict[int, Optional[str]] = {
    0: None,
    1: "q",
    2: "r",
    3: "b",
    4: "n",
}

PIECE_TYPE_TO_PROMO_IDX: Dict[int, int] = {
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}


_ALL_MOVES_LIST = get_all_possible_moves()
_ALL_MOVES_DICT = {m: i for i, m in enumerate(_ALL_MOVES_LIST)}


def get_all_moves_dict() -> Dict[str, int]:
    return _ALL_MOVES_DICT


def legal_moves_to_actions(board: chess.Board) -> List[Tuple[int, int, int, str, int]]:
    """Convert all legal moves of a board position into canonical action tuples.

    Returns:
        List of tuples: (src_square, dst_square, promo_idx, uci_move, vocab_4352_idx)
        Where src_square and dst_square are in [0, 63], oriented from the side to move
        (i.e., mirrored vertically if board.turn == chess.BLACK to match Maia-3 conventions).
        promo_idx is in [0, 4] (0=none, 1=q, 2=r, 3=b, 4=n).
        uci_move is the side-oriented UCI string.
        vocab_4352_idx is the corresponding index in the fixed 4352 vocabulary (or -1 if none).
    """
    actions = []
    is_black = (board.turn == chess.BLACK)

    for move in board.legal_moves:
        uci = move.uci()
        oriented_uci = mirror_move(uci) if is_black else uci

        from_sq = chess.square_mirror(move.from_square) if is_black else move.from_square
        to_sq = chess.square_mirror(move.to_square) if is_black else move.to_square

        if move.promotion is not None:
            promo_idx = PIECE_TYPE_TO_PROMO_IDX.get(move.promotion, 0)
        else:
            promo_idx = 0

        vocab_idx = _ALL_MOVES_DICT.get(oriented_uci, -1)
        actions.append((from_sq, to_sq, promo_idx, oriented_uci, vocab_idx))

    return actions


def collate_legal_actions(
    batch_actions: List[List[Tuple[int, int, int, str, int]]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Collate variable-length legal action lists across a batch into padded tensors.

    Args:
        batch_actions: List of length B, where each item is a list of action tuples for that position.
        device: Target torch device.

    Returns:
        Dictionary containing:
            src_sq: LongTensor of shape (B, L_max)
            dst_sq: LongTensor of shape (B, L_max)
            promo_type: LongTensor of shape (B, L_max)
            action_mask: BoolTensor of shape (B, L_max), True for valid legal actions, False for padding
            vocab_idx: LongTensor of shape (B, L_max), 4352-vocabulary indices
            legal_counts: LongTensor of shape (B,), number of legal actions per position
    """
    B = len(batch_actions)
    legal_counts = [len(actions) for actions in batch_actions]
    L_max = max(legal_counts) if legal_counts else 0

    if L_max == 0:
        # Special case: all positions in batch are terminal (0 legal moves)
        L_max = 1

    src_sq = torch.zeros((B, L_max), dtype=torch.long, device=device)
    dst_sq = torch.zeros((B, L_max), dtype=torch.long, device=device)
    promo_type = torch.zeros((B, L_max), dtype=torch.long, device=device)
    vocab_idx = torch.full((B, L_max), -1, dtype=torch.long, device=device)
    action_mask = torch.zeros((B, L_max), dtype=torch.bool, device=device)

    for b, actions in enumerate(batch_actions):
        for i, (from_s, to_s, p_idx, _uci, v_idx) in enumerate(actions):
            src_sq[b, i] = from_s
            dst_sq[b, i] = to_s
            promo_type[b, i] = p_idx
            vocab_idx[b, i] = v_idx
            action_mask[b, i] = True

    return {
        "src_sq": src_sq,
        "dst_sq": dst_sq,
        "promo_type": promo_type,
        "action_mask": action_mask,
        "vocab_idx": vocab_idx,
        "legal_counts": torch.tensor(legal_counts, dtype=torch.long, device=device),
    }
