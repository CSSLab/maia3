import chess
import torch
from .utils import mirror_move


PIECE_MAP = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  5,
    chess.KING:   6,
}


def tokenize_board(board):

    tokens = torch.zeros((64, 12), dtype=torch.float)

    if board.turn == chess.BLACK:
        board = board.mirror()

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            mapped = PIECE_MAP[piece.piece_type]
            token = mapped + (6 if piece.color == chess.BLACK else 0)
            tokens[square][token - 1] = 1

    return tokens


def get_legal_moves_mask(board, all_moves_dict):

    legal_moves_mask = torch.zeros((len(all_moves_dict),), dtype=torch.bool)
    for legal_move in board.legal_moves:
        move_uci = legal_move.uci() if board.turn == chess.WHITE else mirror_move(legal_move.uci())
        if move_uci in all_moves_dict:
            legal_moves_mask[all_moves_dict[move_uci]] = 1

    return legal_moves_mask


def get_historical_tokens(board_history, cfg, base, inc, clk_left_before, clk_ponder):
    """Build the (64, dim) input tensor for the model from a deque of past board tokenizations.

    For inference, we always pad with the earliest position when history is short, regardless
    of cfg.use_padding (otherwise we couldn't infer at all early in a game).
    """

    historical_tokens = torch.cat(list(board_history), dim=1)  # (64, 12 * len(history))
    if len(board_history) < cfg.history:
        pad = board_history[0].repeat(1, cfg.history - len(board_history))
        historical_tokens = torch.cat([pad, historical_tokens], dim=1)

    if cfg.include_time_info:
        historical_tokens = torch.cat([historical_tokens,
                                       torch.full((64, 1), base / 100),
                                       torch.full((64, 1), inc / 100),
                                       torch.full((64, 1), clk_left_before / 100),
                                       torch.full((64, 1), clk_ponder / 100)], dim=1)
    else:
        historical_tokens = torch.cat([historical_tokens,
                                       torch.full((64, 1), clk_ponder / 100)], dim=1)

    return historical_tokens
