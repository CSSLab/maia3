import os
import random
import numpy as np
import torch
import chess
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_all_possible_moves():

    all_moves = []
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)

            for target_rank in range(8):
                for target_file in range(8):
                    target_square = chess.square(target_file, target_rank)
                    move = chess.Move(square, target_square)
                    all_moves.append(chess.square_name(square) + chess.square_name(target_square))

    # NOTE: promotions will always be from rank 7 to rank 8, because we are flipping
    # the board for black to move. So we only need to consider white pawn promotions.
    promotions = []
    for file_from in 'abcdefgh':
        for file_to in 'abcdefgh':
            for piece in ['q', 'r', 'b', 'n']:
                promotions.append(f'{file_from}7{file_to}8{piece}')

    all_moves.extend(promotions)

    return all_moves


def mirror_square(square):
    file = square[0]
    rank = str(9 - int(square[1]))

    return file + rank


def mirror_move(move_uci):
    is_promotion = len(move_uci) > 4

    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    return mirrored_start + mirrored_end + promotion_piece


def parse_time_control(tc: str):
    # expected forms: "180+2", "180+0", or "180" (no increment)
    if "+" in tc:
        base, inc = tc.split("+", 1)
        return float(base), float(inc)
    else:
        return float(tc), 0.0
