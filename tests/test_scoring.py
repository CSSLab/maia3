import io
import json
import math

import chess
import chess.pgn
import pytest
import torch

from maia3.models import MAIA3Model
from maia3.scoring import Maia3Scorer, main, score_games
from maia3.uci import Maia3UCIEngine, parse_args

# A small randomly initialized model: these tests check the scoring plumbing, not the weights.
MODEL_ARGS = ["--device", "cpu", "--num-blocks", "2", "--no-use-amp"]
MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "e1g1", "f7f6"]
PGN = """[WhiteElo "1800"]
[BlackElo "1400"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O f6 *
"""


@pytest.fixture(scope="module")
def cfg():
    return parse_args(["--checkpoint-path", "unused.pt", *MODEL_ARGS])


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return MAIA3Model(cfg).eval()


def flatten(results):
    return [logp for logps in results for logp in logps]


def board_after(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


def engine_probabilities(cfg, model, moves, use_uci_history):
    engine_cfg = parse_args(["--checkpoint-path", "unused.pt", *MODEL_ARGS, "--multipv", "20"]
                            + (["--use-uci-history"] if use_uci_history else []))
    engine = Maia3UCIEngine(engine_cfg)
    engine.model = model
    engine.cmd_position("position startpos moves " + " ".join(moves))
    _move, top_moves = engine.score_moves()
    return {item["move"]: item["policy"] for item in top_moves}


@pytest.mark.parametrize("plies", [0, 5, 10])
def test_legal_move_probabilities_sum_to_one(cfg, model, plies):
    scorer = Maia3Scorer(cfg, model)
    board = board_after(MOVES[:plies])
    legal = list(board.legal_moves)

    [logps] = scorer.log_probabilities([scorer.featurize_position(board, legal, 1500, 1500)])

    assert len(logps) == len(legal)
    assert math.isclose(sum(math.exp(logp) for logp in logps), 1.0, abs_tol=1e-5)


@pytest.mark.parametrize("use_history", [False, True])
@pytest.mark.parametrize("plies", [5, 10])
def test_matches_uci_engine(cfg, model, use_history, plies):
    scorer = Maia3Scorer(cfg, model, use_history=use_history)
    board = board_after(MOVES[:plies])
    expected = engine_probabilities(cfg, model, MOVES[:plies], use_uci_history=use_history)

    [logps] = scorer.log_probabilities(
        [scorer.featurize_position(board, list(expected), 1500, 1500)])

    for (move, prob), logp in zip(expected.items(), logps):
        assert math.isclose(math.exp(logp), prob, rel_tol=1e-4), move.uci()


def test_history_changes_scores(cfg, model):
    board = board_after(MOVES)
    legal = list(board.legal_moves)

    results = []
    for use_history in [True, False]:
        scorer = Maia3Scorer(cfg, model, use_history=use_history)
        results.append(next(scorer.log_probabilities(
            [scorer.featurize_position(board, legal, 1500, 1500)])))

    assert results[0] != pytest.approx(results[1])


def test_featurize_game_matches_featurize_position(cfg, model):
    scorer = Maia3Scorer(cfg, model)
    game = chess.pgn.read_game(io.StringIO(PGN))

    from_game = list(scorer.log_probabilities(scorer.featurize_game(game, 1800, 1400)))

    board = chess.Board()
    from_positions = []
    for move in MOVES:
        played = chess.Move.from_uci(move)
        elos = (1800, 1400) if board.turn == chess.WHITE else (1400, 1800)
        from_positions.append(scorer.featurize_position(board, [played], *elos))
        board.push(played)

    assert flatten(from_game) == pytest.approx(
        flatten(scorer.log_probabilities(from_positions)), abs=1e-5)


def test_batch_size_does_not_change_results(cfg, model):
    game = chess.pgn.read_game(io.StringIO(PGN))
    results = []
    for batch_size in [1, 3, 1024]:
        scorer = Maia3Scorer(cfg, model, batch_size=batch_size)
        results.append(flatten(scorer.log_probabilities(scorer.featurize_game(game, 1800, 1400))))

    assert results[0] == pytest.approx(results[1], abs=1e-5)
    assert results[0] == pytest.approx(results[2], abs=1e-5)


def test_positions_with_no_moves_to_score(cfg, model):
    scorer = Maia3Scorer(cfg, model)
    inputs = [scorer.featurize_position(board_after(MOVES[:plies]), [], 1500, 1500)
              for plies in range(3)]

    assert list(scorer.log_probabilities(inputs)) == [[], [], []]


def test_illegal_move_raises(cfg, model):
    scorer = Maia3Scorer(cfg, model)

    with pytest.raises(ValueError, match="e2e5 is not legal"):
        scorer.featurize_position(chess.Board(), [chess.Move.from_uci("e2e5")], 1500, 1500)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed precision needs CUDA")
def test_mixed_precision_is_close_to_fp32():
    cuda_cfg = parse_args(["--checkpoint-path", "unused.pt", "--device", "cuda", "--num-blocks", "2"])
    torch.manual_seed(0)
    cuda_model = MAIA3Model(cuda_cfg).to("cuda").eval()
    board = board_after(MOVES)
    legal = list(board.legal_moves)

    results = []
    for use_amp in [False, True]:
        scorer = Maia3Scorer(cuda_cfg, cuda_model, use_amp=use_amp)
        results.append(next(scorer.log_probabilities(
            [scorer.featurize_position(board, legal, 1500, 1500)])))

    assert results[1] == pytest.approx(results[0], abs=0.05)


def test_score_games_rows(cfg, model):
    scorer = Maia3Scorer(cfg, model)
    game = chess.pgn.read_game(io.StringIO(PGN))

    played = list(score_games(scorer, [game]))
    every = list(score_games(scorer, [game], moves="all"))

    assert [row["move"] for row in played] == MOVES
    assert all(row["played"] for row in played)
    assert [(row["self_elo"], row["oppo_elo"]) for row in played[:2]] == [(1800, 1400), (1400, 1800)]
    assert [row["logp"] for row in every if row["played"]] == pytest.approx(
        [row["logp"] for row in played], abs=1e-5)
    assert len(every) == sum(board_after(MOVES[:ply]).legal_moves.count() for ply in range(len(MOVES)))


def test_cli_scores_pgn_file(cfg, model, tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    pgn = tmp_path / "games.pgn"
    pgn.write_text(PGN + "\n" + PGN)
    output = tmp_path / "scores.jsonl"

    main(["--pgn", str(pgn), "--output", str(output), "--checkpoint-path", str(checkpoint),
          "--device", "cpu", "--num-blocks", "2"])

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    expected = list(score_games(Maia3Scorer(cfg, model), [chess.pgn.read_game(io.StringIO(PGN))]))
    assert len(rows) == 2 * len(MOVES)
    assert [row["logp"] for row in rows[:len(MOVES)]] == pytest.approx(
        [row["logp"] for row in expected], abs=1e-5)
    assert [row["game"] for row in rows] == [1] * len(MOVES) + [2] * len(MOVES)
