"""Score moves with Maia3: the model's log-probability of given moves in given positions.

Use `Maia3Scorer` from Python, or the `maia3-score` command to score the moves of PGN
games. Scoring differs from `maia3-uci`'s defaults in two ways, both switchable:

- `use_history` (default on) feeds the model the real previous positions, like
  `maia3-uci --use-uci-history`. Off, the current position is repeated to fill the
  history, like `maia3-uci` without that flag.
- `use_amp` (default off) runs the model under CUDA mixed precision, like `maia3-uci`
  does by default on CUDA. Off, everything runs in fp32. It has no effect on CPU.
"""

import argparse
import collections
import csv
import dataclasses
import json
import sys
from collections.abc import Iterable, Iterator, Sequence

import chess
import chess.pgn
import torch
from torch.amp import autocast

from .dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from .model_registry import ModelResolutionError, resolve_checkpoint_path
from .uci import load_model, parse_args
from .utils import get_all_possible_moves, mirror_move, seed_everything


@dataclasses.dataclass
class ScoringInput:
    """One featurized position and the vocabulary indices of the moves to score in it."""

    tokens: torch.Tensor
    legal_moves_mask: torch.Tensor
    moves_to_score: list[int]
    self_elo: int
    oppo_elo: int


class Maia3Scorer:

    MOVE_INDEX = {move: i for i, move in enumerate(get_all_possible_moves())}

    def __init__(self, cfg, model=None, *, use_history=True, use_amp=False, batch_size=1024):
        """cfg is a config from `maia3.uci.parse_args`. Its `use_uci_history` and `use_amp`
        are engine settings and are ignored here in favour of `use_history` and `use_amp`.

        model defaults to loading cfg's checkpoint, downloading it from Hugging Face when
        cfg names a model but no checkpoint path.
        """
        self.cfg = cfg
        self.use_history = use_history
        self.use_amp = use_amp
        self.batch_size = batch_size

        if model is None:
            if cfg.checkpoint_path is None:
                cfg.checkpoint_path = resolve_checkpoint_path(
                    cfg.model_spec,
                    checkpoint_filename=cfg.checkpoint_filename,
                    cache_dir=cfg.cache_dir,
                    revision=cfg.revision,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    token=cfg.hf_token,
                )
            model = load_model(cfg)
        self.model = model

    @classmethod
    def from_pretrained(cls, model="maia3-5m", checkpoint_path=None, device=None, **kwargs):
        """model is a built-in alias, Hugging Face repo ID or URL. With checkpoint_path,
        model only picks the architecture preset matching that local checkpoint."""
        argv = ["--model", model]
        if checkpoint_path is not None:
            argv += ["--checkpoint-path", checkpoint_path]
        if device is not None:
            argv += ["--device", device]
        return cls(parse_args(argv), **kwargs)

    # -- Featurization ----------------------------------------------------

    def featurize_position(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        self_elo: int,
        oppo_elo: int,
    ) -> ScoringInput:
        """board's move stack supplies the history, so build it by pushing the game's moves
        rather than from a FEN. For whole games, `featurize_game` is faster."""
        positions = []
        replay = board.copy()
        while True:
            positions.append(tokenize_board(replay))
            if not replay.move_stack or len(positions) == self.cfg.history:
                break
            replay.pop()

        history = collections.deque(reversed(positions), maxlen=self.cfg.history)
        return self._featurize(board, history, moves, self_elo, oppo_elo)

    def featurize_game(
        self,
        game: chess.pgn.Game,
        white_elo: int,
        black_elo: int,
        moves_to_score: Iterable[Sequence[chess.Move]] | None = None,
    ) -> Iterator[ScoringInput]:
        """One input per mainline ply, for the position before that move.

        moves_to_score gives the moves to score at each ply, and defaults to the played move.
        """
        mainline = list(game.mainline_moves())
        if moves_to_score is None:
            moves_to_score = [[move] for move in mainline]

        board = game.board()
        history = collections.deque([tokenize_board(board)], maxlen=self.cfg.history)

        for played_move, moves in zip(mainline, moves_to_score, strict=True):
            if board.turn == chess.WHITE:
                self_elo, oppo_elo = white_elo, black_elo
            else:
                self_elo, oppo_elo = black_elo, white_elo

            yield self._featurize(board, history, moves, self_elo, oppo_elo)
            board.push(played_move)
            history.append(tokenize_board(board))

    def _featurize(self, board, history, moves, self_elo, oppo_elo):
        if not self.use_history:
            history = [history[-1]]

        indices = []
        for move in moves:
            if move not in board.legal_moves:
                raise ValueError(f"{move.uci()} is not legal in {board.fen()}")
            indices.append(self._move_index(board, move))

        return ScoringInput(
            tokens=get_historical_tokens(history, self.cfg,
                                         base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0),
            legal_moves_mask=get_legal_moves_mask(board, self.MOVE_INDEX),
            moves_to_score=indices,
            self_elo=self_elo,
            oppo_elo=oppo_elo,
        )

    def _move_index(self, board, move):
        uci = move.uci()
        # Maia predicts from the mover's side of the board, so black's moves are mirrored.
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        return self.MOVE_INDEX[uci]

    # -- Inference --------------------------------------------------------

    def log_probabilities(self, inputs: Iterable[ScoringInput]) -> Iterator[list[float]]:
        """For each input, in order, the log-probabilities of its moves_to_score."""
        batch = []
        for item in inputs:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield from self._infer_batch(batch)
                batch = []

        if batch:
            yield from self._infer_batch(batch)

    @torch.inference_mode()
    def _infer_batch(self, inputs):
        device = self.cfg.device

        with autocast("cuda", enabled=self.use_amp and device.startswith("cuda")):
            logits_move = self.model(
                torch.stack([x.tokens for x in inputs]).to(device),
                torch.tensor([x.self_elo for x in inputs], dtype=torch.long, device=device),
                torch.tensor([x.oppo_elo for x in inputs], dtype=torch.long, device=device),
            )[0]

        # Masking and normalizing stay in fp32 even under mixed precision.
        legal = torch.stack([x.legal_moves_mask for x in inputs]).to(device)
        logps = logits_move.float().masked_fill(~legal, float("-inf")).log_softmax(dim=-1)

        # One gather for every position's moves, rather than indexing position by position.
        rows = torch.tensor([i for i, x in enumerate(inputs) for _ in x.moves_to_score],
                            dtype=torch.long, device=device)
        moves = torch.tensor([move for x in inputs for move in x.moves_to_score],
                             dtype=torch.long, device=device)
        values = logps[rows, moves].tolist()

        offset = 0
        for x in inputs:
            n = len(x.moves_to_score)
            yield values[offset:offset + n]
            offset += n


# -- Command line ---------------------------------------------------------

OUTPUT_FIELDS = ["game", "ply", "fen", "move", "played", "self_elo", "oppo_elo", "logp"]


def parse_cli_args(argv=None):

    parser = argparse.ArgumentParser(
        prog="maia3-score",
        description="Score the moves of PGN games with Maia3, writing each move's log-probability.",
        epilog="Model options are the same as maia3-uci's, such as --model, --checkpoint-path, "
               "--device and the architecture flags; see maia3-uci --help.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pgn", action="append", dest="pgns",
                        help="PGN file to score, or - for stdin. May be repeated. Defaults to stdin")
    parser.add_argument("--output", "-o", type=str, default="-",
                        help="File to write, or - for stdout")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--moves", choices=["played", "all"], default="played",
                        help="Score only the played move, or every legal move, at each ply")
    parser.add_argument("--white-elo", "--white_elo", dest="white_elo", type=int, default=None,
                        help="White's Elo, overriding the PGN's WhiteElo header")
    parser.add_argument("--black-elo", "--black_elo", dest="black_elo", type=int, default=None,
                        help="Black's Elo, overriding the PGN's BlackElo header")
    parser.add_argument("--use-history", "--use_history", dest="use_history",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Feed the model the game's previous positions, like maia3-uci "
                             "--use-uci-history. When off, the current position is repeated")
    parser.add_argument("--use-amp", "--use_amp", dest="use_amp",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Run under CUDA mixed precision, like maia3-uci does by default. "
                             "Faster, but log-probabilities differ slightly from fp32")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1024)

    args, model_argv = parser.parse_known_args(argv)
    if "--use-uci-history" in model_argv or "--use_uci_history" in model_argv:
        parser.error("use --use-history (on by default) instead of --use-uci-history")

    return args, model_argv


def _game_elo(game, header, override):
    if override is not None:
        return override
    try:
        return int(game.headers.get(header, ""))
    except ValueError:
        return None


def _read_games(paths):
    for path in paths:
        if path == "-":
            handle = sys.stdin
        else:
            handle = open(path, encoding="utf-8-sig")
        try:
            while (game := chess.pgn.read_game(handle)) is not None:
                yield game
        finally:
            if handle is not sys.stdin:
                handle.close()


def score_games(scorer, games, *, moves="played", white_elo=None, black_elo=None, default_elo=1500):
    """One output row per scored move, as dicts with OUTPUT_FIELDS keys."""
    pending = collections.deque()

    def inputs():
        for number, game in enumerate(games, start=1):
            if game.errors:
                print(f"warning: skipping game {number}: {game.errors[0]}", file=sys.stderr, flush=True)
                continue
            board = game.board()
            if type(board) is not chess.Board or board.chess960:
                print(f"warning: skipping game {number}: only standard chess is supported",
                      file=sys.stderr, flush=True)
                continue

            white = _game_elo(game, "WhiteElo", white_elo)
            black = _game_elo(game, "BlackElo", black_elo)
            if white is None or black is None:
                print(f"warning: game {number} has no Elo for one side; using {default_elo}",
                      file=sys.stderr, flush=True)
                white = default_elo if white is None else white
                black = default_elo if black is None else black

            per_ply = []
            for ply, played in enumerate(game.mainline_moves(), start=1):
                scored = list(board.legal_moves) if moves == "all" else [played]
                if board.turn == chess.WHITE:
                    self_elo, oppo_elo = white, black
                else:
                    self_elo, oppo_elo = black, white
                pending.append((number, ply, board.fen(), played, scored, self_elo, oppo_elo))
                per_ply.append(scored)
                board.push(played)

            yield from scorer.featurize_game(game, white, black, per_ply)

    for logps in scorer.log_probabilities(inputs()):
        number, ply, fen, played, scored, self_elo, oppo_elo = pending.popleft()
        for move, logp in zip(scored, logps, strict=True):
            yield {
                "game": number,
                "ply": ply,
                "fen": fen,
                "move": move.uci(),
                "played": move == played,
                "self_elo": self_elo,
                "oppo_elo": oppo_elo,
                "logp": logp,
            }


def main(argv=None):
    args, model_argv = parse_cli_args(argv)
    cfg = parse_args(model_argv)
    seed_everything(cfg.seed)

    try:
        scorer = Maia3Scorer(cfg, use_history=args.use_history, use_amp=args.use_amp,
                             batch_size=args.batch_size)
    except ModelResolutionError as exc:
        raise SystemExit(str(exc)) from exc

    rows = score_games(scorer, _read_games(args.pgns or ["-"]), moves=args.moves,
                       white_elo=args.white_elo, black_elo=args.black_elo, default_elo=cfg.elo)

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")
    try:
        if args.format == "csv":
            writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        else:
            for row in rows:
                out.write(json.dumps(row) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
