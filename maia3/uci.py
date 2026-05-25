"""UCI-protocol wrapper around Maia3 for use as a chess engine.

Reads UCI commands from stdin and writes responses to stdout. The model receives the
current position together with up to `--history` previous positions; when launched
with `--use_uci_history`, history is reconstructed from the moves passed in
standard `position ... moves ...` commands. Otherwise,
the input is padded with the current position.

The easiest path is `maia3-uci --model maia3-79m`, which applies the matching
architecture preset and downloads the checkpoint from Hugging Face if needed.
Advanced users can still pass architectural flags directly for custom checkpoints.
"""

import argparse
import sys
from collections import deque

import chess
import torch
from torch.amp import autocast

from .dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from .model_registry import (
    ModelResolutionError,
    apply_model_config,
    format_model_list,
    resolve_checkpoint_path,
    resolve_model_spec,
)
from .models import MAIA3Model
from .utils import get_all_possible_moves, mirror_move, seed_everything


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Run Maia3 as a UCI chess engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, default=None,
                        help="Built-in alias, Hugging Face repo ID, or Hugging Face URL")
    parser.add_argument("--checkpoint", "--checkpoint-path", "--checkpoint_path",
                        dest="checkpoint_path", type=str, default=None,
                        help="Path to a local .pt checkpoint. Use with --model to apply a built-in architecture preset")
    parser.add_argument("--checkpoint-filename", "--checkpoint_filename",
                        dest="checkpoint_filename", type=str, default=None,
                        help="Checkpoint filename inside a Hugging Face repo when it cannot be auto-detected")
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None,
                        help="Optional Hugging Face cache directory")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional Hugging Face revision, branch, or commit")
    parser.add_argument("--local-files-only", "--local_files_only", dest="local_files_only",
                        action="store_true", default=False,
                        help="Use only files already present in the Hugging Face cache")
    parser.add_argument("--force-download", "--force_download", dest="force_download",
                        action="store_true", default=False,
                        help="Force re-downloading the Hugging Face checkpoint")
    parser.add_argument("--hf-token", "--hf_token", dest="hf_token", type=str, default=None,
                        help="Optional Hugging Face token for private model repos")
    parser.add_argument("--trust-checkpoint", "--trust_checkpoint", dest="trust_checkpoint",
                        action="store_true", default=False,
                        help="Allow unsafe pickle loading for trusted legacy checkpoints")
    parser.add_argument("--list-models", "--list_models", dest="list_models",
                        action="store_true", default=False,
                        help="List built-in Maia3 model aliases and exit")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # Inference behavior
    parser.add_argument("--elo", type=int, default=1500, help="Default Elo for both self and opponent (override via UCI 'setoption name SelfElo/OppoElo')")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature on the move policy. 0 = argmax")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0, help="Nucleus sampling threshold (1.0 = disabled)")
    parser.add_argument("--multipv", "--multi-pv", dest="multipv",
                        type=int, default=5,
                        help="Number of candidate moves to emit as standard UCI MultiPV info lines")
    parser.add_argument("--use-uci-history", "--use_uci_history", dest="use_uci_history", action="store_true", default=False,
                        help="Rebuild board history from UCI 'position ... moves' commands. When off, the current position is repeated to fill history")

    # Data / tokenization (must match the checkpoint)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--use-padding", "--use_padding", dest="use_padding", action="store_true", default=False)
    parser.add_argument("--include-time-info", "--include_time_info", dest="include_time_info",
                        action=argparse.BooleanOptionalAction, default=False)

    # Transformer
    parser.add_argument("--dim-emb", "--dim_emb", dest="dim_emb", type=int, default=128)
    parser.add_argument("--dim-vit", "--dim_vit", dest="dim_vit", type=int, default=192)
    parser.add_argument("--num-blocks", "--num_blocks", dest="num_blocks", type=int, default=8)
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", "--mlp_ratio", dest="mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head-hid-dim", "--head_hid_dim", dest="head_hid_dim", type=int, default=192)

    # GAB
    parser.add_argument("--use-gab", "--use_gab", dest="use_gab",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gab-gen-size", "--gab_gen_size", dest="gab_gen_size", type=int, default=64)
    parser.add_argument("--gab-per-square-dim", "--gab_per_square_dim", dest="gab_per_square_dim", type=int, default=0)
    parser.add_argument("--gab-intermediate-dim", "--gab_intermediate_dim", dest="gab_intermediate_dim", type=int, default=64)
    parser.add_argument("--use-rms-norm", "--use_rms_norm", dest="use_rms_norm",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--omit-qkv-biases", "--omit_qkv_biases", dest="omit_qkv_biases",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"])

    # Position encoding alternatives
    parser.add_argument("--use-relative-bias", "--use_relative_bias", dest="use_relative_bias",
                        action="store_true", default=False)
    parser.add_argument("--use-absolute-pe", "--use_absolute_pe", dest="use_absolute_pe",
                        action="store_true", default=False)

    # AMP for inference
    parser.add_argument("--use-amp", "--use_amp", dest="use_amp",
                        action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)

    if args.list_models:
        print(format_model_list())
        raise SystemExit(0)

    try:
        args.model_spec = None
        if args.model is not None:
            spec = resolve_model_spec(args.model)
            apply_model_config(args, spec)
            args.model_spec = spec
        elif args.checkpoint_path is None:
            parser.error("one of --model or --checkpoint-path is required")
    except ModelResolutionError as exc:
        parser.error(str(exc))

    return args


def load_model(cfg):

    model = MAIA3Model(cfg).to(cfg.device)

    ckpt = torch.load(
        cfg.checkpoint_path,
        map_location=cfg.device,
        weights_only=not getattr(cfg, "trust_checkpoint", False),
    )
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Older checkpoints used "smolgen" naming; the current model uses "gab".
    renamed = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(renamed, strict=False)
    if missing:
        print(f"warning: missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}",
              file=sys.stderr, flush=True)
    if unexpected:
        print(f"warning: unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
              file=sys.stderr, flush=True)

    model.eval()
    return model


def sample_from_logits(logits, temperature, top_p):
    """logits: 1-D tensor over the move vocabulary, already masked to legal moves
    (illegal entries replaced with -inf). Returns the chosen index."""

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True  # always keep top-1
        kept_probs = sorted_probs[keep]
        kept_idx = sorted_idx[keep]
        kept_probs = kept_probs / kept_probs.sum()
        choice = torch.multinomial(kept_probs, num_samples=1).item()
        return int(kept_idx[choice].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def _probabilities_to_permille(probs):
    scaled = [max(0.0, float(prob)) * 1000 for prob in probs]
    ints = [int(value) for value in scaled]
    remainder = 1000 - sum(ints)
    order = sorted(range(len(scaled)), key=lambda idx: scaled[idx] - ints[idx], reverse=True)
    for idx in order[:max(0, remainder)]:
        ints[idx] += 1
    return tuple(ints)


def wdl_from_value_logits(logits):
    # Value labels are [loss, draw, win] for the side to move. UCI WDL is
    # reported as [win, draw, loss], in permille.
    loss, draw, win = torch.softmax(logits.float(), dim=-1).tolist()
    return _probabilities_to_permille((win, draw, loss))


def invert_wdl(wdl):
    win, draw, loss = wdl
    return loss, draw, win


def cp_from_wdl(wdl):
    win, _draw, loss = wdl
    return win - loss


def clamp_multipv(value):
    return min(20, max(1, int(value)))


class Maia3UCIEngine:

    def __init__(self, cfg):

        self.cfg = cfg
        self.model = None
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        self.self_elo = cfg.elo
        self.oppo_elo = cfg.elo
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.multipv = clamp_multipv(cfg.multipv)

        self.board = chess.Board()
        self.history = deque(maxlen=cfg.history)
        self.pending_bestmove = None
        self.pending_search = False
        self._reset_history()

    def ensure_model_loaded(self):
        if self.model is not None:
            return

        if self.cfg.checkpoint_path is None:
            spec = getattr(self.cfg, "model_spec", None)
            if spec is None:
                raise RuntimeError("No model or checkpoint was configured.")
            print(f"resolving Maia3 checkpoint for {spec.display_name}", file=sys.stderr, flush=True)
            self.cfg.checkpoint_path = resolve_checkpoint_path(
                spec,
                checkpoint_filename=self.cfg.checkpoint_filename,
                cache_dir=self.cfg.cache_dir,
                revision=self.cfg.revision,
                local_files_only=self.cfg.local_files_only,
                force_download=self.cfg.force_download,
                token=self.cfg.hf_token,
            )

        print(f"loading Maia3 checkpoint {self.cfg.checkpoint_path}", file=sys.stderr, flush=True)
        self.model = load_model(self.cfg)
        print("Maia3 ready", file=sys.stderr, flush=True)

    def _reset_history(self):
        self.history.clear()
        # Always seed with the current tokenization so a FEN or ucinewgame still
        # has something to pad with.
        self.history.append(tokenize_board(self.board))

    def _history_after_move(self, move):
        board = self.board.copy(stack=False)
        board.push(move)
        if self.cfg.use_uci_history:
            history = deque(self.history, maxlen=self.cfg.history)
            history.append(tokenize_board(board))
        else:
            history = deque([tokenize_board(board)], maxlen=self.cfg.history)
        return history

    def _tokens_from_history(self, history):
        return get_historical_tokens(history, self.cfg,
                                     base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)

    @torch.no_grad()
    def _move_from_index(self, idx):
        move_uci = self.idx_to_move[int(idx)]
        # Predictions are in the side-to-move's perspective (board mirrored when black).
        if self.board.turn == chess.BLACK:
            move_uci = mirror_move(move_uci)

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return None
        if move not in self.board.legal_moves:
            return None
        return move

    @torch.no_grad()
    def score_moves(self):

        if self.board.is_game_over():
            return None, []

        legal_mask = get_legal_moves_mask(self.board, self.all_moves_dict)
        if not bool(legal_mask.any()):
            return None, []

        # In --use_uci_history mode, self.history already contains real prior positions
        # (it's appended to on every position-update). Otherwise we keep it as a single
        # current-position entry which get_historical_tokens will replicate to fill `history`.
        tokens = self._tokens_from_history(self.history)
        tokens = tokens.unsqueeze(0).to(self.cfg.device)
        self_elos = torch.tensor([self.self_elo], dtype=torch.long, device=self.cfg.device)
        oppo_elos = torch.tensor([self.oppo_elo], dtype=torch.long, device=self.cfg.device)

        with autocast('cuda', enabled=self.cfg.use_amp and self.cfg.device.startswith('cuda')):
            logits_move, _logits_value, _ = self.model(tokens, self_elos, oppo_elos)

        logits = logits_move[0].float()
        mask = legal_mask.to(self.cfg.device)
        logits = logits.masked_fill(~mask, float('-inf'))

        idx = sample_from_logits(logits, self.temperature, self.top_p)
        move = self._move_from_index(idx)

        probs = torch.softmax(logits, dim=-1)
        top_count = min(self.multipv, int(legal_mask.sum().item()))
        top_probs, top_idxs = torch.topk(probs, k=top_count)

        top_moves = []
        for prob, top_idx in zip(top_probs.tolist(), top_idxs.tolist()):
            top_move = self._move_from_index(top_idx)
            if top_move is not None:
                top_moves.append({"move": top_move, "policy": prob, "wdl": (0, 1000, 0)})

        if top_moves:
            candidate_tokens = torch.stack([
                self._tokens_from_history(self._history_after_move(item["move"]))
                for item in top_moves
            ]).to(self.cfg.device)
            # Candidate boards are after our move, so the side to move is the
            # current opponent. The model's WDL is from that side's perspective;
            # invert it back to the side choosing the candidate.
            candidate_self_elos = torch.full((len(top_moves),), self.oppo_elo,
                                             dtype=torch.long, device=self.cfg.device)
            candidate_oppo_elos = torch.full((len(top_moves),), self.self_elo,
                                             dtype=torch.long, device=self.cfg.device)
            with autocast('cuda', enabled=self.cfg.use_amp and self.cfg.device.startswith('cuda')):
                _, candidate_value_logits, _ = self.model(
                    candidate_tokens,
                    candidate_self_elos,
                    candidate_oppo_elos,
                )
            for item, value_logits in zip(top_moves, candidate_value_logits):
                item["wdl"] = invert_wdl(wdl_from_value_logits(value_logits))

        return move, top_moves

    # -- UCI protocol -----------------------------------------------------

    def cmd_uci(self):

        print("id name Maia3")
        print("id author CSSLab")
        print(f"option name Elo type spin default {self.cfg.elo} min 0 max 5000")
        print(f"option name SelfElo type spin default {self.cfg.elo} min 0 max 5000")
        print(f"option name OppoElo type spin default {self.cfg.elo} min 0 max 5000")
        print(f"option name Temperature type string default {self.cfg.temperature}")
        print(f"option name TopP type string default {self.cfg.top_p}")
        print(f"option name MultiPV type spin default {self.multipv} min 1 max 20")
        print("uciok", flush=True)

    def cmd_setoption(self, line):
        # Expected: "setoption name <name> value <value>"
        try:
            after_name = line.split("name", 1)[1].strip()
            name, _, value = after_name.partition("value")
            name = name.strip().lower()
            value = value.strip()
        except (IndexError, ValueError):
            return

        try:
            if name == "elo":
                self.self_elo = int(value)
                self.oppo_elo = int(value)
            elif name == "selfelo":
                self.self_elo = int(value)
            elif name == "oppoelo":
                self.oppo_elo = int(value)
            elif name == "temperature":
                self.temperature = float(value)
            elif name == "topp":
                self.top_p = float(value)
            elif name == "multipv":
                self.multipv = clamp_multipv(value)
        except ValueError:
            return

    def cmd_ucinewgame(self):
        self.board = chess.Board()
        self.pending_bestmove = None
        self.pending_search = False
        self._reset_history()

    def cmd_position(self, line):

        tokens = line.split()
        if len(tokens) < 2:
            return

        i = 1
        position_kind = tokens[i]
        if tokens[i] == "startpos":
            board = chess.Board()
            i += 1
        elif tokens[i] == "fen":
            # FEN is 6 fields
            if len(tokens) < i + 7:
                return
            fen = " ".join(tokens[i + 1:i + 7])
            try:
                board = chess.Board(fen)
            except ValueError:
                return
            i += 7
        else:
            return

        moves = []
        if i < len(tokens) and tokens[i] == "moves":
            moves = tokens[i + 1:]

        self.pending_bestmove = None
        self.pending_search = False

        if self.cfg.use_uci_history:
            new_history = deque(maxlen=self.cfg.history)
            replay_board = board.copy()
            new_history.append(tokenize_board(replay_board))
            for mv in moves:
                try:
                    move = chess.Move.from_uci(mv)
                    if move not in replay_board.legal_moves:
                        return
                    replay_board.push(move)
                except ValueError:
                    return
                new_history.append(tokenize_board(replay_board))
            self.board = replay_board
            self.history = new_history
        else:
            # Apply moves to update board, then seed history with the final position only.
            for mv in moves:
                try:
                    move = chess.Move.from_uci(mv)
                    if move not in board.legal_moves:
                        return
                    board.push(move)
                except ValueError:
                    return
            self.board = board
            self._reset_history()

    def cmd_go(self, line):
        self.ensure_model_loaded()
        move, top_moves = self.score_moves()
        for rank, item in enumerate(top_moves, start=1):
            win, draw, loss = item["wdl"]
            cp = cp_from_wdl(item["wdl"])
            print(
                f"info depth 1 multipv {rank} score cp {cp} wdl {win} {draw} {loss} "
                f"pv {item['move'].uci()}",
                flush=True,
            )

        if "infinite" in line.split():
            self.pending_bestmove = move
            self.pending_search = True
            return

        self.print_bestmove(move)

    def cmd_stop(self):
        if not self.pending_search:
            return
        self.print_bestmove(self.pending_bestmove)
        self.pending_bestmove = None
        self.pending_search = False

    def print_bestmove(self, move):
        if move is None:
            print("bestmove 0000", flush=True)
        else:
            print(f"bestmove {move.uci()}", flush=True)

    def run(self):

        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue

            if line == "uci":
                self.cmd_uci()
            elif line == "isready":
                self.ensure_model_loaded()
                print("readyok", flush=True)
            elif line == "ucinewgame":
                self.cmd_ucinewgame()
            elif line.split()[0] == "position":
                self.cmd_position(line)
            elif line.split()[0] == "go":
                self.cmd_go(line)
            elif line.split()[0] == "setoption":
                self.cmd_setoption(line)
            elif line == "quit":
                return
            elif line == "stop":
                self.cmd_stop()


def main(argv=None):
    cfg = parse_args(argv)
    seed_everything(cfg.seed)
    engine = Maia3UCIEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
