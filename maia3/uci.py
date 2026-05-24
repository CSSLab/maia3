"""UCI-protocol wrapper around Maia3 for use as a chess engine.

Reads UCI commands from stdin and writes responses to stdout. The model receives the
current position together with up to `--history` previous positions; when launched
with `--use_uci_history`, history is reconstructed from the moves passed in
`position startpos moves ...`. Otherwise (or when the position arrives via `fen`),
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
        if args.model is not None:
            spec = resolve_model_spec(args.model)
            apply_model_config(args, spec)
            if args.checkpoint_path is None:
                args.checkpoint_path = resolve_checkpoint_path(
                    spec,
                    checkpoint_filename=args.checkpoint_filename,
                    cache_dir=args.cache_dir,
                    revision=args.revision,
                    local_files_only=args.local_files_only,
                    force_download=args.force_download,
                    token=args.hf_token,
                )
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
        print(f"info string warning missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    if unexpected:
        print(f"info string warning unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)

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


class Maia3UCIEngine:

    def __init__(self, cfg):

        self.cfg = cfg
        self.model = load_model(cfg)
        self.all_moves = get_all_possible_moves()
        self.all_moves_dict = {m: i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for m, i in self.all_moves_dict.items()}

        self.self_elo = cfg.elo
        self.oppo_elo = cfg.elo
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p

        self.board = chess.Board()
        self.history = deque(maxlen=cfg.history)
        self._reset_history()

    def _reset_history(self):
        self.history.clear()
        # Always seed with the current (post-mirror) tokenization so a bare FEN or
        # ucinewgame still has something to pad with.
        self.history.append(tokenize_board(self.board))

    @torch.no_grad()
    def pick_move(self):

        if self.board.is_game_over():
            return None

        legal_mask = get_legal_moves_mask(self.board, self.all_moves_dict)
        if not bool(legal_mask.any()):
            return None

        # In --use_uci_history mode, self.history already contains real prior positions
        # (it's appended to on every position-update). Otherwise we keep it as a single
        # current-position entry which get_historical_tokens will replicate to fill `history`.
        tokens = get_historical_tokens(self.history, self.cfg,
                                       base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
        tokens = tokens.unsqueeze(0).to(self.cfg.device)
        self_elos = torch.tensor([self.self_elo], dtype=torch.long, device=self.cfg.device)
        oppo_elos = torch.tensor([self.oppo_elo], dtype=torch.long, device=self.cfg.device)

        with autocast('cuda', enabled=self.cfg.use_amp and self.cfg.device.startswith('cuda')):
            logits_move, _, _ = self.model(tokens, self_elos, oppo_elos)

        logits = logits_move[0].float()
        mask = legal_mask.to(self.cfg.device)
        logits = logits.masked_fill(~mask, float('-inf'))

        idx = sample_from_logits(logits, self.temperature, self.top_p)
        move_uci = self.idx_to_move[idx]

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

    # -- UCI protocol -----------------------------------------------------

    def cmd_uci(self):

        print("id name Maia3")
        print("id author CSSLab")
        print(f"option name Elo type spin default {self.cfg.elo} min 0 max 5000")
        print(f"option name SelfElo type spin default {self.cfg.elo} min 0 max 5000")
        print(f"option name OppoElo type spin default {self.cfg.elo} min 0 max 5000")
        print(f"option name Temperature type string default {self.cfg.temperature}")
        print(f"option name TopP type string default {self.cfg.top_p}")
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

    def cmd_ucinewgame(self):
        self.board = chess.Board()
        self._reset_history()

    def cmd_position(self, line):

        tokens = line.split()
        if len(tokens) < 2:
            return

        i = 1
        if tokens[i] == "startpos":
            self.board = chess.Board()
            i += 1
        elif tokens[i] == "fen":
            # FEN is 6 fields
            fen = " ".join(tokens[i + 1:i + 7])
            self.board = chess.Board(fen)
            i += 7
        else:
            return

        moves = []
        if i < len(tokens) and tokens[i] == "moves":
            moves = tokens[i + 1:]

        if self.cfg.use_uci_history and (tokens[1] == "startpos" or not moves):
            # Reconstruct history by replaying from a known starting board.
            # If the position came via FEN with no moves, we only have one snapshot.
            self.history.clear()
            replay_board = chess.Board() if tokens[1] == "startpos" else chess.Board(self.board.fen())
            self.history.append(tokenize_board(replay_board))
            for mv in moves:
                replay_board.push(chess.Move.from_uci(mv))
                self.history.append(tokenize_board(replay_board))
            # Apply any remaining moves to self.board
            self.board = replay_board
        else:
            # Apply moves to update board, then seed history with the final position only.
            for mv in moves:
                self.board.push(chess.Move.from_uci(mv))
            self._reset_history()

    def cmd_go(self, line):
        move = self.pick_move()
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
                print("readyok", flush=True)
            elif line == "ucinewgame":
                self.cmd_ucinewgame()
            elif line.startswith("position"):
                self.cmd_position(line)
            elif line.startswith("go"):
                self.cmd_go(line)
            elif line.startswith("setoption"):
                self.cmd_setoption(line)
            elif line == "quit":
                return
            elif line == "stop":
                continue  # we're synchronous; nothing to interrupt


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    engine = Maia3UCIEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
