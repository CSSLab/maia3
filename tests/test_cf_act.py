import unittest
import types
import os
import tempfile
import chess
import torch
import torch.nn.functional as F

from maia3.models import MAIA3Model
from maia3.models_cf_act import MAIA3ActModel
from maia3.action_head import ActionTokenConstructor, CFActPolicyHead
from maia3.actions import (
    legal_moves_to_actions,
    collate_legal_actions,
    get_all_moves_dict,
    PROMOTION_TO_IDX,
)
from maia3.model_registry import resolve_model_spec, apply_model_config
from maia3.utils import mirror_move


class TestCFAct(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        spec = resolve_model_spec("maia3-5m")
        cls.cfg = types.SimpleNamespace()
        apply_model_config(cls.cfg, spec)

    def test_01_move_bijection_and_special_moves(self):
        """Verify that every legal move maps deterministically to a unique (src, dst, promo) tuple."""
        test_fens = [
            # 1. Starting position (White to move)
            chess.STARTING_FEN,
            # 2. Open game after 1.e4 e5 (Black to move)
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            # 3. Castling rights both sides (White to move)
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            # 4. Castling rights both sides (Black to move)
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            # 5. En passant available for White
            "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2",
            # 6. En passant available for Black
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2",
            # 7. Pawn promotions (White on rank 7 with captures and pushes)
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
            # 8. Legal pawn promotions (black), including capture
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
            "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1",
        ]

        all_moves_dict = get_all_moves_dict()

        for fen in test_fens:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            actions = legal_moves_to_actions(board)

            # Assert count match
            self.assertEqual(len(legal_moves), len(actions), f"Count mismatch on FEN: {fen}")

            seen_tuples = set()
            for from_sq, to_sq, p_idx, uci_str, v_idx in actions:
                self.assertTrue(0 <= from_sq < 64)
                self.assertTrue(0 <= to_sq < 64)
                self.assertTrue(0 <= p_idx < 5)

                action_tuple = (from_sq, to_sq, p_idx)
                self.assertNotIn(action_tuple, seen_tuples, f"Duplicate action tuple {action_tuple} in {fen}")
                seen_tuples.add(action_tuple)

                # Check vocabulary mapping consistency
                if v_idx != -1:
                    self.assertIn(uci_str, all_moves_dict)
                    self.assertEqual(all_moves_dict[uci_str], v_idx)

    def test_01b_promotions_are_legal_and_distinct(self):
        """Verify both colors, all four types, and capture promotions are reached."""
        cases = [
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8"),
            ("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a7a8"),
            ("4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1", "b7a8"),
        ]
        for fen, oriented_prefix in cases:
            board = chess.Board(fen)
            self.assertTrue(board.is_valid(), fen)
            actions = legal_moves_to_actions(board)
            for suffix, promo_id in [("q", 1), ("r", 2), ("b", 3), ("n", 4)]:
                matching = [a for a in actions if a[3] == oriented_prefix + suffix]
                self.assertEqual(len(matching), 1, (fen, suffix))
                self.assertEqual(matching[0][2], promo_id)
                self.assertGreaterEqual(matching[0][4], 0)
                self.assertEqual(matching[0][4], get_all_moves_dict()[matching[0][3]])

    def test_02_shape_integrity(self):
        """Verify tensor shapes at all stages across variable legal move counts."""
        model = MAIA3ActModel(self.cfg)
        model.eval()

        boards = [
            chess.Board(),  # 20 legal moves
            chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),  # 29 legal moves
            chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"),  # few moves
        ]
        B = len(boards)
        tokens = torch.zeros(B, 64, 12 * self.cfg.history + 1)
        self_elo = torch.tensor([1500, 1600, 1700])
        oppo_elo = torch.tensor([1500, 1550, 1700])

        batch_actions = [legal_moves_to_actions(b) for b in boards]
        collated = collate_legal_actions(batch_actions)
        L_max = max(len(a) for a in batch_actions)

        # 1. Forward with actions
        logits_legal, policy_probs, val, ponder, m_0 = model.forward_with_actions(
            tokens,
            self_elo,
            oppo_elo,
            collated["src_sq"],
            collated["dst_sq"],
            collated["promo_type"],
            collated["action_mask"],
        )

        self.assertEqual(m_0.shape, (B, L_max, self.cfg.dim_vit))
        self.assertEqual(logits_legal.shape, (B, L_max))
        self.assertEqual(policy_probs.shape, (B, L_max))
        self.assertEqual(val.shape, (B, 3))
        self.assertEqual(ponder.shape, (B,))

        # 2. Forward returning 4352 vocabulary
        logits_4352, _, _ = model(tokens, self_elo, oppo_elo, boards=boards, return_4352_vocab=True)
        self.assertEqual(logits_4352.shape, (B, 4352))

    def test_03_probability_and_normalization_integrity(self):
        """Verify that legal move probabilities sum to 1 and padding has 0 probability."""
        model = MAIA3ActModel(self.cfg)
        model.eval()

        boards = [
            chess.Board(),
            chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"),
        ]
        B = len(boards)
        tokens = torch.zeros(B, 64, 12 * self.cfg.history + 1)
        self_elo = torch.tensor([1500, 1500])
        oppo_elo = torch.tensor([1500, 1500])

        batch_actions = [legal_moves_to_actions(b) for b in boards]
        collated = collate_legal_actions(batch_actions)

        _logits, probs, _, _, _ = model.forward_with_actions(
            tokens,
            self_elo,
            oppo_elo,
            collated["src_sq"],
            collated["dst_sq"],
            collated["promo_type"],
            collated["action_mask"],
        )

        for b in range(B):
            mask = collated["action_mask"][b]
            legal_probs = probs[b][mask]
            padded_probs = probs[b][~mask]

            self.assertAlmostEqual(legal_probs.sum().item(), 1.0, places=5)
            self.assertTrue(torch.all(padded_probs == 0.0).item())
            self.assertFalse(torch.isnan(probs[b]).any().item())

    def test_04_terminal_position_handling(self):
        """Verify checkmate and stalemate positions (0 legal moves) produce clean zero outputs without NaNs."""
        model = MAIA3ActModel(self.cfg)
        model.eval()

        # Scholar's mate checkmate position
        checkmate_board = chess.Board("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        self.assertTrue(checkmate_board.is_checkmate())
        self.assertEqual(len(list(checkmate_board.legal_moves)), 0)

        # Stalemate position
        stalemate_board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
        self.assertTrue(stalemate_board.is_stalemate())
        self.assertEqual(len(list(stalemate_board.legal_moves)), 0)

        boards = [checkmate_board, stalemate_board, chess.Board()]
        B = len(boards)
        tokens = torch.zeros(B, 64, 12 * self.cfg.history + 1)
        self_elo = torch.tensor([1500, 1500, 1500])
        oppo_elo = torch.tensor([1500, 1500, 1500])

        batch_actions = [legal_moves_to_actions(b) for b in boards]
        collated = collate_legal_actions(batch_actions)

        _logits, probs, val, ponder, _ = model.forward_with_actions(
            tokens,
            self_elo,
            oppo_elo,
            collated["src_sq"],
            collated["dst_sq"],
            collated["promo_type"],
            collated["action_mask"],
        )

        # First two positions should have zero probability mass and NO NaNs
        self.assertFalse(torch.isnan(probs).any().item())
        self.assertTrue(torch.all(probs[0] == 0.0).item())
        self.assertTrue(torch.all(probs[1] == 0.0).item())
        # Third position (regular) sums to 1.0
        self.assertAlmostEqual(probs[2][collated["action_mask"][2]].sum().item(), 1.0, places=5)

    def test_05_gradient_flow_and_frozen_backbone(self):
        """Verify gradient flow under frozen backbone: encoder weights untouched, policy head updated."""
        model = MAIA3ActModel(self.cfg)
        model.train()

        # Freeze encoder and value heads (Stage 1 isolation protocol)
        for name, param in model.named_parameters():
            if not name.startswith("policy_head"):
                param.requires_grad = False

        board = chess.Board()
        actions = legal_moves_to_actions(board)
        collated = collate_legal_actions([actions])

        tokens = torch.zeros(1, 64, 12 * self.cfg.history + 1)
        self_elo = torch.tensor([1500])
        oppo_elo = torch.tensor([1500])

        logits_legal, probs, _val, _ponder, _ = model.forward_with_actions(
            tokens,
            self_elo,
            oppo_elo,
            collated["src_sq"],
            collated["dst_sq"],
            collated["promo_type"],
            collated["action_mask"],
        )

        # Policy cross-entropy loss against target move index 0
        target = torch.tensor([0])
        loss = F.cross_entropy(logits_legal, target)
        loss.backward()

        # Check policy head parameters have valid gradients
        self.assertIsNotNone(model.policy_head.constructor.proj_src.weight.grad)
        self.assertIsNotNone(model.policy_head.constructor.proj_dst.weight.grad)
        self.assertIsNotNone(model.policy_head.constructor.promo_embedding.weight.grad)
        self.assertIsNotNone(model.policy_head.constructor.action_bias.grad)
        self.assertIsNotNone(model.policy_head.fc1.weight.grad)
        self.assertIsNotNone(model.policy_head.fc2.weight.grad)

        # Check encoder parameters strictly have None gradients
        self.assertIsNone(model.token_projection.weight.grad)
        self.assertIsNone(model.fc_value.weight.grad)
        self.assertIsNone(model.transformer.layers[0].linear1.weight.grad)

    def test_06_backbone_parity_with_cf_base(self):
        """Verify that MAIA3ActModel shares exact encoder and value outputs when using identical weights."""
        base_model = MAIA3Model(self.cfg)
        act_model = MAIA3ActModel(self.cfg)

        # Copy encoder and value weights from base to act model
        base_state = base_model.state_dict()
        act_state = act_model.state_dict()

        for k in base_state:
            if k in act_state:
                act_state[k].copy_(base_state[k])

        base_model.eval()
        act_model.eval()

        tokens = torch.randn(2, 64, 12 * self.cfg.history + 1)
        self_elo = torch.tensor([1500, 1800])
        oppo_elo = torch.tensor([1500, 1600])

        with torch.no_grad():
            _, base_val, base_ponder = base_model(tokens, self_elo, oppo_elo)
            _, act_val, act_ponder = act_model.encode_board(tokens, self_elo, oppo_elo)

        self.assertTrue(torch.allclose(base_val, act_val, atol=1e-5))
        self.assertTrue(torch.allclose(base_ponder, act_ponder, atol=1e-5))

    def test_07_scatter_rejects_invalid_and_colliding_indices(self):
        logits = torch.tensor([[1.0, 2.0]])
        mask = torch.tensor([[True, True]])
        with self.assertRaisesRegex(ValueError, "unmapped or out-of-range"):
            CFActPolicyHead.scatter_to_vocab_4352(logits, torch.tensor([[0, -1]]), mask)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CFActPolicyHead.scatter_to_vocab_4352(logits, torch.tensor([[3, 3]]), mask)
        output = CFActPolicyHead.scatter_to_vocab_4352(logits, torch.tensor([[3, 9]]), mask)
        self.assertEqual(output.shape, (1, 4352))
        self.assertEqual(output[0, 3].item(), 1.0)
        self.assertEqual(output[0, 9].item(), 2.0)
        self.assertEqual(torch.isfinite(output).sum().item(), 2)

    def test_08_strict_checkpoint_load_and_missing_weights(self):
        original = MAIA3Model(self.cfg)
        modified = MAIA3ActModel(self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            good = os.path.join(directory, "good.pt")
            bad = os.path.join(directory, "missing.pt")
            torch.save({"model_state_dict": original.state_dict()}, good)
            report = modified.load_baseline_checkpoint(good)
            self.assertEqual(report["loaded_keys"], report["expected_keys"])
            self.assertEqual(report["missing_keys"], [])
            self.assertEqual(report["unexpected_keys"], [])
            self.assertEqual(report["shape_mismatch_keys"], [])
            self.assertTrue(torch.equal(modified.fc_value.weight, original.fc_value.weight))

            incomplete = dict(original.state_dict())
            del incomplete["fc_value.weight"]
            torch.save({"model_state_dict": incomplete}, bad)
            before = modified.fc_value.weight.detach().clone()
            with self.assertRaisesRegex(RuntimeError, "checkpoint parity failure"):
                modified.load_baseline_checkpoint(bad)
            self.assertTrue(torch.equal(modified.fc_value.weight, before))
            report = modified.load_baseline_checkpoint(bad, strict=False)
            self.assertIn("fc_value.weight", report["missing_keys"])

    def test_09_strict_checkpoint_rejects_wrong_shape(self):
        original = MAIA3Model(self.cfg)
        broken = dict(original.state_dict())
        broken["fc_value.weight"] = broken["fc_value.weight"][:2].clone()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "wrong_shape.pt")
            torch.save(broken, path)
            with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
                MAIA3ActModel(self.cfg).load_baseline_checkpoint(path)

    def test_10_forward_rejects_wrong_board_batch_length(self):
        model = MAIA3ActModel(self.cfg)
        x = torch.zeros(2, 64, 12 * self.cfg.history + 1)
        elo = torch.tensor([1500, 1500])
        with self.assertRaisesRegex(ValueError, "board objects must equal"):
            model(x, elo, elo, boards=[chess.Board()])


if __name__ == "__main__":
    unittest.main()
