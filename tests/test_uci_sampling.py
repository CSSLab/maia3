"""Sampler regressions; run with python -m unittest discover -s tests -v.

These tests use synthetic logits and require no model checkpoint. Only the
random draw is mocked, so the probabilities and returned vocabulary indices
can be checked exactly without statistical/flaky assertions.
"""

import unittest
from unittest.mock import patch

import torch

from maia3.uci import sample_from_logits


class SampleFromLogitsTests(unittest.TestCase):
    def assert_distribution(self, logits, top_p, probabilities, indices,
                            temperature=1.0):
        for dtype in (torch.float32, torch.float64):
            values = torch.tensor(logits, dtype=dtype)
            expected = torch.tensor(probabilities, dtype=dtype)
            for draw, expected_index in enumerate(indices):
                with self.subTest(dtype=dtype, draw=draw):
                    def choose(probs, num_samples):
                        torch.testing.assert_close(probs, expected)
                        self.assertEqual(num_samples, 1)
                        return torch.tensor([draw])

                    with patch("maia3.uci.torch.multinomial",
                               side_effect=choose) as sample:
                        index = sample_from_logits(values, temperature, top_p)
                    sample.assert_called_once()
                    self.assertIsInstance(index, int)
                    self.assertEqual(index, expected_index)

    def test_includes_move_crossing_threshold(self):
        # 60% alone is below 80%; keep the 30% move too, then renormalize.
        self.assert_distribution(
            torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log().tolist(),
            0.8, [2 / 3, 1 / 3], [0, 1],
        )

    def test_crossing_move_can_be_last_move(self):
        self.assert_distribution(
            torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log().tolist(),
            0.95, [0.6, 0.3, 0.1], [0, 1, 2],
        )

    def test_exact_threshold_does_not_keep_an_extra_move(self):
        # An exact 50% top move already reaches TopP; no second move needed.
        self.assert_distribution(
            torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64).log().tolist(),
            0.5, [1.0], [0],
        )

    def test_top_move_exceeding_threshold_is_kept(self):
        self.assert_distribution(
            torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log().tolist(),
            0.1, [1.0], [0],
        )

    def test_exact_threshold_after_multiple_moves(self):
        self.assert_distribution(
            torch.tensor([0.5, 0.25, 0.125, 0.125],
                         dtype=torch.float64).log().tolist(),
            0.75, [2 / 3, 1 / 3], [0, 1],
        )

    def test_single_legal_move(self):
        self.assert_distribution(
            [float("-inf"), 0.0, float("-inf")],
            0.99, [1.0], [1],
        )

    def test_returns_original_vocabulary_indices(self):
        self.assert_distribution(
            torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64).log().tolist(),
            0.8, [2 / 3, 1 / 3], [1, 2],
        )

    def test_temperature_is_applied_before_cutoff(self):
        # Temperature 2 turns weights 16:4:1 into 4:2:1. The first move
        # then has 4/7 probability, so TopP 0.7 needs the second move too.
        self.assert_distribution(
            torch.tensor([16.0, 4.0, 1.0], dtype=torch.float64).log().tolist(),
            0.7, [2 / 3, 1 / 3], [0, 1], temperature=2.0,
        )

    def test_masked_illegal_moves_are_excluded(self):
        self.assert_distribution(
            torch.tensor([0.0, 0.6, 0.3, 0.0, 0.1],
                         dtype=torch.float64).log().tolist(),
            0.8, [2 / 3, 1 / 3], [1, 2],
        )

    def test_top_p_one_preserves_full_distribution(self):
        self.assert_distribution(
            torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64).log().tolist(),
            1.0, [0.1, 0.6, 0.3], [0, 1, 2],
        )

    def test_zero_temperature_remains_argmax(self):
        logits = torch.tensor([float("-inf"), -2.0, -1.0, -3.0])
        for top_p in (0.1, 0.8, 1.0):
            with self.subTest(top_p=top_p):
                with patch("maia3.uci.torch.multinomial") as sample:
                    index = sample_from_logits(logits, 0.0, top_p)
                self.assertEqual(index, 2)
                sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
