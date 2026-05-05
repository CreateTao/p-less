"""Step 1 Test: Verify renormalization correctness (sum == 1.0 after p-less)."""

import sys
import pytest
import torch

sys.path.insert(0, ".")

from p_less_samplers import p_less_decode, p_less_norm_decode
from verification.tests.conftest import generate_random_distribution


VOCAB_SIZES = [100, 1000, 10000]
DISTRIBUTION_TYPES = ["random", "peaked", "bimodal", "degenerate"]
NUM_TRIALS = 100


class TestRenormalization:

    def test_p_less_output_distribution_sums_to_one(self):
        """After p-less masking and renormalization, probs sum == 1.0."""
        for vocab_size in VOCAB_SIZES:
            for dist_type in DISTRIBUTION_TYPES:
                for _ in range(NUM_TRIALS):
                    probs = generate_random_distribution(vocab_size, dist_type)
                    probs_clone = probs.clone()
                    _ = p_less_decode(probs_clone)
                    # After mutation, surviving probs should sum to 1.0
                    total = probs_clone.sum().item()
                    assert abs(total - 1.0) < 1e-6, (
                        f"Renormalization failed: sum={total}, "
                        f"vocab={vocab_size}, dist={dist_type}"
                    )

    def test_p_less_norm_output_distribution_sums_to_one(self):
        """After p-less-norm masking and renormalization, probs sum == 1.0."""
        for vocab_size in VOCAB_SIZES:
            for dist_type in DISTRIBUTION_TYPES:
                for _ in range(NUM_TRIALS):
                    probs = generate_random_distribution(vocab_size, dist_type)
                    probs_clone = probs.clone()
                    _ = p_less_norm_decode(probs_clone)
                    total = probs_clone.sum().item()
                    assert abs(total - 1.0) < 1e-6, (
                        f"Renormalization failed (norm): sum={total}, "
                        f"vocab={vocab_size}, dist={dist_type}"
                    )

    def test_p_less_masked_tokens_are_zero(self):
        """Tokens below threshold are exactly zero after p-less."""
        for vocab_size in VOCAB_SIZES:
            probs = generate_random_distribution(vocab_size, "peaked")
            probs_clone = probs.clone()

            threshold = probs_clone.square().sum(dim=-1).item()
            mask_before = probs_clone < threshold

            _ = p_less_decode(probs_clone)

            # Check that masked positions are exactly 0.0
            for i in range(vocab_size):
                if mask_before[i]:
                    assert probs_clone[i].item() == 0.0, (
                        f"Masked token not zero: index={i}, value={probs_clone[i].item()}"
                    )

    def test_p_less_no_negative_probabilities(self):
        """All probabilities are non-negative after p-less."""
        for vocab_size in VOCAB_SIZES:
            for dist_type in ["random", "peaked"]:
                probs = generate_random_distribution(vocab_size, dist_type)
                probs_clone = probs.clone()
                _ = p_less_decode(probs_clone)
                assert (probs_clone >= 0).all(), "Negative probabilities found after p-less"