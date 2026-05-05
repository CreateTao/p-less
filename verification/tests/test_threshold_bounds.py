"""Step 1 Test: Verify that p-less threshold is bounded in [1/V, max(probs)]."""

import sys
import pytest
import torch

sys.path.insert(0, ".")

from verification.tests.conftest import generate_random_distribution


VOCAB_SIZES = [100, 1000, 10000, 32000]
DISTRIBUTION_TYPES = ["random", "uniform", "peaked", "bimodal", "degenerate", "near_uniform"]
NUM_TRIALS_PER_CONFIG = 1000


class TestThresholdBounds:

    def test_p_less_threshold_lower_bound(self):
        """p-less threshold >= 1/V for all distributions."""
        for vocab_size in VOCAB_SIZES:
            lower_bound = 1.0 / vocab_size
            for dist_type in DISTRIBUTION_TYPES:
                for _ in range(NUM_TRIALS_PER_CONFIG):
                    probs = generate_random_distribution(vocab_size, dist_type)
                    p = probs.square().sum(dim=-1).item()
                    assert p >= lower_bound - 1e-10, (
                        f"Threshold below 1/V: p={p}, 1/V={lower_bound}, "
                        f"vocab={vocab_size}, dist={dist_type}"
                    )

    def test_p_less_threshold_upper_bound(self):
        """p-less threshold <= max(probs) for all distributions."""
        for vocab_size in VOCAB_SIZES:
            for dist_type in DISTRIBUTION_TYPES:
                for _ in range(NUM_TRIALS_PER_CONFIG):
                    probs = generate_random_distribution(vocab_size, dist_type)
                    p = probs.square().sum(dim=-1).item()
                    max_prob = probs.max().item()
                    assert p <= max_prob + 1e-10, (
                        f"Threshold above max(probs): p={p}, max={max_prob}, "
                        f"vocab={vocab_size}, dist={dist_type}"
                    )

    def test_p_less_threshold_at_uniform_equals_1v(self):
        """For uniform distribution, p = 1/V (tight lower bound)."""
        for vocab_size in VOCAB_SIZES:
            probs = torch.ones(vocab_size) / vocab_size
            p = probs.square().sum(dim=-1).item()
            expected = 1.0 / vocab_size
            assert abs(p - expected) < 1e-10, (
                f"Uniform threshold mismatch: p={p}, expected={expected}"
            )

    def test_p_less_threshold_at_degenerate_near_max(self):
        """For degenerate distribution, p approaches max(probs)."""
        vocab_size = 1000
        probs = generate_random_distribution(vocab_size, "degenerate")
        p = probs.square().sum(dim=-1).item()
        max_prob = probs.max().item()
        # Degenerate: p should be close to max_prob^2 ≈ 0.98
        assert p > 0.9, f"Degenerate threshold too low: p={p}"