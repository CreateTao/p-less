"""Step 1 Test: Verify that p-less and p-less-norm always produce non-empty candidate sets."""

import sys
import pytest
import torch

sys.path.insert(0, ".")

from p_less_samplers import p_less_decode, p_less_norm_decode
from verification.tests.conftest import generate_random_distribution


VOCAB_SIZES = [100, 1000, 10000, 32000]
DISTRIBUTION_TYPES = ["random", "uniform", "peaked", "bimodal", "degenerate", "near_uniform"]
NUM_TRIALS_PER_CONFIG = 1000


def _compute_candidate_set_size_p_less(probs: torch.Tensor) -> int:
    """Compute candidate set size for p-less without mutation."""
    p = probs.square().sum(dim=-1).item()
    return (probs >= p).sum().item()


def _compute_candidate_set_size_p_less_norm(probs: torch.Tensor) -> int:
    """Compute candidate set size for p-less-norm without mutation."""
    v = probs.size(-1)
    p_norm = ((v * probs.square().sum(dim=-1)) - 1.0) / (v - 1.0)
    return (probs >= p_norm.item()).sum().item()


class TestCandidateSetNonEmpty:

    def test_p_less_all_distribution_types(self):
        """p-less candidate set is always non-empty across all distribution types."""
        for vocab_size in VOCAB_SIZES:
            for dist_type in DISTRIBUTION_TYPES:
                for _ in range(NUM_TRIALS_PER_CONFIG):
                    probs = generate_random_distribution(vocab_size, dist_type)
                    size = _compute_candidate_set_size_p_less(probs)
                    assert size >= 1, (
                        f"Empty candidate set: vocab={vocab_size}, "
                        f"dist={dist_type}, size={size}"
                    )

    def test_p_less_norm_all_distribution_types(self):
        """p-less-norm candidate set is always non-empty across all distribution types."""
        for vocab_size in VOCAB_SIZES:
            for dist_type in DISTRIBUTION_TYPES:
                for _ in range(NUM_TRIALS_PER_CONFIG):
                    probs = generate_random_distribution(vocab_size, dist_type)
                    size = _compute_candidate_set_size_p_less_norm(probs)
                    assert size >= 1, (
                        f"Empty candidate set (norm): vocab={vocab_size}, "
                        f"dist={dist_type}, size={size}"
                    )

    def test_p_less_norm_uniform_retains_all(self):
        """p-less-norm on uniform distribution retains all tokens."""
        for vocab_size in VOCAB_SIZES:
            probs = torch.ones(vocab_size) / vocab_size
            size = _compute_candidate_set_size_p_less_norm(probs)
            assert size == vocab_size, (
                f"Uniform distribution should retain all tokens: "
                f"vocab={vocab_size}, retained={size}"
            )

    def test_p_less_decoding_returns_valid_token(self):
        """p_less_decode returns a valid token index."""
        torch.manual_seed(42)
        for vocab_size in VOCAB_SIZES:
            for dist_type in ["random", "peaked", "bimodal"]:
                probs = generate_random_distribution(vocab_size, dist_type)
                probs_clone = probs.clone()
                next_token = p_less_decode(probs_clone)
                assert 0 <= next_token.item() < vocab_size, (
                    f"Invalid token index: {next_token.item()}, vocab={vocab_size}"
                )

    def test_p_less_norm_decoding_returns_valid_token(self):
        """p_less_norm_decode returns a valid token index."""
        torch.manual_seed(42)
        for vocab_size in VOCAB_SIZES:
            for dist_type in ["random", "peaked", "bimodal"]:
                probs = generate_random_distribution(vocab_size, dist_type)
                probs_clone = probs.clone()
                next_token = p_less_norm_decode(probs_clone)
                assert 0 <= next_token.item() < vocab_size, (
                    f"Invalid token index (norm): {next_token.item()}, vocab={vocab_size}"
                )