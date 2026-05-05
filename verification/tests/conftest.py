from __future__ import annotations

import pytest
import torch
import numpy as np


def generate_random_distribution(vocab_size: int, distribution_type: str = "random") -> torch.Tensor:
    """Generate a random probability distribution for testing.

    Args:
        vocab_size: Size of the vocabulary
        distribution_type: Type of distribution to generate
            - "random": Dirichlet-sampled random distribution
            - "uniform": Perfectly uniform distribution
            - "peaked": Single peak with small noise
            - "bimodal": Two peaks
            - "degenerate": One token dominates (99%)
            - "near_uniform": Nearly uniform with slight variation
    """
    if distribution_type == "uniform":
        return torch.ones(vocab_size) / vocab_size

    if distribution_type == "peaked":
        probs = torch.full((vocab_size,), 1e-6)
        probs[0] = 0.9
        probs[1:] = (1.0 - 0.9) / (vocab_size - 1)
        return probs / probs.sum()

    if distribution_type == "bimodal":
        probs = torch.full((vocab_size,), 1e-6)
        probs[0] = 0.4
        probs[1] = 0.4
        remaining = 1.0 - 0.4 - 0.4 - (vocab_size - 2) * 1e-6
        probs[2:] = remaining / (vocab_size - 2) + 1e-6
        return probs / probs.sum()

    if distribution_type == "degenerate":
        probs = torch.full((vocab_size,), 1e-8)
        probs[0] = 0.99
        return probs / probs.sum()

    if distribution_type == "near_uniform":
        probs = torch.ones(vocab_size) / vocab_size + torch.randn(vocab_size) * 0.001
        probs = torch.clamp(probs, min=1e-10)
        return probs / probs.sum()

    # Default: Dirichlet-sampled random distribution
    alpha = np.random.uniform(0.1, 10.0, size=vocab_size)
    probs = np.random.dirichlet(alpha)
    return torch.tensor(probs, dtype=torch.float32)


@pytest.fixture
def vocab_sizes():
    """Common vocabulary sizes for testing."""
    return [100, 1000, 10000, 32000]


@pytest.fixture
def distribution_types():
    """All distribution types for testing."""
    return ["random", "uniform", "peaked", "bimodal", "degenerate", "near_uniform"]


@pytest.fixture
def temperatures():
    """Temperature values for adaptation testing."""
    return [0.01, 0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]