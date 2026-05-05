"""Step 1 Test: Verify p-less threshold adapts smoothly with temperature."""

import sys
import pytest
import torch

sys.path.insert(0, ".")

from verification.tests.conftest import generate_random_distribution


TEMPERATURES = [0.01, 0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]


class TestTemperatureAdaptation:

    def test_threshold_decreases_with_increasing_temperature(self):
        """p-less threshold decreases as temperature increases (distribution flattens)."""
        vocab_size = 1000
        logits = torch.randn(vocab_size)  # Fixed logits

        thresholds = []
        for temp in TEMPERATURES:
            probs = torch.softmax(logits / temp, dim=-1)
            p = probs.square().sum(dim=-1).item()
            thresholds.append(p)

        # Threshold should generally decrease as temperature increases
        # (higher temperature -> flatter distribution -> lower p)
        for i in range(len(thresholds) - 1):
            if TEMPERATURES[i] < TEMPERATURES[i + 1]:
                assert thresholds[i] >= thresholds[i + 1] - 1e-10, (
                    f"Threshold not decreasing: T={TEMPERATURES[i]} -> p={thresholds[i]}, "
                    f"T={TEMPERATURES[i+1]} -> p={thresholds[i+1]}"
                )

    def test_threshold_approaches_max_prob_at_low_temperature(self):
        """At T→0, p-less threshold approaches max(probs) (greedy behavior)."""
        vocab_size = 1000
        logits = torch.randn(vocab_size)

        probs_low_t = torch.softmax(logits / 0.01, dim=-1)
        p_low_t = probs_low_t.square().sum(dim=-1).item()
        max_prob_low_t = probs_low_t.max().item()

        # At very low temperature, distribution is sharply peaked
        # p ≈ max_prob^2 ≈ max_prob (since max_prob ≈ 1)
        assert abs(p_low_t - max_prob_low_t) < 0.1, (
            f"Low temperature threshold not near max: p={p_low_t}, max={max_prob_low_t}"
        )

    def test_threshold_approaches_1v_at_high_temperature(self):
        """At T→∞, p-less threshold approaches 1/V (uniform behavior)."""
        vocab_size = 1000
        logits = torch.randn(vocab_size)

        probs_high_t = torch.softmax(logits / 100.0, dim=-1)
        p_high_t = probs_high_t.square().sum(dim=-1).item()
        expected_uniform = 1.0 / vocab_size

        # At very high temperature, distribution is nearly uniform
        # p ≈ 1/V
        assert abs(p_high_t - expected_uniform) < 0.01, (
            f"High temperature threshold not near 1/V: p={p_high_t}, 1/V={expected_uniform}"
        )

    def test_threshold_smooth_no_discontinuity(self):
        """Threshold curve is smooth (no sudden jumps between consecutive temperatures)."""
        vocab_size = 1000
        logits = torch.randn(vocab_size)

        thresholds = []
        fine_temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for temp in fine_temps:
            probs = torch.softmax(logits / temp, dim=-1)
            p = probs.square().sum(dim=-1).item()
            thresholds.append(p)

        # Check that consecutive differences are small (no jumps)
        for i in range(len(thresholds) - 1):
            diff = abs(thresholds[i] - thresholds[i + 1])
            assert diff < 0.5, (
                f"Discontinuity in threshold curve: diff={diff}, "
                f"between T={fine_temps[i]} and T={fine_temps[i+1]}"
            )

    def test_p_less_norm_threshold_zero_at_uniform(self):
        """p-less-norm threshold is exactly 0 for uniform distribution (regardless of temperature)."""
        vocab_size = 1000
        # Uniform distribution: probs = 1/V, which is the same at any temperature
        # if logits are already uniform
        probs = torch.ones(vocab_size) / vocab_size
        v = vocab_size
        p_norm = ((v * probs.square().sum(dim=-1)) - 1.0) / (v - 1.0)

        assert abs(p_norm.item()) < 1e-8, (
            f"Uniform distribution should yield p_norm=0: got {p_norm.item()}"
        )