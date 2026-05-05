from __future__ import annotations

import sys
import time

import torch

from verification.samplers.base import SamplingStrategy

# Import original p-less functions from project root
sys.path.insert(0, ".")

from p_less_samplers import p_less_decode, p_less_norm_decode


class PLessStrategy(SamplingStrategy):
    """p-less sampling strategy wrapper.

    Handles in-place mutation by cloning probs before calling p_less_decode.
    """

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()
        threshold = probs.square().sum(dim=-1).item()

        # Clone to protect against in-place mutation
        probs_clone = probs.clone()
        next_token = p_less_decode(probs_clone)

        candidate_set_size = (probs_clone > 0).sum(dim=-1).item()

        self._record_metrics(threshold, candidate_set_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return "p_less"

    @property
    def params(self) -> dict:
        return {}


class PLessNormStrategy(SamplingStrategy):
    """p-less-norm sampling strategy wrapper.

    Handles in-place mutation by cloning probs before calling p_less_norm_decode.
    """

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()
        v = probs.size(-1)
        threshold = ((v * probs.square().sum(dim=-1)) - 1.0) / (v - 1.0)
        threshold_val = threshold.item()

        # Clone to protect against in-place mutation
        probs_clone = probs.clone()
        next_token = p_less_norm_decode(probs_clone)

        candidate_set_size = (probs_clone > 0).sum(dim=-1).item()

        self._record_metrics(threshold_val, candidate_set_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return "p_less_norm"

    @property
    def params(self) -> dict:
        return {}