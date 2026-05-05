from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy


class MinPStrategy(SamplingStrategy):
    """Min-p sampling: keep tokens whose probability >= min_p * max(probability)."""

    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.min_p = min_p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()
        threshold = self.min_p * top_token_prob

        # Mask tokens below threshold
        mask = probs < threshold
        probs[mask] = 0.0

        candidate_set_size = (probs > 0).sum(dim=-1).item()

        # Fallback for empty set
        if candidate_set_size == 0:
            probs[probs.argmax(dim=-1)] = 1.0
            candidate_set_size = 1

        # Renormalize
        probs.div_(probs.sum(dim=-1, keepdim=True))

        next_token = torch.multinomial(probs, num_samples=1)

        self._record_metrics(threshold, candidate_set_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return f"min_p_{self.min_p}"

    @property
    def params(self) -> dict:
        return {"min_p": self.min_p}