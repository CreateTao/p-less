from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy


class TemperatureOnlyStrategy(SamplingStrategy):
    """Pure temperature sampling with no truncation."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()
        vocab_size = probs.size(-1)

        next_token = torch.multinomial(probs, num_samples=1)

        self._record_metrics(0.0, vocab_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return f"temperature_{self._temperature}"

    @property
    def params(self) -> dict:
        return {"temperature": self._temperature}