from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy


class GreedyStrategy(SamplingStrategy):
    """Greedy decoding: always select the highest-probability token."""

    def __init__(self):
        super().__init__(temperature=1.0)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()
        next_token = logits.argmax(dim=-1, keepdim=True)
        self._record_metrics(
            threshold_value=0.0,
            candidate_set_size=1,
            top_token_prob=torch.softmax(logits, dim=-1).max().item(),
            start_time=start_time,
        )
        return next_token

    @property
    def name(self) -> str:
        return "greedy"

    @property
    def params(self) -> dict:
        return {}