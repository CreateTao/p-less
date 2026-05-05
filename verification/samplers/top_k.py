from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy


class TopKStrategy(SamplingStrategy):
    """Top-k sampling: keep only the k highest-probability tokens."""

    def __init__(self, top_k: int = 40, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.top_k = top_k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)

        # Mask all tokens except top_k
        if self.top_k > 0 and self.top_k < logits.size(-1):
            top_k_logits, top_k_indices = torch.topk(scaled_logits, min(self.top_k, logits.size(-1)))
            probs = torch.zeros_like(scaled_logits)
            probs.scatter_(dim=-1, index=top_k_indices, src=torch.softmax(top_k_logits, dim=-1))
        else:
            # top_k=-1 or top_k >= vocab_size: no truncation
            probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()
        candidate_set_size = (probs > 0).sum(dim=-1).item()

        next_token = torch.multinomial(probs, num_samples=1)

        self._record_metrics(float(self.top_k), candidate_set_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return f"top_k_{self.top_k}"

    @property
    def params(self) -> dict:
        return {"top_k": self.top_k}