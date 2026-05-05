from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy


class TopPStrategy(SamplingStrategy):
    """Nucleus (top-p) sampling: keep tokens whose cumulative probability >= top_p."""

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.top_p = top_p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()

        # Sort descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # Keep at least one token to avoid empty set
        sorted_mask = cumulative_probs - sorted_probs > self.top_p
        sorted_probs[sorted_mask] = 0.0

        # Count candidates
        candidate_set_size = (sorted_probs > 0).sum(dim=-1).item()

        # Renormalize
        if sorted_probs.sum() > 0:
            sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        else:
            # Fallback: keep only the top token (edge case defense)
            sorted_probs[0] = 1.0
            candidate_set_size = 1

        # Sample
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(dim=-1, index=next_token)

        self._record_metrics(self.top_p, candidate_set_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return f"top_p_{self.top_p}"

    @property
    def params(self) -> dict:
        return {"top_p": self.top_p}