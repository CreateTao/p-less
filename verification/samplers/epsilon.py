from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy


class EpsilonSamplingStrategy(SamplingStrategy):
    """Epsilon sampling: keep tokens whose probability >= epsilon.

    Falls back to the top token if all tokens are below epsilon.
    """

    def __init__(self, epsilon: float = 1e-3, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.epsilon = epsilon

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()

        # Mask tokens below epsilon
        mask = probs < self.epsilon
        probs[mask] = 0.0

        candidate_set_size = (probs > 0).sum(dim=-1).item()

        # Fallback: keep top token if empty set
        if candidate_set_size == 0:
            probs[probs.argmax(dim=-1)] = 1.0
            candidate_set_size = 1

        # Renormalize
        probs.div_(probs.sum(dim=-1, keepdim=True))

        next_token = torch.multinomial(probs, num_samples=1)

        self._record_metrics(self.epsilon, candidate_set_size, top_token_prob, start_time)
        return next_token

    @property
    def name(self) -> str:
        return f"epsilon_{self.epsilon}"

    @property
    def params(self) -> dict:
        return {"epsilon": self.epsilon}