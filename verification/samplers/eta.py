from __future__ import annotations

import math
import time

import torch

from verification.samplers.base import SamplingStrategy


class EtaSamplingStrategy(SamplingStrategy):
    """Eta sampling: entropy-based dynamic threshold.

    Threshold = min(exp(-eta), softmax(logits)[argmax(logits)])
    Falls back to the top token if all tokens are below threshold.
    """

    def __init__(self, eta: float = 0.3, temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.eta = eta

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        start_time = time.perf_counter()

        scaled_logits = self._apply_temperature(logits)
        probs = torch.softmax(scaled_logits, dim=-1)

        top_token_prob = probs.max().item()

        # Compute threshold: min(exp(-eta), max_prob)
        threshold = min(math.exp(-self.eta), top_token_prob)

        # Mask tokens below threshold
        mask = probs < threshold
        probs[mask] = 0.0

        candidate_set_size = (probs > 0).sum(dim=-1).item()

        # Fallback: keep top token if empty set
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
        return f"eta_{self.eta}"

    @property
    def params(self) -> dict:
        return {"eta": self.eta}
