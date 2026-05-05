from __future__ import annotations

import time

import torch

from verification.samplers.base import SamplingStrategy
from verification.generation.result import GenerationResult


class GenerationEngine:
    """Generalized autoregressive generation engine.

    Replaces the notebook's manual loop with a strategy-agnostic engine that:
    - Accepts any SamplingStrategy
    - Manages KV-cache for efficiency
    - Records per-step SamplingMetrics
    - Handles seed management for reproducibility
    - Handles EOS detection and max-token limits
    """

    def __init__(
        self,
        model,
        tokenizer,
        strategy: SamplingStrategy,
        max_tokens: int = 512,
        seed: int = 42,
        record_metrics: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.seed = seed
        self.record_metrics = record_metrics

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a single completion from a prompt."""
        torch.manual_seed(self.seed)

        encodings = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        past_key_values = None
        generated_token_ids = []
        per_step_metrics = []
        strategy = self.strategy
        strategy.reset_metrics()

        total_start = time.perf_counter()

        with torch.no_grad():
            for _ in range(self.max_tokens):
                output = self.model(
                    input_ids=encodings,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                )

                logits = output.logits[0, -1]
                next_token = strategy.sample(logits)
                token_id = next_token.item()
                generated_token_ids.append(token_id)

                if self.record_metrics:
                    strategy.accumulate_metrics()

                if token_id == self.tokenizer.eos_token_id:
                    break

                past_key_values = output.past_key_values
                encodings = next_token.unsqueeze(0)

        total_time = time.perf_counter() - total_start
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return GenerationResult(
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            num_tokens=len(generated_token_ids),
            per_step_metrics=strategy.get_accumulated_metrics() if self.record_metrics else None,
            generation_time_s=total_time,
            seed=self.seed,
        )

    def generate_k_samples(self, prompt: str, k: int) -> list[GenerationResult]:
        """Generate k independent samples for Pass@k evaluation.

        Each sample uses seed = base_seed + sample_index.
        """
        results = []
        for i in range(k):
            engine = GenerationEngine(
                model=self.model,
                tokenizer=self.tokenizer,
                strategy=self.strategy,
                max_tokens=self.max_tokens,
                seed=self.seed + i,
                record_metrics=self.record_metrics,
            )
            results.append(engine.generate(prompt))
        return results