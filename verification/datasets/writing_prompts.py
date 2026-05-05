from __future__ import annotations

import re

from verification.datasets.base import DatasetHandler, DatasetItem


class WritingPromptsHandler(DatasetHandler):
    """Writing Prompts creative writing dataset handler.

    Evaluates diversity and quality of generated creative text.
    Uses MAUVE score for distributional similarity and Distinct-n for diversity.
    """

    PROMPT_TEMPLATE = """Write a creative story based on the following prompt:

{prompt}

Story:"""

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("reddit", "writing_prompts", split="test", trust_remote_code=True)
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        items = []
        for i, item in enumerate(ds):
            prompt_text = item.get("prompt", item.get("title", ""))
            if not prompt_text:
                continue
            items.append(DatasetItem(
                question_id=i,
                prompt_input=prompt_text,
                ground_truth=item.get("normalizedBody", item.get("selftext", "")),
                metadata={"subreddit": item.get("subreddit", "WritingPrompts")},
            ))
        return items

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        return self.PROMPT_TEMPLATE.format(prompt=item.prompt_input)

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Return the full generated text as the 'answer' for diversity metrics."""
        # Remove the prompt prefix if echoed
        text = generated_text.strip()
        if text.startswith("Story:"):
            text = text[len("Story:"):].strip()
        return text

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Writing prompts uses diversity metrics, not correctness.

        For compatibility with the evaluation framework, we compute
        Distinct-2 as a proxy score and threshold at 0.5.
        The real evaluation happens in run_analysis.py via MAUVE.
        """
        distinct2 = self._distinct_n(extracted_answer, n=2)
        return distinct2 >= 0.5

    @staticmethod
    def _distinct_n(text: str, n: int = 2) -> float:
        """Compute Distinct-n metric: unique n-grams / total n-grams."""
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    @property
    def dataset_name(self) -> str:
        return "writing_prompts"

    @property
    def metric_name(self) -> str:
        return "diversity_score"
