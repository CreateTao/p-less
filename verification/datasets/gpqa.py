from __future__ import annotations

import re

from verification.datasets.base import DatasetHandler, DatasetItem


class GPQAHandler(DatasetHandler):
    """GPQA (Google-Proof Q&A) expert-level reasoning dataset handler.

    Uses the Diamond subset for maximum difficulty.
    Multiple-choice format: A/B/C/D with expert-level science questions.
    """

    PROMPT_TEMPLATE = """What is the correct answer to the following question?

{question}

Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Answer:"""

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/GPQA", "gpqa_diamond", split="train")
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        return [
            DatasetItem(
                question_id=i,
                prompt_input=item["question"],
                ground_truth=item["answer"],
                metadata={
                    "choices": [
                        item.get("Choice A", ""),
                        item.get("Choice B", ""),
                        item.get("Choice C", ""),
                        item.get("Choice D", ""),
                    ],
                    "correct_index": self._find_correct_index(item),
                },
            )
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        choices = item.metadata.get("choices", ["", "", "", ""])
        return self.PROMPT_TEMPLATE.format(
            question=item.prompt_input,
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract letter choice (A/B/C/D) from generated text."""
        # Look for "The answer is (X)" or "answer: (X)" patterns
        match = re.search(r"[Aa]nswer\s*(?:is)?\s*[\(:]?\s*([A-D])\b", generated_text)
        if match:
            return match.group(1).upper()

        # Look for standalone letter at the end
        match = re.search(r"\b([A-D])\b\s*$", generated_text.strip())
        if match:
            return match.group(1).upper()

        # Fallback: first occurrence of A/B/C/D in the text
        match = re.search(r"\b([A-D])\b", generated_text)
        if match:
            return match.group(1).upper()

        return ""

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Check if extracted letter matches the correct answer index."""
        correct_index = item.metadata.get("correct_index", -1)
        if correct_index < 0:
            return False

        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        extracted_index = letter_map.get(extracted_answer.upper(), -1)
        return extracted_index == correct_index

    @staticmethod
    def _find_correct_index(item: dict) -> int:
        """Find the index of the correct answer choice."""
        answer = item.get("answer", "")
        choices = [
            item.get("Choice A", ""),
            item.get("Choice B", ""),
            item.get("Choice C", ""),
            item.get("Choice D", ""),
        ]
        for i, choice in enumerate(choices):
            if choice == answer:
                return i
        # Fallback: check letter prefix
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        if answer.upper() in letter_map:
            return letter_map[answer.upper()]
        return -1

    @property
    def dataset_name(self) -> str:
        return "gpqa"

    @property
    def metric_name(self) -> str:
        return "accuracy"
