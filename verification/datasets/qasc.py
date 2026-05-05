from __future__ import annotations

import re

from verification.datasets.base import DatasetHandler, DatasetItem


class QASCHandler(DatasetHandler):
    """QASC science reasoning dataset handler (8-way multiple choice)."""

    TEMPLATE = """Question: {question}
Choices:
{choices}
Answer: """

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("qasc", split="test")
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        return [
            DatasetItem(
                question_id=i,
                prompt_input=item["question"],
                ground_truth=item["answerKey"],
                metadata={
                    "choices": {
                        "labels": item["choices"]["label"],
                        "texts": item["choices"]["text"],
                    },
                },
            )
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        choices = item.metadata["choices"]
        choice_lines = [
            f"{label}) {text}"
            for label, text in zip(choices["labels"], choices["texts"])
        ]
        return self.TEMPLATE.format(
            question=item.prompt_input,
            choices="\n".join(choice_lines),
        )

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract the choice letter (A-H) from generated text."""
        match = re.search(r"[A-H]", generated_text)
        return match.group(0) if match else ""

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        return extracted_answer.upper() == ground_truth.upper()

    @property
    def dataset_name(self) -> str:
        return "qasc"

    @property
    def metric_name(self) -> str:
        return "accuracy"