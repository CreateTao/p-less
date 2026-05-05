from __future__ import annotations

import re

from verification.datasets.base import DatasetHandler, DatasetItem


class GSM8KHandler(DatasetHandler):
    """GSM8K math reasoning dataset handler."""

    # 8-shot prompt template for step-by-step math reasoning
    FEW_SHOT_TEMPLATE = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done there will be 21 trees. How many trees did the grove workers plant today?
Answer: We start with 15 trees and end with 21. So the workers planted 21 - 15 = 6 trees. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are 3 cars and 2 more arrive, so 3 + 2 = 5 cars. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Leah had 32 and her sister had 42, total = 32 + 42 = 74. They ate 35, so 74 - 35 = 39. The answer is 39.

Question: There are 12 girls and 4 boys in a class. Each girl has 2 stickers and each boy has 3 stickers. How many stickers do all the children have in total?
Answer: Girls: 12 × 2 = 24 stickers. Boys: 4 × 3 = 12 stickers. Total = 24 + 12 = 36. The answer is 36.

Question: A baker made 125 cupcakes. He sold 85 of them. Then he made 35 more cupcakes. How many cupcakes does the baker have now?
Answer: 125 - 85 = 40 cupcakes left. Then 40 + 35 = 75 cupcakes. The answer is 75.

Question: A rectangle has a length of 8 cm and a width of 5 cm. What is its area?
Answer: Area = length × width = 8 × 5 = 40 sq cm. The answer is 40.

Question: There are 15 students in a class. 3 students are absent. How many students are present?
Answer: 15 - 3 = 12 students. The answer is 12.

Question: {question}
Answer: """

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        return [
            DatasetItem(
                question_id=i,
                prompt_input=item["question"],
                ground_truth=self._extract_ground_truth(item["answer"]),
                metadata={"full_answer": item["answer"]},
            )
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        return self.FEW_SHOT_TEMPLATE.replace("{question}", item.prompt_input)

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract the final numeric answer from generated text."""
        # Look for "The answer is [NUMBER]" pattern
        match = re.search(r"The answer is\s*([\d]+\.?\d*)", generated_text)
        if match:
            return match.group(1)

        # Fallback: last number in the text
        numbers = re.findall(r"[\d]+\.?\d*", generated_text)
        if numbers:
            return numbers[-1]

        return ""

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Exact numeric match with float tolerance."""
        try:
            extracted = float(extracted_answer)
            expected = float(ground_truth)
            return abs(extracted - expected) < 1e-3
        except (ValueError, TypeError):
            return extracted_answer.strip() == ground_truth.strip()

    @staticmethod
    def _extract_ground_truth(answer_text: str) -> str:
        """Extract numeric answer from GSM8K answer field (e.g. '#### 42')."""
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip()
        # Fallback: last number
        numbers = re.findall(r"[\d]+\.?\d*", answer_text)
        return numbers[-1] if numbers else ""

    @property
    def dataset_name(self) -> str:
        return "gsm8k"

    @property
    def metric_name(self) -> str:
        return "exact_match_accuracy"