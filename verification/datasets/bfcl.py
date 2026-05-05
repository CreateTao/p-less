from __future__ import annotations

import json
import re

from verification.datasets.base import DatasetHandler, DatasetItem


class BFCLHandler(DatasetHandler):
    """BFCL v3 function calling dataset handler.

    Evaluates both format compliance (valid JSON) and accuracy (correct function call).
    """

    TEMPLATE = """You are a helpful assistant with access to the following functions. Use them if needed -

{function_definitions}

{query}

Please provide the function call(s) that best answer the query. Format your response as a JSON array of function calls."""

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        """Load BFCL dataset from local JSON or HuggingFace."""
        try:
            from datasets import load_dataset
            ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="simple")
        except Exception:
            # Fallback: try local file
            import os
            local_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bfcl_simple.json")
            if os.path.exists(local_path):
                with open(local_path, "r") as f:
                    data = json.load(f)
            else:
                raise FileNotFoundError("BFCL dataset not found. Download from https://github.com/gorilla-llm/Berkeley-Function-Calling-Leaderboard")

        if subset_fraction < 1.0:
            import random
            random.seed(42)
            indices = random.sample(range(len(data)), int(len(data) * subset_fraction))
            data = [data[i] for i in indices]

        return [
            DatasetItem(
                question_id=i,
                prompt_input=item.get("question", ""),
                ground_truth=json.dumps(item.get("expected", [])),
                metadata={
                    "function_definitions": json.dumps(item.get("functions", [])),
                    "test_category": item.get("test_category", "simple"),
                },
            )
            for i, item in enumerate(data)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        return self.TEMPLATE.format(
            function_definitions=item.metadata.get("function_definitions", ""),
            query=item.prompt_input,
        )

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract JSON function call(s) from generated text."""
        # Try direct JSON parse
        try:
            json.loads(generated_text)
            return generated_text.strip()
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n(.*?)```", generated_text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1).strip()
            except json.JSONDecodeError:
                pass

        # Try finding any JSON-like structure
        match = re.search(r"\[.*\]", generated_text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(0))
                return match.group(0).strip()
            except json.JSONDecodeError:
                pass

        return generated_text.strip()

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Two-stage evaluation: format compliance + accuracy."""
        # Stage 1: format compliance
        try:
            parsed = json.loads(extracted_answer)
        except json.JSONDecodeError:
            return False

        # Stage 2: accuracy - compare function name and parameters
        try:
            expected = json.loads(ground_truth)
            return _compare_function_calls(parsed, expected)
        except json.JSONDecodeError:
            return False

    @property
    def dataset_name(self) -> str:
        return "bfcl"

    @property
    def metric_name(self) -> str:
        return "accuracy_and_format_rate"


def _compare_function_calls(predicted: list | dict, expected: list | dict) -> bool:
    """Compare predicted and expected function calls."""
    # Normalize to list format
    if isinstance(predicted, dict):
        predicted = [predicted]
    if isinstance(expected, dict):
        expected = [expected]

    if len(predicted) != len(expected):
        return False

    for pred, exp in zip(predicted, expected):
        if pred.get("name") != exp.get("name"):
            return False
        if pred.get("arguments", {}) != exp.get("arguments", {}):
            return False

    return True