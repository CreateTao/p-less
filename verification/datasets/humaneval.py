from __future__ import annotations

import json
import multiprocessing
import os
import re
import tempfile
import textwrap
import time

from verification.datasets.base import DatasetHandler, DatasetItem


class HumanEvalHandler(DatasetHandler):
    """HumanEval code generation dataset handler.

    Uses sandboxed subprocess execution for safety.
    """

    TIMEOUT_SECONDS = 10

    def load(self, subset_fraction: float = 1.0) -> list[DatasetItem]:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        if subset_fraction < 1.0:
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * subset_fraction)))
        return [
            DatasetItem(
                question_id=i,
                prompt_input=item["prompt"],
                ground_truth=item["canonical_solution"],
                metadata={
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                    "task_id": item["task_id"],
                },
            )
            for i, item in enumerate(ds)
        ]

    def format_prompt(self, item: DatasetItem, model_name: str) -> str:
        """Format as code completion prompt."""
        return item.prompt_input

    def extract_answer(self, generated_text: str, item: DatasetItem) -> str:
        """Extract code from generated text."""
        # Try to extract code block
        match = re.search(r"```(?:python)?\s*\n(.*?)```", generated_text, re.DOTALL)
        if match:
            return match.group(1)

        # Fallback: take entire generated text as code
        return generated_text

    def evaluate(self, extracted_answer: str, ground_truth: str, item: DatasetItem) -> bool:
        """Execute code in sandboxed subprocess and run unit tests."""
        entry_point = item.metadata["entry_point"]
        test_code = item.metadata["test"]
        prompt = item.prompt_input
        completion = extracted_answer

        # Combine prompt + completion + test
        full_code = prompt + completion + "\n" + test_code + "\ncheck()"

        try:
            result = _run_code_in_subprocess(full_code, self.TIMEOUT_SECONDS)
            return result is None  # None means no exception, i.e. all tests passed
        except Exception:
            return False

    @property
    def dataset_name(self) -> str:
        return "humaneval"

    @property
    def metric_name(self) -> str:
        return "pass@1"


def _run_code_in_subprocess(code: str, timeout: int) -> str | None:
    """Run code in a subprocess with timeout. Returns error message or None if success."""
    def _worker(code_str, result_queue):
        try:
            exec(code_str, {})
            result_queue.put(None)
        except Exception as e:
            result_queue.put(str(e))

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker, args=(code, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "timeout"

    if not result_queue.empty():
        return result_queue.get()

    return None