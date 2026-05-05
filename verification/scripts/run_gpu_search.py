"""GPU Step 6: Hyperparameter grid search for baseline methods.

Finds optimal hyperparameters for each baseline method on a 20% subset,
so that the GPU full evaluation (Step 7) compares p-less against the BEST
possible baselines — proving non-degradation even when baselines are tuned.
"""

import argparse
import json
import os
import sys
import time

import torch
from itertools import product

sys.path.insert(0, ".")

from verification.samplers.registry import StrategyRegistry
from verification.datasets.registry import DatasetRegistry
from verification.generation.engine import GenerationEngine
from verification.storage.schema import GridSearchResult
from verification.storage.io import save_results


# Default grid for each baseline method
DEFAULT_GRIDS = {
    "top_p": {"top_p": [0.8, 0.9, 0.95, 0.99], "temperature": [0.3, 0.5, 0.7, 1.0]},
    "top_k": {"top_k": [10, 20, 40, 100], "temperature": [0.3, 0.5, 0.7, 1.0]},
    "min_p": {"min_p": [0.01, 0.05, 0.1, 0.2], "temperature": [0.3, 0.5, 0.7, 1.0]},
    "epsilon": {"epsilon": [0.001, 0.005, 0.01, 0.05], "temperature": [0.3, 0.5, 0.7, 1.0]},
    "eta": {"eta": [0.1, 0.3, 0.5, 0.7], "temperature": [0.3, 0.5, 0.7, 1.0]},
}


def run_grid_search(
    model, tokenizer, method_name: str, param_grid: dict,
    dataset_handler, subset_fraction: float,
    max_tokens: int, seed: int, results_dir: str, model_name: str,
) -> GridSearchResult:
    """Run grid search for a single method on a dataset subset."""
    items = dataset_handler.load(subset_fraction=subset_fraction)
    print(f"  Grid search on {len(items)} samples ({subset_fraction:.0%} subset)")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_configs = [dict(zip(param_names, vals)) for vals in product(*param_values)]

    print(f"  Total configurations to evaluate: {len(all_configs)}")

    best_accuracy = -1.0
    best_params = {}
    all_evaluated = []

    for i, config in enumerate(all_configs):
        # Extract temperature from config, default to 1.0
        temperature = config.pop("temperature", 1.0)

        strategy = StrategyRegistry.create(
            name=method_name,
            params=config,
            temperature=temperature,
        )

        engine = GenerationEngine(
            model=model,
            tokenizer=tokenizer,
            strategy=strategy,
            max_tokens=max_tokens,
            seed=seed,
            record_metrics=False,
        )

        correct_count = 0
        for item in items:
            prompt = dataset_handler.format_prompt(item, model_name)
            gen_result = engine.generate(prompt)
            extracted = dataset_handler.extract_answer(gen_result.generated_text, item)
            if dataset_handler.evaluate(extracted, item.ground_truth, item):
                correct_count += 1

        accuracy = correct_count / len(items)
        eval_record = {**config, "temperature": temperature, "accuracy": accuracy}
        all_evaluated.append(eval_record)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {**config, "temperature": temperature}

        print(f"    [{i+1}/{len(all_configs)}] {config} T={temperature:.1f} -> {accuracy:.4f}")

        # Restore temperature in config for next iteration
        config["temperature"] = temperature

    result = GridSearchResult(
        method=method_name,
        dataset=dataset_handler.dataset_name,
        model=model_name,
        best_params=best_params,
        best_accuracy=best_accuracy,
        all_configs_evaluated=all_evaluated,
        search_subset_fraction=subset_fraction,
    )

    # Save
    subdir = os.path.join(results_dir, dataset_handler.dataset_name, model_name)
    filename = f"grid_search_{method_name}.json"
    save_results(result, subdir, filename)

    print(f"  Best {method_name}: {best_params} -> {best_accuracy:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="GPU hyperparameter grid search")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval"],
                        help="Datasets to search on")
    parser.add_argument("--methods", nargs="+",
                        default=["top_p", "top_k", "min_p", "epsilon", "eta"],
                        help="Baseline methods to grid search")
    parser.add_argument("--subset-fraction", type=float, default=0.2,
                        help="Fraction of dataset to use for search")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="verification/outputs/results/gpu_search")
    parser.add_argument("--hf-token", default="")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== GPU Grid Search ===")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16,
    )

    all_results = {}

    for dataset_name in args.datasets:
        dataset_handler = DatasetRegistry.create(dataset_name)
        print(f"\n--- Dataset: {dataset_name} ---")

        for method_name in args.methods:
            param_grid = DEFAULT_GRIDS.get(method_name)
            if param_grid is None:
                print(f"  Skipping {method_name}: no grid defined")
                continue

            result = run_grid_search(
                model=model,
                tokenizer=tokenizer,
                method_name=method_name,
                param_grid=param_grid,
                dataset_handler=dataset_handler,
                subset_fraction=args.subset_fraction,
                max_tokens=args.max_tokens,
                seed=args.seed,
                results_dir=args.results_dir,
                model_name=args.model,
            )
            all_results[f"{dataset_name}/{method_name}"] = result

    # Print summary
    print("\n=== Grid Search Summary ===")
    for key, result in all_results.items():
        print(f"  {key}: best={result.best_accuracy:.4f} params={result.best_params}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
