"""Entry point: CPU Steps 2-5 - Run accuracy benchmark on CPU."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")

from verification.config.loader import load_config
from verification.samplers.registry import StrategyRegistry
from verification.datasets.registry import DatasetRegistry
from verification.generation.engine import GenerationEngine
from verification.storage.schema import SampleResult
from verification.storage.io import save_sample_result, save_aggregated_result


def run_single_benchmark(
    model, tokenizer, strategy, dataset_handler, config,
    results_dir: str,
) -> list[SampleResult]:
    """Run benchmark for a single (model, strategy, dataset) combination."""
    engine = GenerationEngine(
        model=model,
        tokenizer=tokenizer,
        strategy=strategy,
        max_tokens=config.max_tokens,
        seed=config.seed,
        record_metrics=True,
    )

    items = dataset_handler.load()
    if config.dataset.num_samples > 0:
        items = items[:config.dataset.num_samples]
    results = []
    model_name = config.models[0].name if config.models else "unknown"

    for item in items:
        # Skip if result already exists (resume support)
        result_path = os.path.join(
            results_dir, dataset_handler.dataset_name,
            model_name,
            strategy.name, f"t{strategy.temperature}",
            f"q_{item.question_id:04d}.json",
        )
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results.append(SampleResult(**json.load(f)))
            continue

        prompt = dataset_handler.format_prompt(item, model_name)
        gen_result = engine.generate(prompt)

        extracted_answer = dataset_handler.extract_answer(gen_result.generated_text, item)
        is_correct = dataset_handler.evaluate(extracted_answer, item.ground_truth, item)

        # Compute average metrics
        metrics = gen_result.per_step_metrics
        avg_candidate_set_size = sum(m.candidate_set_size for m in metrics) / len(metrics) if metrics else 0
        avg_threshold_value = sum(m.threshold_value for m in metrics) / len(metrics) if metrics else 0
        avg_sampling_time_ms = sum(m.sampling_time_ms for m in metrics) / len(metrics) if metrics else 0

        sample_result = SampleResult(
            experiment_id=config.experiment_id,
            model_name=model_name,
            method_name=strategy.name,
            temperature=strategy.temperature,
            method_params=strategy.params,
            dataset_name=dataset_handler.dataset_name,
            question_id=item.question_id,
            prompt=prompt,
            generated_text=gen_result.generated_text,
            generated_tokens=gen_result.num_tokens,
            correct=is_correct,
            extracted_answer=extracted_answer,
            ground_truth_answer=item.ground_truth,
            avg_candidate_set_size=avg_candidate_set_size,
            avg_threshold_value=avg_threshold_value,
            avg_sampling_time_ms=avg_sampling_time_ms,
            generation_time_s=gen_result.generation_time_s,
            seed=config.seed,
        )

        save_sample_result(sample_result, results_dir)
        results.append(sample_result)

    return results


def main():
    parser = argparse.ArgumentParser(description="CPU accuracy benchmark")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--results-dir", default="verification/outputs/results/cpu")
    parser.add_argument("--hf-token", default="", help="HuggingFace access token")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"=== CPU Benchmark: {config.experiment_id} ===")
    print(f"Dataset: {config.dataset.name}")
    print(f"Models: {[m.name for m in config.models]}")

    # Load dataset handler
    dataset_handler = DatasetRegistry.create(config.dataset.name)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for model_cfg in config.models:
        print(f"\nLoading model: {model_cfg.hf_id} on {model_cfg.device}")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            use_cache=True,
            device_map=model_cfg.device,
            dtype=torch.float32,
        )

        # Run each method
        for method_cfg in config.methods:
            print(f"\n  Method: {method_cfg.name} (T={method_cfg.temperature or 1.0})")
            strategy = StrategyRegistry.create(
                name=method_cfg.strategy_class,
                params=method_cfg.params,
                temperature=method_cfg.temperature or 1.0,
            )

            results = run_single_benchmark(
                model, tokenizer, strategy, dataset_handler, config, args.results_dir,
            )

            # Compute and save aggregated result
            accuracy = sum(r.correct for r in results) / len(results)
            avg_tokens = sum(r.generated_tokens for r in results) / len(results)
            avg_candidate_set = sum(r.avg_candidate_set_size for r in results) / len(results)

            from verification.stats.bootstrap import bootstrap_ci
            acc, ci_lower, ci_upper = bootstrap_ci([r.correct for r in results])

            from verification.storage.schema import AggregatedResult
            agg = AggregatedResult(
                experiment_id=config.experiment_id,
                model_name=model_cfg.name,
                method_name=strategy.name,
                temperature=strategy.temperature,
                method_params=strategy.params,
                dataset_name=dataset_handler.dataset_name,
                num_samples=len(results),
                accuracy=acc,
                bootstrap_ci_lower=ci_lower,
                bootstrap_ci_upper=ci_upper,
                avg_generated_tokens=avg_tokens,
                avg_candidate_set_size=avg_candidate_set,
                seed=config.seed,
            )
            save_aggregated_result(agg, args.results_dir)

            print(f"    Accuracy: {accuracy:.4f} ({len(results)} samples)")

        # Free GPU/CPU memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()