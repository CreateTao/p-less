"""GPU Step 7: Full evaluation with TOST equivalence testing.

Compares p-less/p-less-norm against optimally-tuned baselines (from grid search)
on the FULL dataset. Uses TOST equivalence test to prove non-degradation:
H0: |p_less - baseline| >= delta  vs  H1: |p_less - baseline| < delta

If TOST passes, p-less is statistically equivalent to the best baseline (within delta).
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")

from verification.samplers.registry import StrategyRegistry
from verification.datasets.registry import DatasetRegistry
from verification.generation.engine import GenerationEngine
from verification.storage.schema import SampleResult, AggregatedResult, TOSTResult
from verification.storage.io import (
    save_sample_result, save_aggregated_result, save_results, load_results,
)
from verification.stats.tost import tost_equivalence_test
from verification.stats.bootstrap import bootstrap_ci
from verification.stats.paired_tests import paired_t_test


def load_best_params(search_dir: str, dataset: str, model: str, method: str) -> dict | None:
    """Load best parameters from grid search results."""
    filepath = os.path.join(search_dir, dataset, model, f"grid_search_{method}.json")
    if not os.path.exists(filepath):
        return None
    data = load_results(filepath)
    return data.get("best_params")


def run_full_evaluation(
    model, tokenizer, strategy, dataset_handler,
    max_tokens: int, seed: int, results_dir: str, model_name: str,
) -> list[SampleResult]:
    """Run full evaluation for a single (model, strategy, dataset)."""
    engine = GenerationEngine(
        model=model,
        tokenizer=tokenizer,
        strategy=strategy,
        max_tokens=max_tokens,
        seed=seed,
        record_metrics=True,
    )

    items = dataset_handler.load(subset_fraction=1.0)
    results = []

    for item in items:
        # Resume support
        result_path = os.path.join(
            results_dir, dataset_handler.dataset_name, model_name,
            strategy.name, f"t{strategy.temperature}",
            f"q_{item.question_id:04d}.json",
        )
        if os.path.exists(result_path):
            continue

        prompt = dataset_handler.format_prompt(item, model_name)
        gen_result = engine.generate(prompt)
        extracted = dataset_handler.extract_answer(gen_result.generated_text, item)
        is_correct = dataset_handler.evaluate(extracted, item.ground_truth, item)

        metrics = gen_result.per_step_metrics
        avg_cs = sum(m.candidate_set_size for m in metrics) / len(metrics) if metrics else 0
        avg_th = sum(m.threshold_value for m in metrics) / len(metrics) if metrics else 0
        avg_st = sum(m.sampling_time_ms for m in metrics) / len(metrics) if metrics else 0

        sample = SampleResult(
            experiment_id="gpu_full_eval",
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
            extracted_answer=extracted,
            ground_truth_answer=item.ground_truth,
            avg_candidate_set_size=avg_cs,
            avg_threshold_value=avg_th,
            avg_sampling_time_ms=avg_st,
            generation_time_s=gen_result.generation_time_s,
            seed=seed,
        )
        save_sample_result(sample, results_dir)
        results.append(sample)

    return results


def main():
    parser = argparse.ArgumentParser(description="GPU full evaluation with TOST")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval", "bfcl", "gpqa"])
    parser.add_argument("--methods", nargs="+",
                        default=["top_p", "top_k", "min_p", "p_less", "p_less_norm"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.3, 0.7, 1.0])
    parser.add_argument("--delta", type=float, default=0.02,
                        help="TOST equivalence margin (default 2%%)")
    parser.add_argument("--search-dir", default="verification/outputs/results/gpu_search",
                        help="Directory with grid search results for best params")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="verification/outputs/results/gpu_eval")
    parser.add_argument("--hf-token", default="")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== GPU Full Evaluation (TOST δ={args.delta}) ===")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"Temperatures: {args.temperatures}")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16,
    )

    # Collect all per-sample correctness for TOST
    # Key: (dataset, model, method, temperature) -> list[bool]
    correctness_map = {}

    for dataset_name in args.datasets:
        dataset_handler = DatasetRegistry.create(dataset_name)
        print(f"\n--- Dataset: {dataset_name} ---")

        for method_name in args.methods:
            # Load best params from grid search for baselines
            best_params = None
            if method_name not in ("p_less", "p_less_norm", "greedy"):
                best_params = load_best_params(args.search_dir, dataset_name, args.model, method_name)
                if best_params:
                    print(f"  {method_name}: using best params from grid search: {best_params}")

            temperatures = args.temperatures
            if method_name in ("p_less", "p_less_norm"):
                temperatures = args.temperatures  # Test at multiple temperatures
            elif best_params and "temperature" in best_params:
                temperatures = [best_params.pop("temperature")]  # Use best temp only

            for temp in temperatures:
                params = dict(best_params) if best_params else {}
                strategy = StrategyRegistry.create(
                    name=method_name, params=params, temperature=temp,
                )
                print(f"  Running {strategy.name} at T={temp}...")

                results = run_full_evaluation(
                    model, tokenizer, strategy, dataset_handler,
                    args.max_tokens, args.seed, args.results_dir, args.model,
                )

                # Aggregate
                accuracy = sum(r.correct for r in results) / len(results)
                acc, ci_lo, ci_hi = bootstrap_ci([r.correct for r in results])
                avg_tokens = sum(r.generated_tokens for r in results) / len(results)
                avg_cs = sum(r.avg_candidate_set_size for r in results) / len(results)

                agg = AggregatedResult(
                    experiment_id="gpu_full_eval",
                    model_name=args.model,
                    method_name=strategy.name,
                    temperature=temp,
                    method_params=strategy.params,
                    dataset_name=dataset_handler.dataset_name,
                    num_samples=len(results),
                    accuracy=acc,
                    bootstrap_ci_lower=ci_lo,
                    bootstrap_ci_upper=ci_hi,
                    avg_generated_tokens=avg_tokens,
                    avg_candidate_set_size=avg_cs,
                    seed=args.seed,
                )
                save_aggregated_result(agg, args.results_dir)
                print(f"    Accuracy: {accuracy:.4f} ({len(results)} samples)")

                # Store for TOST
                key = (dataset_name, args.model, strategy.name, temp)
                correctness_map[key] = [r.correct for r in results]

    # Run TOST equivalence tests: p-less vs each baseline
    print("\n=== TOST Equivalence Tests (δ={:.1%}) ===".format(args.delta))
    tost_results = []

    p_less_keys = [k for k in correctness_map if k[2] in ("p_less", "p_less_norm")]
    baseline_keys = [k for k in correctness_map if k[2] not in ("p_less", "p_less_norm")]

    for pk in p_less_keys:
        dataset, model_name, pless_method, pless_temp = pk
        for bk in baseline_keys:
            if bk[0] != dataset or bk[1] != model_name:
                continue
            baseline_method, baseline_temp = bk[2], bk[3]

            tost = tost_equivalence_test(
                results_a=correctness_map[pk],
                results_b=correctness_map[bk],
                delta=args.delta,
                method_a_name=pless_method,
                method_b_name=baseline_method,
                dataset_name=dataset,
                model_name=model_name,
            )

            status = "EQUIVALENT" if tost.is_equivalent else "NOT EQUIVALENT"
            print(f"  {pless_method}(T={pless_temp}) vs {baseline_method}(T={baseline_temp}) "
                  f"on {dataset}: diff={tost.mean_diff:+.4f} [{status}]")

            # Save TOST result
            tost_schema = TOSTResult(
                method_a=pless_method,
                method_b=baseline_method,
                dataset=dataset,
                model=model_name,
                delta=args.delta,
                lower_test_pvalue=tost.lower_test_pvalue,
                upper_test_pvalue=tost.upper_test_pvalue,
                is_equivalent=tost.is_equivalent,
                mean_diff=tost.mean_diff,
                diff_ci_lower=tost.diff_ci_lower,
                diff_ci_upper=tost.diff_ci_upper,
            )
            subdir = os.path.join(args.results_dir, dataset, model_name, "tost")
            filename = f"tost_{pless_method}_vs_{baseline_method}_t{pless_temp}.json"
            save_results(tost_schema, subdir, filename)
            tost_results.append(tost_schema)

    # Summary
    equiv_count = sum(1 for t in tost_results if t.is_equivalent)
    total = len(tost_results)
    print(f"\n  TOST Summary: {equiv_count}/{total} comparisons passed equivalence")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
