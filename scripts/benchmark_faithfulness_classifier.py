"""Benchmarks the local faithfulness classifier against `LLMJudge` on real
SciFact validation examples with real gold labels — the comparison the
README's "Faithfulness classifier" section promises: not just that the
classifier works, but how it stacks up against the LLM-judging-LLM
baseline it's meant to replace, on accuracy, agreement, and latency.

Thin orchestration over already-tested library code, like the project's
other benchmark/training scripts — not unit tested itself.

    python scripts/benchmark_faithfulness_classifier.py --llm ollama --llm-model llama3.2:1b

Requires a trained classifier checkpoint (see train_faithfulness_classifier.py)
and a reachable LLM backend for LLMJudge (Ollama by default — zero cost,
runs locally).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from doctor_rounds.adapters.llm import LLM, AnthropicLLM, OllamaLLM, OpenAILLM
from doctor_rounds.classifier.model import LocalFaithfulnessClassifier
from doctor_rounds.classifier.training import compute_classification_metrics
from doctor_rounds.data.scifact import load_examples
from doctor_rounds.metrics.generation import LLMJudge


def _build_llm(name: str, model: str | None, ollama_base_url: str) -> LLM:
    if name == "ollama":
        return OllamaLLM(model=model or "llama3.2:1b", base_url=ollama_base_url)
    if name == "anthropic":
        return AnthropicLLM(model=model or "claude-sonnet-5")
    if name == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini")
    raise ValueError(f"Unsupported --llm: {name!r} (expected ollama, anthropic, or openai)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-dir", default="models/faithfulness-classifier")
    parser.add_argument("--llm", default="ollama", choices=["ollama", "anthropic", "openai"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of validation examples used")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score >= threshold counts as 'supported'")
    parser.add_argument("--out", default="scripts/faithfulness_benchmark_results.json")
    args = parser.parse_args()

    print("Loading real SciFact validation examples...")
    examples = load_examples("validation")
    if args.limit is not None:
        examples = examples[: args.limit]
    print(f"  -> benchmarking on {len(examples)} examples")

    classifier = LocalFaithfulnessClassifier(args.classifier_dir)
    judge = LLMJudge(_build_llm(args.llm, args.llm_model, args.ollama_base_url))

    gold = [int(ex.label) for ex in examples]
    clf_scores: list[float] = []
    clf_latencies: list[float] = []
    llm_scores: list[float] = []
    llm_latencies: list[float] = []

    for i, ex in enumerate(examples):
        t0 = time.perf_counter()
        clf_scores.append(classifier.score_faithfulness(ex.claim, ex.context))
        clf_latencies.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        llm_scores.append(judge.score_faithfulness(ex.claim, ex.context))
        llm_latencies.append(time.perf_counter() - t0)

        if (i + 1) % 25 == 0 or (i + 1) == len(examples):
            print(f"  {i + 1}/{len(examples)}")

    clf_preds = [int(s >= args.threshold) for s in clf_scores]
    llm_preds = [int(s >= args.threshold) for s in llm_scores]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    report = {
        "n_examples": len(examples),
        "threshold": args.threshold,
        "classifier": {
            **compute_classification_metrics(gold, clf_preds),
            "mean_latency_ms": 1000 * mean(clf_latencies),
        },
        "llm_judge": {
            **compute_classification_metrics(gold, llm_preds),
            "mean_latency_ms": 1000 * mean(llm_latencies),
            "llm": args.llm,
            "llm_model": args.llm_model,
        },
        "classifier_vs_llm_judge_agreement": {
            "cohen_kappa": float(cohen_kappa_score(clf_preds, llm_preds)),
            "raw_agreement": sum(int(a == b) for a, b in zip(clf_preds, llm_preds, strict=True)) / len(examples),
        },
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
