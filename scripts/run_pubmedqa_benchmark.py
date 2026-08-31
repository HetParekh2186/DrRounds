"""Runs a real retrieval benchmark against PubMedQA and reports genuine
recall@k / precision@k / MRR / NDCG@k numbers.

This is the concrete demonstration behind the README's data claims: not a
handful of toy documents, but the 1,000 expert-labeled PubMedQA questions
retrieved against a corpus of tens of thousands of real biomedical
passages (the labeled split's own passages plus a large sample from the
artificial split as distractors).

Usage:
    python scripts/run_pubmedqa_benchmark.py [--corpus-size N] [--k K]

Takes a few minutes: most of the time is spent embedding the corpus with
sentence-transformers on CPU. Results are printed and written to
scripts/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from doctor_rounds.adapters.vectorstore import ChromaVectorStore
from doctor_rounds.data.pubmedqa import load_corpus, load_test_cases
from doctor_rounds.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=5000,
        help="Number of pqa_artificial examples to add as distractors (~4 passages each)",
    )
    parser.add_argument("--k", type=int, default=10, help="Retrieve top-k for every metric")
    parser.add_argument(
        "--test-cases", type=int, default=None, help="Limit test cases (default: all 1,000)"
    )
    args = parser.parse_args()

    print(f"Loading {args.test_cases or 'all'} labeled PubMedQA test cases...")
    test_cases = load_test_cases(limit=args.test_cases)
    print(f"  -> {len(test_cases)} test cases")

    print(f"Building corpus: labeled passages + {args.corpus_size} artificial-split examples...")
    t0 = time.time()
    corpus = load_corpus(limit=args.corpus_size, include_labeled=True)
    print(f"  -> {len(corpus)} chunks ({time.time() - t0:.1f}s)")

    print("Embedding and indexing corpus (this is the slow part)...")
    t0 = time.time()
    store = ChromaVectorStore(collection_name="pubmedqa-benchmark")
    store.add(corpus)
    print(f"  -> indexed in {time.time() - t0:.1f}s")

    print(f"Running retrieval for {len(test_cases)} test cases (k={args.k})...")
    t0 = time.time()
    per_case = []
    for tc in test_cases:
        relevant = set(tc.ground_truth_chunk_ids)
        retrieved = store.retrieve(tc.question, k=args.k)
        retrieved_ids = [r.chunk.id for r in retrieved]
        per_case.append(
            {
                "id": tc.id,
                "recall@3": recall_at_k(retrieved_ids, relevant, k=3),
                "recall@5": recall_at_k(retrieved_ids, relevant, k=5),
                f"recall@{args.k}": recall_at_k(retrieved_ids, relevant, k=args.k),
                "precision@3": precision_at_k(retrieved_ids, relevant, k=3),
                f"ndcg@{args.k}": ndcg_at_k(retrieved_ids, relevant, k=args.k),
                "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant),
            }
        )
    elapsed = time.time() - t0
    print(f"  -> done in {elapsed:.1f}s ({elapsed / len(test_cases) * 1000:.0f}ms/query)")

    def mean(key: str) -> float:
        return sum(c[key] for c in per_case) / len(per_case)

    summary = {
        "corpus_size": len(corpus),
        "n_test_cases": len(test_cases),
        "k": args.k,
        "mean_recall@3": mean("recall@3"),
        "mean_recall@5": mean("recall@5"),
        f"mean_recall@{args.k}": mean(f"recall@{args.k}"),
        "mean_precision@3": mean("precision@3"),
        f"mean_ndcg@{args.k}": mean(f"ndcg@{args.k}"),
        "mrr": mean_reciprocal_rank([c["reciprocal_rank"] for c in per_case]),
        "seconds_per_query_ms": elapsed / len(test_cases) * 1000,
    }

    print("\n=== Results ===")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
