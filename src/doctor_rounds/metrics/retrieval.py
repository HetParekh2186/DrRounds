"""Retrieval-quality metrics.

These answer one question: "did the retriever surface the chunks the answer
actually depends on?" — independent of whatever the generator did with them
afterward. Keeping retrieval and generation metrics separate is the whole
point of splitting a RAG pipeline into stages for evaluation: a low overall
score is ambiguous ("is my retriever or my prompt broken?"), but a low
recall@k with a high faithfulness score points squarely at retrieval.

Every function here is pure and takes plain lists of IDs, not the pydantic
models in `doctor_rounds.core.types` — that keeps them trivially testable
and reusable outside this project.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top `k` retrieved.

    1.0 means every chunk the answer depends on was retrieved somewhere in
    the top k; 0.0 means none were. Undefined (returns 0.0) if there are no
    relevant chunks to find, since that usually signals a malformed test
    case rather than a real "recall of zero" result.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top `k` retrieved chunks that are actually relevant.

    Low precision with high recall is a specific, actionable failure mode:
    the retriever eventually finds what it needs, but buries it in noise
    that a downstream LLM then has to sift through — exactly the setup that
    produces distraction-induced hallucination.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (rank of the first relevant chunk), 1-indexed; 0.0 if none found.

    Rewards getting *a* relevant chunk near the top over eventually finding
    all of them — appropriate when a generator only reliably attends to the
    first few chunks in its context window.
    """
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(per_query_reciprocal_ranks: list[float]) -> float:
    """Mean of `reciprocal_rank` across queries. A thin wrapper — kept as a
    named function because "MRR" is what the IR literature calls it, and a
    reader skimming for it shouldn't have to know it's just a mean."""
    if not per_query_reciprocal_ranks:
        return 0.0
    return sum(per_query_reciprocal_ranks) / len(per_query_reciprocal_ranks)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at `k`, with binary relevance.

    Unlike recall/precision, NDCG is rank-sensitive: a relevant chunk at
    position 1 counts more than the same chunk at position 10. That makes
    it the right metric when a generator is known to weight early context
    more heavily (true of most LLMs in practice), and the wrong one if you
    only care whether the right chunk showed up *somewhere* in the window —
    use `recall_at_k` for that case instead.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 if cid in relevant_ids else 0.0) / math.log2(i + 1)
            for i, cid in enumerate(ids[:k], start=1)
        )

    # ideal_ids is non-empty here: relevant_ids and k are both already
    # guaranteed non-empty/positive by the guard above, so idcg is always > 0.
    ideal_ids = list(relevant_ids)[:k]
    idcg = dcg(ideal_ids)
    return dcg(retrieved_ids) / idcg
