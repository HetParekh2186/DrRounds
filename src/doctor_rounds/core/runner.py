"""Orchestrates a full evaluation run: retrieve, generate, score.

This is the CLI's engine, kept separate from `doctor_rounds.cli` so it's
testable with fake adapters (no network, no real LLM calls) and reusable
outside the config-file workflow — `scripts/run_pubmedqa_benchmark.py`
predates this module and computes retrieval metrics directly for exactly
that reason; `run_evaluation` is the general version of the same idea,
extended to also generate an answer and score it.
"""

from __future__ import annotations

from doctor_rounds.adapters.llm import LLM
from doctor_rounds.adapters.vectorstore import RetrievedChunk, VectorStore
from doctor_rounds.core.types import CaseResult, EvalReport, MetricScore, PipelineOutput, TestCase
from doctor_rounds.metrics.generation import Judge
from doctor_rounds.metrics.retrieval import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

_RAG_PROMPT = """Answer the question using only the information in the context below. \
If the context doesn't contain enough information to answer, say so explicitly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """The RAG prompt template every `run_evaluation` call uses to turn
    retrieved chunks into a question a generator LLM can actually answer."""
    context = "\n\n".join(f"[{i + 1}] {rc.chunk.text}" for i, rc in enumerate(retrieved_chunks))
    return _RAG_PROMPT.format(context=context, question=question)


def run_evaluation(
    *,
    vector_store: VectorStore,
    llm: LLM,
    test_cases: list[TestCase],
    k: int = 10,
    judge: Judge | None = None,
    pipeline_name: str = "unnamed-pipeline",
) -> EvalReport:
    """Runs retrieve -> generate -> score for every test case.

    Retrieval metrics (recall@k, precision@k, NDCG@k, reciprocal rank) are
    always computed — they're pure functions over IDs, no extra cost.
    Generation metrics (faithfulness, relevance) are only computed when a
    `judge` is passed, since scoring those requires an extra LLM call per
    case and per metric.
    """
    case_results = []
    for tc in test_cases:
        retrieved = vector_store.retrieve(tc.question, k=k)
        retrieved_ids = [r.chunk.id for r in retrieved]
        relevant_ids = set(tc.ground_truth_chunk_ids)

        prompt = build_prompt(tc.question, retrieved)
        answer = llm.generate(prompt)

        scores = [
            MetricScore(name=f"recall@{k}", value=recall_at_k(retrieved_ids, relevant_ids, k)),
            MetricScore(name=f"precision@{k}", value=precision_at_k(retrieved_ids, relevant_ids, k)),
            MetricScore(name=f"ndcg@{k}", value=ndcg_at_k(retrieved_ids, relevant_ids, k)),
            MetricScore(name="reciprocal_rank", value=reciprocal_rank(retrieved_ids, relevant_ids)),
        ]

        if judge is not None:
            context_text = "\n\n".join(rc.chunk.text for rc in retrieved)
            scores.append(
                MetricScore(name="faithfulness", value=judge.score_faithfulness(answer, context_text))
            )
            scores.append(
                MetricScore(name="relevance", value=judge.score_relevance(answer, tc.question))
            )

        output = PipelineOutput(test_case_id=tc.id, retrieved_chunks=retrieved, generated_answer=answer)
        case_results.append(CaseResult(test_case=tc, output=output, scores=scores))

    return EvalReport(pipeline_name=pipeline_name, case_results=case_results)
