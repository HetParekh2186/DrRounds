"""Core data models shared across the evaluation pipeline.

The vocabulary here deliberately mirrors clinical rounds: a `TestCase` is a
"patient" (a question with a known-good answer), a `PipelineOutput` is the
"chart" your RAG system produced for them, and an `EvalReport` is the
attending's summary after making rounds on the whole ward.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """How a synthetic test-set question was constructed.

    Kept explicit (rather than inferred after the fact) because retrieval
    difficulty varies systematically by type — averaging them together
    hides exactly the failure modes a RAG pipeline is most likely to have.
    """

    SINGLE_HOP = "single_hop"       # answerable from one chunk
    MULTI_HOP = "multi_hop"         # requires synthesizing multiple chunks
    DISTRACTOR = "distractor"       # near-miss chunks are deliberately present
    UNANSWERABLE = "unanswerable"   # correct behavior is to abstain


class Chunk(BaseModel):
    """One retrievable unit of context (a passage from the source corpus)."""

    id: str
    text: str
    source: str | None = Field(default=None, description="Document/file this chunk came from")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A `Chunk` as returned by a retriever, with its rank and score."""

    chunk: Chunk
    rank: int = Field(ge=0, description="0-indexed position in the ranked results")
    score: float | None = None


class TestCase(BaseModel):
    """One evaluation item: a question with a known-good answer and the
    ground-truth chunk(s) that answer should be grounded in."""

    # "TestCase" is the right domain term (borrowed deliberately from
    # testing frameworks — see the module docstring), but it collides with
    # pytest's default collection pattern; this opts it out explicitly.
    __test__: ClassVar[bool] = False

    id: str
    question: str
    ground_truth_answer: str
    ground_truth_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks in the source corpus that support the ground-truth answer",
    )
    question_type: QuestionType = QuestionType.SINGLE_HOP
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineOutput(BaseModel):
    """What a RAG pipeline under test actually produced for one `TestCase`."""

    test_case_id: str
    retrieved_chunks: list[RetrievedChunk]
    generated_answer: str
    latency_ms: float | None = None


class MetricScore(BaseModel):
    """One metric's result for one test case."""

    name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """All metric scores for a single test case, plus the inputs that
    produced them — kept together so a failing case can be inspected
    without re-joining several tables by hand."""

    test_case: TestCase
    output: PipelineOutput
    scores: list[MetricScore]

    def score(self, name: str) -> float | None:
        for s in self.scores:
            if s.name == name:
                return s.value
        return None


class EvalReport(BaseModel):
    """The result of running a full test set against a pipeline: per-case
    detail plus aggregates, so both "how did it do overall" and "which
    specific cases failed" are answerable from one object."""

    pipeline_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    case_results: list[CaseResult]

    def aggregate(self, metric_name: str) -> float | None:
        """Mean of one metric across all cases where it was computed."""
        values = [
            s.value
            for r in self.case_results
            for s in r.scores
            if s.name == metric_name
        ]
        return sum(values) / len(values) if values else None

    def aggregates(self) -> dict[str, float]:
        """Mean of every metric that appears anywhere in the report."""
        names = {s.name for r in self.case_results for s in r.scores}
        return {name: v for name in names if (v := self.aggregate(name)) is not None}

    def worst_cases(self, metric_name: str, n: int = 5) -> list[CaseResult]:
        """The `n` cases with the lowest score for `metric_name` — the
        first place to look when a regression is flagged."""
        scored = [r for r in self.case_results if r.score(metric_name) is not None]
        scored.sort(key=lambda r: r.score(metric_name))  # type: ignore[arg-type,return-value]
        return scored[:n]
