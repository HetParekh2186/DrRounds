"""Tests for the EvalReport aggregation/triage logic in core/types.py.

The metrics functions are tested in isolation elsewhere; this file tests
the *reporting* layer that sits on top of them — aggregation across cases
and picking out the worst ones for triage.
"""

import pytest

from doctor_rounds.core.types import (
    CaseResult,
    Chunk,
    EvalReport,
    MetricScore,
    PipelineOutput,
    RetrievedChunk,
    TestCase,
)


def make_case_result(case_id: str, recall: float, faithfulness: float | None = None) -> CaseResult:
    """Builds a minimal, valid CaseResult for a given pair of metric scores
    — a fixture, not something under test itself, so kept intentionally
    simple rather than exercising every field."""
    chunk = Chunk(id=f"chunk-{case_id}", text="some clinical passage")
    test_case = TestCase(
        id=case_id,
        question="What is the recommended dosage?",
        ground_truth_answer="500mg twice daily.",
        ground_truth_chunk_ids=[chunk.id],
    )
    output = PipelineOutput(
        test_case_id=case_id,
        retrieved_chunks=[RetrievedChunk(chunk=chunk, rank=0, score=0.9)],
        generated_answer="500mg twice daily.",
    )
    scores = [MetricScore(name="recall@3", value=recall)]
    if faithfulness is not None:
        scores.append(MetricScore(name="faithfulness", value=faithfulness))
    return CaseResult(test_case=test_case, output=output, scores=scores)


class TestCaseResultScore:
    def test_returns_named_metric_value(self):
        result = make_case_result("c1", recall=0.75)
        assert result.score("recall@3") == 0.75

    def test_missing_metric_returns_none(self):
        result = make_case_result("c1", recall=0.75)
        assert result.score("faithfulness") is None


class TestEvalReportAggregate:
    def test_mean_across_cases(self):
        report = EvalReport(
            pipeline_name="test-pipeline",
            case_results=[
                make_case_result("c1", recall=1.0),
                make_case_result("c2", recall=0.0),
                make_case_result("c3", recall=0.5),
            ],
        )
        assert report.aggregate("recall@3") == pytest.approx(0.5)

    def test_unknown_metric_returns_none(self):
        report = EvalReport(pipeline_name="test-pipeline", case_results=[make_case_result("c1", 1.0)])
        assert report.aggregate("nonexistent-metric") is None

    def test_empty_report_returns_none(self):
        report = EvalReport(pipeline_name="test-pipeline", case_results=[])
        assert report.aggregate("recall@3") is None

    def test_ignores_cases_missing_the_metric(self):
        # only c2 has a faithfulness score — aggregate should average over
        # just that one case, not treat the missing case as a zero.
        report = EvalReport(
            pipeline_name="test-pipeline",
            case_results=[
                make_case_result("c1", recall=1.0),
                make_case_result("c2", recall=1.0, faithfulness=0.8),
            ],
        )
        assert report.aggregate("faithfulness") == pytest.approx(0.8)


class TestEvalReportAggregates:
    def test_returns_mean_for_every_metric_present(self):
        report = EvalReport(
            pipeline_name="test-pipeline",
            case_results=[
                make_case_result("c1", recall=1.0, faithfulness=0.9),
                make_case_result("c2", recall=0.0, faithfulness=0.7),
            ],
        )
        result = report.aggregates()
        assert result == {
            "recall@3": pytest.approx(0.5),
            "faithfulness": pytest.approx(0.8),
        }


class TestEvalReportWorstCases:
    def test_returns_lowest_scoring_cases_first(self):
        report = EvalReport(
            pipeline_name="test-pipeline",
            case_results=[
                make_case_result("best", recall=1.0),
                make_case_result("worst", recall=0.0),
                make_case_result("middle", recall=0.5),
            ],
        )
        worst = report.worst_cases("recall@3", n=2)
        assert [r.test_case.id for r in worst] == ["worst", "middle"]

    def test_respects_n_limit(self):
        report = EvalReport(
            pipeline_name="test-pipeline",
            case_results=[make_case_result(f"c{i}", recall=i / 10) for i in range(10)],
        )
        assert len(report.worst_cases("recall@3", n=3)) == 3

    def test_excludes_cases_without_the_metric(self):
        report = EvalReport(
            pipeline_name="test-pipeline",
            case_results=[
                make_case_result("has-it", recall=1.0, faithfulness=0.5),
                make_case_result("missing-it", recall=1.0),
            ],
        )
        worst = report.worst_cases("faithfulness", n=5)
        assert [r.test_case.id for r in worst] == ["has-it"]
