"""Tests for doctor_rounds.core.compare."""

import pytest

from doctor_rounds.core.compare import compare_reports, format_comparison_markdown
from doctor_rounds.core.types import (
    CaseResult,
    Chunk,
    EvalReport,
    MetricScore,
    PipelineOutput,
    RetrievedChunk,
    TestCase,
)


def make_report(pipeline_name: str, **metrics: float) -> EvalReport:
    """Builds a minimal, valid single-case EvalReport with the given
    metric name/value pairs — a fixture, not something under test."""
    chunk = Chunk(id="c1", text="some clinical passage")
    test_case = TestCase(
        id="tc1",
        question="What is the recommended dosage?",
        ground_truth_answer="500mg twice daily.",
        ground_truth_chunk_ids=[chunk.id],
    )
    output = PipelineOutput(
        test_case_id="tc1",
        retrieved_chunks=[RetrievedChunk(chunk=chunk, rank=0, score=0.9)],
        generated_answer="500mg twice daily.",
    )
    scores = [MetricScore(name=name, value=value) for name, value in metrics.items()]
    return EvalReport(pipeline_name=pipeline_name, case_results=[CaseResult(test_case=test_case, output=output, scores=scores)])


class TestCompareReports:
    def test_computes_deltas_for_shared_metrics(self):
        baseline = make_report("main", **{"recall@10": 0.80, "faithfulness": 0.50})
        new = make_report("pr-123", **{"recall@10": 0.85, "faithfulness": 0.40})
        result = compare_reports(baseline, new)

        by_name = {d.name: d for d in result.deltas}
        assert by_name["recall@10"].delta == pytest.approx(0.05)
        assert by_name["faithfulness"].delta == pytest.approx(-0.10)

    def test_flags_regression_below_threshold(self):
        baseline = make_report("main", **{"recall@10": 0.80})
        new = make_report("pr-123", **{"recall@10": 0.70})
        result = compare_reports(baseline, new, regression_threshold=0.02)
        assert result.deltas[0].is_regression is True
        assert result.has_regressions is True

    def test_small_drop_within_threshold_is_not_a_regression(self):
        baseline = make_report("main", **{"recall@10": 0.80})
        new = make_report("pr-123", **{"recall@10": 0.79})
        result = compare_reports(baseline, new, regression_threshold=0.02)
        assert result.deltas[0].is_regression is False
        assert result.has_regressions is False

    def test_improvement_is_not_a_regression(self):
        baseline = make_report("main", **{"recall@10": 0.80})
        new = make_report("pr-123", **{"recall@10": 0.95})
        result = compare_reports(baseline, new)
        assert result.deltas[0].is_regression is False

    def test_metric_only_in_baseline_is_skipped_not_treated_as_zero(self):
        baseline = make_report("main", **{"recall@10": 0.80, "faithfulness": 0.50})
        new = make_report("pr-123", **{"recall@10": 0.80})
        result = compare_reports(baseline, new)
        names = {d.name for d in result.deltas}
        assert names == {"recall@10"}

    def test_no_shared_metrics_returns_empty_deltas(self):
        baseline = make_report("main", faithfulness=0.5)
        new = make_report("pr-123", relevance=0.5)
        result = compare_reports(baseline, new)
        assert result.deltas == []
        assert result.has_regressions is False

    def test_carries_pipeline_names_through(self):
        baseline = make_report("main", **{"recall@10": 0.8})
        new = make_report("pr-123", **{"recall@10": 0.8})
        result = compare_reports(baseline, new)
        assert result.baseline_pipeline == "main"
        assert result.new_pipeline == "pr-123"


class TestFormatComparisonMarkdown:
    def test_includes_pipeline_names_and_metric_row(self):
        baseline = make_report("main", **{"recall@10": 0.80})
        new = make_report("pr-123", **{"recall@10": 0.85})
        markdown = format_comparison_markdown(compare_reports(baseline, new))
        assert "main" in markdown
        assert "pr-123" in markdown
        assert "recall@10" in markdown
        assert "0.8000" in markdown
        assert "0.8500" in markdown

    def test_flags_regression_in_output(self):
        baseline = make_report("main", **{"recall@10": 0.80})
        new = make_report("pr-123", **{"recall@10": 0.50})
        markdown = format_comparison_markdown(compare_reports(baseline, new))
        assert "Possible regression" in markdown
        assert "recall@10" in markdown.split("Possible regression")[1]

    def test_no_regressions_message_when_clean(self):
        baseline = make_report("main", **{"recall@10": 0.80})
        new = make_report("pr-123", **{"recall@10": 0.85})
        markdown = format_comparison_markdown(compare_reports(baseline, new))
        assert "No regressions detected" in markdown

    def test_no_shared_metrics_message(self):
        baseline = make_report("main", faithfulness=0.5)
        new = make_report("pr-123", relevance=0.5)
        markdown = format_comparison_markdown(compare_reports(baseline, new))
        assert "No metrics are shared" in markdown
