"""Diffs two `EvalReport`s — the mechanism behind `doctor-rounds compare`
and the GitHub Action workflow that posts metric-diff PR comments ("CI
for your RAG pipeline"): given a baseline run (e.g. main) and a new run
(e.g. a PR branch), report which metrics moved, by how much, and flag
genuine regressions rather than every bit of run-to-run noise.

Every metric this project currently produces (recall@k, precision@k,
NDCG@k, MRR, faithfulness, relevance) is "higher is better" — there's no
lower-is-better metric (like latency) in `EvalReport.aggregates()` yet,
so regression detection below assumes that direction uniformly. If a
lower-is-better metric is ever added here, this needs a per-metric
direction, not a global one.
"""

from __future__ import annotations

from pydantic import BaseModel

from doctor_rounds.core.types import EvalReport


class MetricDelta(BaseModel):
    """One metric's change between a baseline and a new run."""

    name: str
    baseline: float
    new: float
    delta: float  # new - baseline; positive means improved (see module docstring)
    is_regression: bool


class ComparisonResult(BaseModel):
    """The full diff between two `EvalReport`s."""

    baseline_pipeline: str
    new_pipeline: str
    deltas: list[MetricDelta]

    @property
    def has_regressions(self) -> bool:
        return any(d.is_regression for d in self.deltas)


def compare_reports(
    baseline: EvalReport,
    new: EvalReport,
    *,
    regression_threshold: float = 0.02,
) -> ComparisonResult:
    """Compares aggregate metrics between two `EvalReport`s.

    Only metrics present in *both* reports are compared — a metric
    measured in one run but not the other (e.g. faithfulness when
    `use_judge` differed between the two configs) is skipped rather than
    treated as a phantom 0, which would otherwise report a misleading
    "regression" for something that was never actually measured twice.
    """
    baseline_metrics = baseline.aggregates()
    new_metrics = new.aggregates()
    shared = sorted(set(baseline_metrics) & set(new_metrics))

    deltas = []
    for name in shared:
        b, n = baseline_metrics[name], new_metrics[name]
        delta = n - b
        deltas.append(
            MetricDelta(name=name, baseline=b, new=n, delta=delta, is_regression=delta < -regression_threshold)
        )

    return ComparisonResult(baseline_pipeline=baseline.pipeline_name, new_pipeline=new.pipeline_name, deltas=deltas)


def format_comparison_markdown(result: ComparisonResult) -> str:
    """Renders a `ComparisonResult` as a GitHub-flavored Markdown table,
    ready to post as a PR comment."""
    lines = [
        f"### Doctor Rounds evaluation: `{result.baseline_pipeline}` → `{result.new_pipeline}`",
        "",
        "| Metric | Baseline | New | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for d in result.deltas:
        marker = "\U0001f534" if d.is_regression else ("\U0001f7e2" if d.delta > 0 else "⚪")
        sign = "+" if d.delta >= 0 else ""
        lines.append(f"| {d.name} | {d.baseline:.4f} | {d.new:.4f} | {marker} {sign}{d.delta:.4f} |")

    if not result.deltas:
        lines += ["", "_No metrics are shared between the two runs to compare._"]
    elif result.has_regressions:
        regressed = ", ".join(d.name for d in result.deltas if d.is_regression)
        lines += ["", f"⚠️ **Possible regression** in: {regressed}"]
    else:
        lines += ["", "✅ No regressions detected."]

    return "\n".join(lines)
