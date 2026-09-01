"""The `doctor-rounds` command-line tool.

`doctor-rounds init` scaffolds a starter YAML config. `doctor-rounds run
config.yaml` builds the vector store / LLM / corpus / test cases the
config describes, runs a full evaluation via `core.runner.run_evaluation`,
and prints + saves the results — the thing that turns the rest of this
package from "a library you'd script against" into something you can
actually run.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, cast

import typer
import yaml
from rich.console import Console
from rich.table import Table

from doctor_rounds.adapters.llm import LLM, AnthropicLLM, OllamaLLM, OpenAILLM
from doctor_rounds.adapters.vectorstore import ChromaVectorStore, VectorStore
from doctor_rounds.core.compare import compare_reports, format_comparison_markdown
from doctor_rounds.core.runner import run_evaluation
from doctor_rounds.core.types import Chunk, EvalReport, TestCase
from doctor_rounds.data import pubmedqa
from doctor_rounds.metrics.generation import Judge, LLMJudge
from doctor_rounds.testset.generator import generate_test_set

if sys.platform == "win32":
    # Windows' console defaults to the system codepage (cp1252 here), not
    # UTF-8 -- this project's own messages use real Unicode (em dashes,
    # arrows, emoji in `compare`'s output), which crashed with
    # UnicodeEncodeError printing to a plain PowerShell/cmd terminal.
    # Same root cause as the earlier Path.write_text() encoding fix, just
    # hitting stdout instead of a file this time. Must happen before
    # Console() below, which reads the stream's encoding at construction.
    # sys.stdout/stderr are typed as plain TextIO (no .reconfigure) even
    # though they're really io.TextIOWrapper at runtime on every real
    # interpreter -- cast rather than suppress, so a genuinely different
    # stream type would still be a real mypy error elsewhere.
    cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")
    cast(io.TextIOWrapper, sys.stderr).reconfigure(encoding="utf-8")

app = typer.Typer(
    add_completion=False,
    help="Doctor Rounds — a clinical RAG evaluation framework.",
)
console = Console()

_STARTER_CONFIG = """\
pipeline_name: my-rag-pipeline

# Only "chroma" is supported today (see adapters/vectorstore.py to add another).
vector_store:
  type: chroma
  collection_name: doctor_rounds
  embedding_model: all-MiniLM-L6-v2
  # persist_directory: ./chroma_data   # uncomment to persist the index across runs

# type: ollama (default, no API key needed — run `ollama pull llama3.1` first),
# anthropic, or openai. anthropic/openai read their API key from the usual
# env var (ANTHROPIC_API_KEY / OPENAI_API_KEY) unless api_key is set here.
llm:
  type: ollama
  model: llama3.1

# Only "pubmedqa" is supported today (see data/pubmedqa.py to add another).
corpus:
  source: pubmedqa
  split: pqa_artificial
  limit: 1000          # number of artificial-split examples to add as distractors

# source: pubmedqa (default) uses PubMedQA's labeled questions. Or
# source: generated has the `llm` above write questions from the corpus
# itself (see testset/generator.py) — use this to evaluate against your
# own documents instead of PubMedQA's pre-labeled set.
test_cases:
  source: pubmedqa
  limit: 50             # number of labeled test cases to evaluate; omit for all 1,000
  # n: 50               # (source: generated only) how many questions to generate
  # seed: 42             # (source: generated only) makes chunk sampling reproducible

k: 10                   # how many chunks to retrieve per question
use_judge: true         # score faithfulness/relevance (extra calls, LLM or local classifier)

# type: llm (default) scores both faithfulness and relevance with the `llm` above.
# type: local_classifier scores faithfulness with a locally fine-tuned classifier
# (see scripts/train_faithfulness_classifier.py — much cheaper/faster and, per the
# README's real benchmark numbers, more reliable than a small local LLM judge) and
# relevance with the `llm` above, since the classifier is only trained for faithfulness.
judge:
  type: llm
  # model_dir: models/faithfulness-classifier   # (local_classifier only) trained checkpoint path
  # max_length: 256                              # (local_classifier only) must match training
"""


@app.command()
def init(path: str = typer.Argument("doctor-rounds.yaml", help="Where to write the config")) -> None:
    """Write a starter config file to PATH."""
    out = Path(path)
    if out.exists():
        console.print(f"[red]{path} already exists — not overwriting.[/red]")
        raise typer.Exit(1)
    out.write_text(_STARTER_CONFIG, encoding="utf-8")
    console.print(f"[green]Wrote {path}.[/green] Edit it, then run: doctor-rounds run {path}")


def _build_vector_store(cfg: dict[str, Any]) -> VectorStore:
    store_type = cfg.get("type", "chroma")
    if store_type != "chroma":
        raise typer.BadParameter(f"Unsupported vector_store.type: {store_type!r} (only 'chroma' is built in)")
    return ChromaVectorStore(
        collection_name=cfg.get("collection_name", "doctor_rounds"),
        embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        persist_directory=cfg.get("persist_directory"),
    )


def _build_llm(cfg: dict[str, Any]) -> LLM:
    llm_type = cfg.get("type", "ollama")
    if llm_type == "ollama":
        return OllamaLLM(
            model=cfg.get("model", "llama3.1"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
        )
    if llm_type == "anthropic":
        return AnthropicLLM(model=cfg.get("model", "claude-sonnet-5"), api_key=cfg.get("api_key"))
    if llm_type == "openai":
        return OpenAILLM(model=cfg.get("model", "gpt-4o-mini"), api_key=cfg.get("api_key"))
    raise typer.BadParameter(f"Unsupported llm.type: {llm_type!r} (expected ollama, anthropic, or openai)")


def _build_judge(cfg: dict[str, Any], llm: LLM) -> Judge:
    judge_type = cfg.get("type", "llm")
    if judge_type == "llm":
        return LLMJudge(llm)
    if judge_type == "local_classifier":
        return _build_classifier_judge(cfg, llm)
    raise typer.BadParameter(f"Unsupported judge.type: {judge_type!r} (expected llm or local_classifier)")


def _build_classifier_judge(cfg: dict[str, Any], llm: LLM) -> Judge:
    """Split out from `_build_judge` so tests can monkeypatch this one
    function instead of needing a real torch/transformers install and a
    trained checkpoint on disk for every CLI-flow test — see test_cli.py.

    The import is local (not at module level) because torch/transformers
    are a genuinely heavy, GPU-shaped dependency this project keeps out
    of the default install — see the `dev` extra's comment in
    pyproject.toml. Nothing here needs them unless judge.type is
    actually set to local_classifier.
    """
    from doctor_rounds.classifier.model import ClassifierJudge, LocalFaithfulnessClassifier

    classifier = LocalFaithfulnessClassifier(
        cfg.get("model_dir", "models/faithfulness-classifier"),
        max_length=cfg.get("max_length", 256),  # matches scripts/train_faithfulness_classifier.py's default
    )
    return ClassifierJudge(classifier, relevance_judge=LLMJudge(llm))


def _load_corpus(cfg: dict[str, Any]) -> list[Chunk]:
    source = cfg.get("source", "pubmedqa")
    if source != "pubmedqa":
        raise typer.BadParameter(f"Unsupported corpus.source: {source!r} (only 'pubmedqa' is built in)")
    return pubmedqa.load_corpus(
        split=cfg.get("split", "pqa_artificial"),
        limit=cfg.get("limit"),
        include_labeled=cfg.get("include_labeled", True),
    )


def _load_test_cases(cfg: dict[str, Any], corpus: list[Chunk], llm: LLM) -> list[TestCase]:
    source = cfg.get("source", "pubmedqa")
    if source == "pubmedqa":
        return pubmedqa.load_test_cases(limit=cfg.get("limit"))
    if source == "generated":
        # LLM-generated from the already-loaded corpus, for evaluating a
        # pipeline over your own documents rather than PubMedQA's — see
        # testset/generator.py. Uses the same `llm` the pipeline itself
        # will run against unless generator_llm overrides it below.
        return generate_test_set(
            corpus,
            llm,
            n=cfg.get("n", 50),
            seed=cfg.get("seed"),
        )
    raise typer.BadParameter(f"Unsupported test_cases.source: {source!r} (expected pubmedqa or generated)")


@app.command()
def run(config_path: str = typer.Argument(..., help="Path to a YAML config (see `doctor-rounds init`)")) -> None:
    """Run a full evaluation from a YAML config file."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    console.print("[bold]Loading corpus...[/bold]")
    corpus = _load_corpus(config.get("corpus", {}))
    console.print(f"  -> {len(corpus)} chunks")

    console.print("[bold]Indexing corpus...[/bold]")
    store = _build_vector_store(config.get("vector_store", {}))
    store.add(corpus)

    llm = _build_llm(config.get("llm", {}))
    judge = _build_judge(config.get("judge", {}), llm) if config.get("use_judge", True) else None

    console.print("[bold]Loading test cases...[/bold]")
    test_cases = _load_test_cases(config.get("test_cases", {}), corpus, llm)
    console.print(f"  -> {len(test_cases)} test cases")

    console.print("[bold]Running evaluation...[/bold]")
    report = run_evaluation(
        vector_store=store,
        llm=llm,
        test_cases=test_cases,
        k=config.get("k", 10),
        judge=judge,
        pipeline_name=config.get("pipeline_name", "unnamed-pipeline"),
    )

    table = Table(title=f"Results: {report.pipeline_name}")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    for name, value in sorted(report.aggregates().items()):
        table.add_row(name, f"{value:.4f}")
    console.print(table)

    out_path = Path(config_path).with_suffix(".results.json")
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Full results written to {out_path}[/green]")


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Path to a baseline .results.json (e.g. from main)"),
    new: str = typer.Argument(..., help="Path to a new .results.json (e.g. from a PR branch)"),
    threshold: float = typer.Option(0.02, help="A metric dropping by more than this counts as a regression"),
    markdown_out: str | None = typer.Option(None, "--markdown-out", help="Also write the Markdown table to this path"),
) -> None:
    """Diff two evaluation results and report metric deltas / regressions.

    Exits non-zero if any metric regressed — the mechanism behind the
    metric-diff GitHub Action ("CI for your RAG pipeline"), which runs
    this against a committed baseline and posts the result as a PR
    comment. See .github/workflows/rag-eval-pr-comment.yml.
    """
    baseline_report = EvalReport.model_validate_json(Path(baseline).read_text(encoding="utf-8"))
    new_report = EvalReport.model_validate_json(Path(new).read_text(encoding="utf-8"))

    result = compare_reports(baseline_report, new_report, regression_threshold=threshold)
    markdown = format_comparison_markdown(result)
    console.print(markdown)

    if markdown_out:
        Path(markdown_out).write_text(markdown, encoding="utf-8")

    if result.has_regressions:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
