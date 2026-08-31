"""The `doctor-rounds` command-line tool.

`doctor-rounds init` scaffolds a starter YAML config. `doctor-rounds run
config.yaml` builds the vector store / LLM / corpus / test cases the
config describes, runs a full evaluation via `core.runner.run_evaluation`,
and prints + saves the results — the thing that turns the rest of this
package from "a library you'd script against" into something you can
actually run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from doctor_rounds.adapters.llm import LLM, AnthropicLLM, OllamaLLM, OpenAILLM
from doctor_rounds.adapters.vectorstore import ChromaVectorStore, VectorStore
from doctor_rounds.core.runner import run_evaluation
from doctor_rounds.core.types import Chunk, TestCase
from doctor_rounds.data import pubmedqa
from doctor_rounds.metrics.generation import LLMJudge

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

test_cases:
  source: pubmedqa
  limit: 50             # number of labeled test cases to evaluate; omit for all 1,000

k: 10                   # how many chunks to retrieve per question
use_judge: true         # score faithfulness/relevance with an LLM judge (extra LLM calls)
"""


@app.command()
def init(path: str = typer.Argument("doctor-rounds.yaml", help="Where to write the config")) -> None:
    """Write a starter config file to PATH."""
    out = Path(path)
    if out.exists():
        console.print(f"[red]{path} already exists — not overwriting.[/red]")
        raise typer.Exit(1)
    out.write_text(_STARTER_CONFIG)
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


def _load_corpus(cfg: dict[str, Any]) -> list[Chunk]:
    source = cfg.get("source", "pubmedqa")
    if source != "pubmedqa":
        raise typer.BadParameter(f"Unsupported corpus.source: {source!r} (only 'pubmedqa' is built in)")
    return pubmedqa.load_corpus(
        split=cfg.get("split", "pqa_artificial"),
        limit=cfg.get("limit"),
        include_labeled=cfg.get("include_labeled", True),
    )


def _load_test_cases(cfg: dict[str, Any]) -> list[TestCase]:
    source = cfg.get("source", "pubmedqa")
    if source != "pubmedqa":
        raise typer.BadParameter(f"Unsupported test_cases.source: {source!r} (only 'pubmedqa' is built in)")
    return pubmedqa.load_test_cases(limit=cfg.get("limit"))


@app.command()
def run(config_path: str = typer.Argument(..., help="Path to a YAML config (see `doctor-rounds init`)")) -> None:
    """Run a full evaluation from a YAML config file."""
    config = yaml.safe_load(Path(config_path).read_text())

    console.print("[bold]Loading corpus...[/bold]")
    corpus = _load_corpus(config.get("corpus", {}))
    console.print(f"  -> {len(corpus)} chunks")

    console.print("[bold]Indexing corpus...[/bold]")
    store = _build_vector_store(config.get("vector_store", {}))
    store.add(corpus)

    console.print("[bold]Loading test cases...[/bold]")
    test_cases = _load_test_cases(config.get("test_cases", {}))
    console.print(f"  -> {len(test_cases)} test cases")

    llm = _build_llm(config.get("llm", {}))
    judge = LLMJudge(llm) if config.get("use_judge", True) else None

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
    out_path.write_text(report.model_dump_json(indent=2))
    console.print(f"[green]Full results written to {out_path}[/green]")


if __name__ == "__main__":
    app()
