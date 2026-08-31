"""Tests for doctor_rounds.cli.

`run` is tested with the adapter-building functions monkeypatched to
return fakes — real Chroma/Ollama/HuggingFace calls have no place in the
fast test suite, and the CLI's own logic (config parsing, wiring adapters
together, writing results) is what's under test here, not any adapter's
behavior (that's covered in tests/test_runner.py and the adapter-specific
test files).
"""

import json

import pytest
from typer.testing import CliRunner

from doctor_rounds import cli
from doctor_rounds.adapters.llm import AnthropicLLM, OllamaLLM, OpenAILLM
from doctor_rounds.core.types import Chunk, RetrievedChunk, TestCase

runner = CliRunner()

CHUNKS = [Chunk(id="c1", text="Metformin is first-line therapy for type 2 diabetes.")]
TEST_CASE = TestCase(
    id="tc1",
    question="What is first-line therapy for type 2 diabetes?",
    ground_truth_answer="Metformin.",
    ground_truth_chunk_ids=["c1"],
)

MINIMAL_CONFIG = """\
pipeline_name: test-pipeline
vector_store:
  type: chroma
llm:
  type: ollama
corpus:
  source: pubmedqa
test_cases:
  source: pubmedqa
k: 5
use_judge: false
"""


class FakeVectorStore:
    def add(self, chunks: list[Chunk]) -> None:
        pass

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=c, rank=i) for i, c in enumerate(CHUNKS[:k])]


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "A generated answer."


class TestInit:
    def test_writes_starter_config(self, tmp_path):
        out = tmp_path / "config.yaml"
        result = runner.invoke(cli.app, ["init", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "pipeline_name" in out.read_text()

    def test_refuses_to_overwrite_existing_file(self, tmp_path):
        out = tmp_path / "config.yaml"
        out.write_text("existing content")
        result = runner.invoke(cli.app, ["init", str(out)])
        assert result.exit_code != 0
        assert out.read_text() == "existing content"  # untouched


class TestRun:
    def _invoke_with_fakes(self, monkeypatch, tmp_path, config_text: str):
        monkeypatch.setattr(cli, "_build_vector_store", lambda cfg: FakeVectorStore())
        monkeypatch.setattr(cli, "_build_llm", lambda cfg: FakeLLM())
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg: [TEST_CASE])

        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_text)
        result = runner.invoke(cli.app, ["run", str(config_path)])
        return result, config_path

    def test_full_run_succeeds_and_prints_results_table(self, monkeypatch, tmp_path):
        result, _ = self._invoke_with_fakes(monkeypatch, tmp_path, MINIMAL_CONFIG)
        assert result.exit_code == 0, result.output
        assert "test-pipeline" in result.output
        assert "recall@5" in result.output

    def test_writes_results_json_with_expected_shape(self, monkeypatch, tmp_path):
        result, config_path = self._invoke_with_fakes(monkeypatch, tmp_path, MINIMAL_CONFIG)
        assert result.exit_code == 0, result.output
        results_path = config_path.with_suffix(".results.json")
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert data["pipeline_name"] == "test-pipeline"
        assert len(data["case_results"]) == 1
        assert data["case_results"][0]["test_case"]["id"] == "tc1"

    def test_use_judge_false_omits_generation_metrics(self, monkeypatch, tmp_path):
        result, config_path = self._invoke_with_fakes(monkeypatch, tmp_path, MINIMAL_CONFIG)
        assert result.exit_code == 0, result.output
        data = json.loads(config_path.with_suffix(".results.json").read_text())
        score_names = {s["name"] for s in data["case_results"][0]["scores"]}
        assert "faithfulness" not in score_names

    def test_unsupported_vector_store_type_fails_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg: [TEST_CASE])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(MINIMAL_CONFIG.replace("type: chroma", "type: pinecone"))
        result = runner.invoke(cli.app, ["run", str(config_path)])
        assert result.exit_code != 0
        assert "pinecone" in str(result.exception) or "pinecone" in result.output

    def test_unsupported_llm_type_fails_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_build_vector_store", lambda cfg: FakeVectorStore())
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg: [TEST_CASE])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(MINIMAL_CONFIG.replace("type: ollama", "type: made-up-provider"))
        result = runner.invoke(cli.app, ["run", str(config_path)])
        assert result.exit_code != 0
        assert "made-up-provider" in str(result.exception) or "made-up-provider" in result.output


class TestBuildLLM:
    """Direct unit tests, not through the CLI — constructing an adapter
    doesn't touch the network (only .generate() does), so these are cheap
    and don't need mocking."""

    def test_ollama_is_the_default(self):
        assert isinstance(cli._build_llm({}), OllamaLLM)

    def test_ollama_passes_through_model_and_base_url(self):
        llm = cli._build_llm({"type": "ollama", "model": "mistral", "base_url": "http://x:1234"})
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "mistral"
        assert llm.base_url == "http://x:1234"

    def test_anthropic(self):
        llm = cli._build_llm({"type": "anthropic", "model": "claude-sonnet-5"})
        assert isinstance(llm, AnthropicLLM)
        assert llm.model == "claude-sonnet-5"

    def test_openai(self):
        llm = cli._build_llm({"type": "openai", "model": "gpt-4o-mini"})
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o-mini"

    def test_unsupported_type_raises_with_the_bad_value_named(self):
        with pytest.raises(Exception, match="made-up-provider"):
            cli._build_llm({"type": "made-up-provider"})


class TestLoadCorpusAndTestCasesSourceValidation:
    """Only the source-validation branch — the pubmedqa-backed happy path
    is real network I/O, covered by data/pubmedqa.py's own integration
    tests rather than re-tested here."""

    def test_load_corpus_rejects_unsupported_source(self):
        with pytest.raises(Exception, match="made-up-source"):
            cli._load_corpus({"source": "made-up-source"})

    def test_load_test_cases_rejects_unsupported_source(self):
        with pytest.raises(Exception, match="made-up-source"):
            cli._load_test_cases({"source": "made-up-source"})


@pytest.mark.integration
class TestBuildVectorStoreReal:
    def test_chroma_type_builds_a_working_store(self):
        store = cli._build_vector_store({"type": "chroma"})
        store.add(list(CHUNKS))
        results = store.retrieve("diabetes treatment", k=1)
        assert results[0].chunk.id == "c1"
