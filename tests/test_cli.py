"""Tests for doctor_rounds.cli.

`run` is tested with the adapter-building functions monkeypatched to
return fakes — real Chroma/Ollama/HuggingFace calls have no place in the
fast test suite, and the CLI's own logic (config parsing, wiring adapters
together, writing results) is what's under test here, not any adapter's
behavior (that's covered in tests/test_runner.py and the adapter-specific
test files).
"""

import json
import locale

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


class UnicodeFakeLLM:
    """Returns text containing characters cp1252 can't represent — real
    PubMedQA abstracts and generated answers routinely do too."""

    def generate(self, prompt: str) -> str:
        return "Café study — α-blockers reduced risk → significant."


class FakeJudgeForClassifierTest:
    def score_faithfulness(self, answer: str, context: str) -> float:
        return 0.9

    def score_relevance(self, answer: str, question: str) -> float:
        return 0.8


class TestInit:
    def test_writes_starter_config(self, tmp_path):
        out = tmp_path / "config.yaml"
        result = runner.invoke(cli.app, ["init", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "pipeline_name" in out.read_text(encoding="utf-8")

    def test_refuses_to_overwrite_existing_file(self, tmp_path):
        out = tmp_path / "config.yaml"
        out.write_text("existing content", encoding="utf-8")
        result = runner.invoke(cli.app, ["init", str(out)])
        assert result.exit_code != 0
        assert out.read_text(encoding="utf-8") == "existing content"  # untouched


class TestRun:
    def _invoke_with_fakes(self, monkeypatch, tmp_path, config_text: str):
        monkeypatch.setattr(cli, "_build_vector_store", lambda cfg: FakeVectorStore())
        monkeypatch.setattr(cli, "_build_llm", lambda cfg: FakeLLM())
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg, corpus, llm: [TEST_CASE])

        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")
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
        data = json.loads(results_path.read_text(encoding="utf-8"))
        assert data["pipeline_name"] == "test-pipeline"
        assert len(data["case_results"]) == 1
        assert data["case_results"][0]["test_case"]["id"] == "tc1"

    def test_use_judge_false_omits_generation_metrics(self, monkeypatch, tmp_path):
        result, config_path = self._invoke_with_fakes(monkeypatch, tmp_path, MINIMAL_CONFIG)
        assert result.exit_code == 0, result.output
        data = json.loads(config_path.with_suffix(".results.json").read_text(encoding="utf-8"))
        score_names = {s["name"] for s in data["case_results"][0]["scores"]}
        assert "faithfulness" not in score_names

    def test_judge_type_local_classifier_reaches_build_classifier_judge(self, monkeypatch, tmp_path):
        # end-to-end config wiring for judge.type: local_classifier --
        # _build_classifier_judge itself is exercised for real (against a
        # tiny checkpoint) by TestBuildClassifierJudgeReal above; this
        # only checks that the config option actually reaches it.
        captured = {}

        def fake_classifier_judge(cfg, llm):
            captured["cfg"] = cfg
            return FakeJudgeForClassifierTest()

        monkeypatch.setattr(cli, "_build_classifier_judge", fake_classifier_judge)
        config_text = MINIMAL_CONFIG.replace("use_judge: false", "use_judge: true") + (
            "judge:\n  type: local_classifier\n  model_dir: some/dir\n"
        )
        result, _ = self._invoke_with_fakes(monkeypatch, tmp_path, config_text)
        assert result.exit_code == 0, result.output
        assert captured["cfg"] == {"type": "local_classifier", "model_dir": "some/dir"}

    def test_writes_results_with_characters_outside_cp1252(self, monkeypatch, tmp_path):
        # Regression test: Path.write_text()'s default encoding is the
        # platform's preferred one (cp1252 on Windows), not UTF-8 -- real
        # PubMedQA/generated text routinely contains characters cp1252
        # can't represent (en/em dashes, Greek letters, accents), which
        # crashed `doctor-rounds run` on Windows with a UnicodeEncodeError
        # writing the results file. cli.py now passes encoding="utf-8"
        # explicitly everywhere it reads/writes text. Patching
        # locale.getpreferredencoding (the mechanism Path.write_text
        # actually falls back to) reproduces the crash on any OS this
        # suite runs on, not just Windows -- CI runs on ubuntu-latest,
        # which alone would never have caught this.
        monkeypatch.setattr(locale, "getpreferredencoding", lambda do_setlocale=True: "cp1252")
        monkeypatch.setattr(cli, "_build_vector_store", lambda cfg: FakeVectorStore())
        monkeypatch.setattr(cli, "_build_llm", lambda cfg: UnicodeFakeLLM())
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg, corpus, llm: [TEST_CASE])

        config_path = tmp_path / "config.yaml"
        config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")
        result = runner.invoke(cli.app, ["run", str(config_path)])
        assert result.exit_code == 0, result.output

        data = json.loads(config_path.with_suffix(".results.json").read_text(encoding="utf-8"))
        assert "α-blockers" in data["case_results"][0]["output"]["generated_answer"]

    def test_unsupported_vector_store_type_fails_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg, corpus, llm: [TEST_CASE])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(MINIMAL_CONFIG.replace("type: chroma", "type: pinecone"), encoding="utf-8")
        result = runner.invoke(cli.app, ["run", str(config_path)])
        assert result.exit_code != 0
        assert "pinecone" in str(result.exception) or "pinecone" in result.output

    def test_unsupported_llm_type_fails_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "_build_vector_store", lambda cfg: FakeVectorStore())
        monkeypatch.setattr(cli, "_load_corpus", lambda cfg: list(CHUNKS))
        monkeypatch.setattr(cli, "_load_test_cases", lambda cfg, corpus, llm: [TEST_CASE])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(MINIMAL_CONFIG.replace("type: ollama", "type: made-up-provider"), encoding="utf-8")
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


class TestBuildJudge:
    """`_build_classifier_judge` is monkeypatched here rather than
    exercised for real -- it needs torch/transformers and a trained
    checkpoint on disk, covered separately by the
    @pytest.mark.integration test below using a tiny test-fixture model."""

    def test_llm_is_the_default(self):
        judge = cli._build_judge({}, FakeLLM())
        from doctor_rounds.metrics.generation import LLMJudge

        assert isinstance(judge, LLMJudge)

    def test_llm_type_explicit(self):
        judge = cli._build_judge({"type": "llm"}, FakeLLM())
        from doctor_rounds.metrics.generation import LLMJudge

        assert isinstance(judge, LLMJudge)

    def test_local_classifier_type_delegates_to_build_classifier_judge(self, monkeypatch):
        sentinel = object()
        captured = {}

        def fake_build(cfg, llm):
            captured["cfg"] = cfg
            captured["llm"] = llm
            return sentinel

        monkeypatch.setattr(cli, "_build_classifier_judge", fake_build)
        llm = FakeLLM()
        cfg = {"type": "local_classifier", "model_dir": "some/dir"}
        result = cli._build_judge(cfg, llm)
        assert result is sentinel
        assert captured["cfg"] == cfg
        assert captured["llm"] is llm

    def test_unsupported_type_raises_with_the_bad_value_named(self):
        with pytest.raises(Exception, match="made-up-judge"):
            cli._build_judge({"type": "made-up-judge"}, FakeLLM())


@pytest.mark.integration
class TestBuildClassifierJudgeReal:
    """Builds a real `ClassifierJudge` against a tiny HF test-fixture
    checkpoint (kilobytes, not a real trained model) to verify the wiring
    actually works end to end -- torch/transformers are required, hence
    `pytest.importorskip`, consistent with test_classifier_model.py."""

    def test_builds_a_working_judge(self, tmp_path):
        pytest.importorskip("torch")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = "hf-internal-testing/tiny-random-BertForSequenceClassification"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        tokenizer.save_pretrained(tmp_path)
        model.save_pretrained(tmp_path)

        class NumericFakeLLM:
            def generate(self, prompt: str) -> str:
                return "5"  # LLMJudge normalizes 0-10 scores to 0-1

        judge = cli._build_classifier_judge({"model_dir": str(tmp_path)}, NumericFakeLLM())
        assert 0.0 <= judge.score_faithfulness("a claim", "a context") <= 1.0
        # relevance is delegated to the plain LLMJudge, not the classifier
        assert judge.score_relevance("an answer", "a question") == pytest.approx(0.5)


class TestLoadCorpusAndTestCasesSourceValidation:
    """Only the source-validation branch — the pubmedqa-backed happy path
    is real network I/O, covered by data/pubmedqa.py's own integration
    tests rather than re-tested here."""

    def test_load_corpus_rejects_unsupported_source(self):
        with pytest.raises(Exception, match="made-up-source"):
            cli._load_corpus({"source": "made-up-source"})

    def test_load_test_cases_rejects_unsupported_source(self):
        with pytest.raises(Exception, match="made-up-source"):
            cli._load_test_cases({"source": "made-up-source"}, [], FakeLLM())


class TestLoadTestCasesGenerated:
    """source: generated needs no network — it's testset.generator run
    against an already-loaded corpus, so (unlike pubmedqa) it's exercised
    for real here rather than deferred to an integration test."""

    def test_generates_from_corpus_using_the_given_llm(self):
        class GeneratingFakeLLM:
            def generate(self, prompt: str) -> str:
                return "QUESTION: What is it?\nANSWER: Metformin."

        result = cli._load_test_cases(
            {"source": "generated", "n": 1, "seed": 0}, list(CHUNKS), GeneratingFakeLLM()
        )
        assert len(result) == 1
        assert result[0].ground_truth_answer == "Metformin."

    def test_defaults_n_to_50(self):
        class CountingFakeLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt: str) -> str:
                self.calls += 1
                return "QUESTION: Q?\nANSWER: A."

        chunks = [Chunk(id=f"c{i}", text=f"passage {i}") for i in range(60)]
        llm = CountingFakeLLM()
        result = cli._load_test_cases({"source": "generated"}, chunks, llm)
        assert len(result) == 50
        assert llm.calls == 50


@pytest.mark.integration
class TestBuildVectorStoreReal:
    def test_chroma_type_builds_a_working_store(self):
        store = cli._build_vector_store({"type": "chroma"})
        store.add(list(CHUNKS))
        results = store.retrieve("diabetes treatment", k=1)
        assert results[0].chunk.id == "c1"
