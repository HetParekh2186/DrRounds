"""Tests for doctor_rounds.core.runner.

Uses small fakes for VectorStore/LLM/Judge (each just implementing the
one or two methods the Protocol requires) rather than the real Chroma/LLM
adapters — run_evaluation's own orchestration logic (what gets called,
in what order, how results are assembled) is what's under test here, not
any particular adapter's behavior.
"""

import pytest

from doctor_rounds.core.runner import build_prompt, run_evaluation
from doctor_rounds.core.types import Chunk, QuestionType, RetrievedChunk, TestCase


class FakeVectorStore:
    """Always returns the same fixed set of chunks, in order, regardless
    of query — retrieval quality isn't what these tests check."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.retrieve_calls: list[tuple[str, int]] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, k))
        return [RetrievedChunk(chunk=c, rank=i) for i, c in enumerate(self.chunks[:k])]


class FakeLLM:
    def __init__(self, response: str = "A generated answer.") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeJudge:
    def __init__(self, faithfulness: float = 0.8, relevance: float = 0.7) -> None:
        self.faithfulness = faithfulness
        self.relevance = relevance
        self.faithfulness_calls: list[tuple[str, str]] = []
        self.relevance_calls: list[tuple[str, str]] = []

    def score_faithfulness(self, answer: str, context: str) -> float:
        self.faithfulness_calls.append((answer, context))
        return self.faithfulness

    def score_relevance(self, answer: str, question: str) -> float:
        self.relevance_calls.append((answer, question))
        return self.relevance


CHUNKS = [
    Chunk(id="c1", text="Metformin is first-line therapy for type 2 diabetes."),
    Chunk(id="c2", text="Aspirin inhibits platelet aggregation."),
]

TEST_CASE = TestCase(
    id="tc1",
    question="What is first-line therapy for type 2 diabetes?",
    ground_truth_answer="Metformin.",
    ground_truth_chunk_ids=["c1"],
    question_type=QuestionType.SINGLE_HOP,
)


class TestBuildPrompt:
    def test_includes_question(self):
        retrieved = [RetrievedChunk(chunk=CHUNKS[0], rank=0)]
        prompt = build_prompt("What treats diabetes?", retrieved)
        assert "What treats diabetes?" in prompt

    def test_includes_all_retrieved_chunk_text(self):
        retrieved = [RetrievedChunk(chunk=c, rank=i) for i, c in enumerate(CHUNKS)]
        prompt = build_prompt("q", retrieved)
        assert CHUNKS[0].text in prompt
        assert CHUNKS[1].text in prompt

    def test_empty_retrieval_still_produces_a_valid_prompt(self):
        prompt = build_prompt("q", [])
        assert "q" in prompt


class TestRunEvaluation:
    def test_produces_one_case_result_per_test_case(self):
        report = run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=FakeLLM(),
            test_cases=[TEST_CASE],
            k=5,
        )
        assert len(report.case_results) == 1
        assert report.case_results[0].test_case.id == "tc1"

    def test_retrieval_metrics_always_computed(self):
        report = run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=FakeLLM(),
            test_cases=[TEST_CASE],
            k=5,
        )
        result = report.case_results[0]
        assert result.score("recall@5") == pytest.approx(1.0)  # c1 is in the fixed chunk list
        assert result.score("reciprocal_rank") == pytest.approx(1.0)  # c1 is ranked first

    def test_generation_metrics_absent_without_a_judge(self):
        report = run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=FakeLLM(),
            test_cases=[TEST_CASE],
            k=5,
            judge=None,
        )
        result = report.case_results[0]
        assert result.score("faithfulness") is None
        assert result.score("relevance") is None

    def test_generation_metrics_present_with_a_judge(self):
        judge = FakeJudge(faithfulness=0.9, relevance=0.6)
        report = run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=FakeLLM(),
            test_cases=[TEST_CASE],
            k=5,
            judge=judge,
        )
        result = report.case_results[0]
        assert result.score("faithfulness") == pytest.approx(0.9)
        assert result.score("relevance") == pytest.approx(0.6)

    def test_judge_receives_the_generated_answer_and_question(self):
        judge = FakeJudge()
        llm = FakeLLM(response="Metformin is first-line.")
        run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=llm,
            test_cases=[TEST_CASE],
            k=5,
            judge=judge,
        )
        assert judge.relevance_calls == [("Metformin is first-line.", TEST_CASE.question)]
        answer, context = judge.faithfulness_calls[0]
        assert answer == "Metformin is first-line."
        assert CHUNKS[0].text in context

    def test_llm_called_with_a_prompt_containing_retrieved_context(self):
        llm = FakeLLM()
        run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=llm,
            test_cases=[TEST_CASE],
            k=5,
        )
        assert CHUNKS[0].text in llm.prompts[0]
        assert TEST_CASE.question in llm.prompts[0]

    def test_pipeline_name_is_preserved_on_the_report(self):
        report = run_evaluation(
            vector_store=FakeVectorStore(list(CHUNKS)),
            llm=FakeLLM(),
            test_cases=[TEST_CASE],
            pipeline_name="my-test-pipeline",
        )
        assert report.pipeline_name == "my-test-pipeline"

    def test_k_is_forwarded_to_the_vector_store(self):
        store = FakeVectorStore(list(CHUNKS))
        run_evaluation(vector_store=store, llm=FakeLLM(), test_cases=[TEST_CASE], k=3)
        assert store.retrieve_calls == [(TEST_CASE.question, 3)]

    def test_empty_test_cases_produces_an_empty_report(self):
        report = run_evaluation(vector_store=FakeVectorStore([]), llm=FakeLLM(), test_cases=[])
        assert report.case_results == []
        assert report.aggregate("recall@10") is None
