"""Tests for doctor_rounds.metrics.generation.

Uses a small fake LLM (just implements the `LLM` protocol's `generate`
method) rather than mocking a real adapter — LLMJudge only ever depends on
that one method, so a fake is simpler and faster than patching Anthropic/
OpenAI/Ollama internals the way tests/test_llm_adapters.py does for the
adapters themselves.
"""

import pytest

from doctor_rounds.metrics.generation import LLMJudge, _parse_score


class FakeLLM:
    """Returns a pre-programmed response regardless of prompt, and records
    every prompt it was called with so tests can assert on prompt content."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class TestParseScore:
    def test_extracts_plain_integer(self):
        assert _parse_score("8") == pytest.approx(0.8)

    def test_extracts_integer_from_surrounding_text(self):
        assert _parse_score("The score is 7 out of 10.") == pytest.approx(0.7)

    def test_extracts_decimal(self):
        assert _parse_score("7.5") == pytest.approx(0.75)

    def test_zero_score(self):
        assert _parse_score("0") == 0.0

    def test_ten_score_normalizes_to_one(self):
        assert _parse_score("10") == 1.0

    def test_clamps_out_of_range_score(self):
        # a judge model ignoring the 0-10 instruction shouldn't silently
        # produce a score outside the 0-1 range every other metric uses
        assert _parse_score("15") == 1.0

    def test_raises_when_no_number_present(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_score("I cannot score this.")


class TestLLMJudgeFaithfulness:
    def test_returns_normalized_score(self):
        judge = LLMJudge(FakeLLM("9"))
        assert judge.score_faithfulness("Metformin is first-line.", "context here") == pytest.approx(0.9)

    def test_prompt_includes_context_and_answer(self):
        llm = FakeLLM("5")
        judge = LLMJudge(llm)
        judge.score_faithfulness(answer="the answer text", context="the context text")
        assert "the answer text" in llm.prompts[0]
        assert "the context text" in llm.prompts[0]


class TestLLMJudgeRelevance:
    def test_returns_normalized_score(self):
        judge = LLMJudge(FakeLLM("6"))
        assert judge.score_relevance("some answer", "some question") == pytest.approx(0.6)

    def test_prompt_includes_question_and_answer(self):
        llm = FakeLLM("5")
        judge = LLMJudge(llm)
        judge.score_relevance(answer="the answer text", question="the question text")
        assert "the answer text" in llm.prompts[0]
        assert "the question text" in llm.prompts[0]

    def test_faithfulness_and_relevance_use_different_prompts(self):
        # regression guard against accidentally wiring both methods to the
        # same prompt template
        llm = FakeLLM("5")
        judge = LLMJudge(llm)
        judge.score_faithfulness(answer="x", context="y")
        judge.score_relevance(answer="x", question="y")
        assert llm.prompts[0] != llm.prompts[1]
