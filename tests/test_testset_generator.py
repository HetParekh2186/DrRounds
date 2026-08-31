"""Tests for doctor_rounds.testset.generator.

Uses the same FakeLLM-double pattern as test_generation_metrics.py — a
canned response per test, no real network or model calls.
"""

import pytest

from doctor_rounds.core.types import Chunk, QuestionType
from doctor_rounds.testset.generator import (
    _parse_response,
    generate_single_hop_test_case,
    generate_test_set,
)


class FakeLLM:
    """Returns pre-programmed responses in order (or one fixed response for
    every call), and records every prompt it was called with."""

    def __init__(self, response=None, responses=None) -> None:
        self.response = response
        self._responses = list(responses) if responses is not None else None
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._responses is not None:
            return self._responses.pop(0)
        return self.response


WELL_FORMED = "QUESTION: What is first-line therapy for type 2 diabetes?\nANSWER: Metformin."

CHUNK = Chunk(id="c1", text="Metformin is recommended as first-line pharmacologic therapy.")


class TestParseResponse:
    def test_parses_well_formed_response(self):
        result = _parse_response(WELL_FORMED)
        assert result == ("What is first-line therapy for type 2 diabetes?", "Metformin.")

    def test_parses_multiline_answer(self):
        raw = "QUESTION: Q?\nANSWER: Line one.\nLine two."
        assert _parse_response(raw) == ("Q?", "Line one.\nLine two.")

    def test_tolerates_preamble_before_question(self):
        raw = "Sure, here you go:\nQUESTION: Q?\nANSWER: A."
        assert _parse_response(raw) == ("Q?", "A.")

    def test_case_insensitive_labels(self):
        raw = "question: Q?\nanswer: A."
        assert _parse_response(raw) == ("Q?", "A.")

    def test_missing_answer_returns_none(self):
        assert _parse_response("QUESTION: Q?") is None

    def test_missing_question_returns_none(self):
        assert _parse_response("ANSWER: A.") is None

    def test_empty_string_returns_none(self):
        assert _parse_response("") is None

    def test_blank_question_returns_none(self):
        assert _parse_response("QUESTION:   \nANSWER: A.") is None


class TestGenerateSingleHopTestCase:
    def test_builds_test_case_from_well_formed_response(self):
        tc = generate_single_hop_test_case(CHUNK, FakeLLM(WELL_FORMED))
        assert tc is not None
        assert tc.question == "What is first-line therapy for type 2 diabetes?"
        assert tc.ground_truth_answer == "Metformin."
        assert tc.ground_truth_chunk_ids == ["c1"]
        assert tc.question_type == QuestionType.SINGLE_HOP

    def test_id_is_unique_per_call(self):
        llm = FakeLLM(WELL_FORMED)
        tc1 = generate_single_hop_test_case(CHUNK, llm)
        tc2 = generate_single_hop_test_case(CHUNK, llm)
        assert tc1.id != tc2.id

    def test_metadata_records_source_chunk(self):
        tc = generate_single_hop_test_case(CHUNK, FakeLLM(WELL_FORMED))
        assert tc.metadata["source_chunk_id"] == "c1"

    def test_prompt_includes_chunk_text(self):
        llm = FakeLLM(WELL_FORMED)
        generate_single_hop_test_case(CHUNK, llm)
        assert CHUNK.text in llm.prompts[0]

    def test_returns_none_on_unparseable_response(self):
        assert generate_single_hop_test_case(CHUNK, FakeLLM("not the right format")) is None

    def test_raises_on_empty_chunk_text(self):
        empty_chunk = Chunk(id="c2", text="   ")
        with pytest.raises(ValueError, match="c2"):
            generate_single_hop_test_case(empty_chunk, FakeLLM(WELL_FORMED))


class TestGenerateTestSet:
    def test_generates_one_case_per_chunk_up_to_n(self):
        chunks = [Chunk(id=f"c{i}", text=f"Passage {i} content.") for i in range(5)]
        result = generate_test_set(chunks, FakeLLM(WELL_FORMED), n=3, seed=42)
        assert len(result) == 3

    def test_caps_at_available_chunks_when_n_exceeds_corpus(self):
        chunks = [Chunk(id=f"c{i}", text=f"Passage {i}.") for i in range(2)]
        result = generate_test_set(chunks, FakeLLM(WELL_FORMED), n=10, seed=1)
        assert len(result) == 2

    def test_empty_corpus_returns_empty_list(self):
        assert generate_test_set([], FakeLLM(WELL_FORMED), n=5) == []

    def test_n_zero_returns_empty_list(self):
        chunks = [Chunk(id="c1", text="text")]
        assert generate_test_set(chunks, FakeLLM(WELL_FORMED), n=0) == []

    def test_same_seed_samples_the_same_chunks(self):
        chunks = [Chunk(id=f"c{i}", text=f"Passage {i}.") for i in range(20)]
        result_a = generate_test_set(chunks, FakeLLM(WELL_FORMED), n=5, seed=7)
        result_b = generate_test_set(chunks, FakeLLM(WELL_FORMED), n=5, seed=7)
        ids_a = {tc.ground_truth_chunk_ids[0] for tc in result_a}
        ids_b = {tc.ground_truth_chunk_ids[0] for tc in result_b}
        assert ids_a == ids_b

    def test_skips_chunks_with_unparseable_responses(self):
        chunks = [Chunk(id=f"c{i}", text=f"Passage {i}.") for i in range(3)]
        # first two responses malformed, third well-formed
        llm = FakeLLM(responses=["garbage", "also garbage", WELL_FORMED])
        result = generate_test_set(chunks, llm, n=3, seed=0)
        assert len(result) == 1
