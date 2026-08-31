"""Tests for doctor_rounds.data.pubmedqa.

The row-parsing functions are pure and tested here with hand-built fixture
rows shaped exactly like a real PubMedQA dataset row — no network needed.
The `load_*` functions that actually hit HuggingFace Hub are covered by a
single `integration`-marked test at the bottom, which is excluded from the
default `pytest` run (see pyproject.toml) but does run in CI.
"""

import pytest

from doctor_rounds.core.types import QuestionType
from doctor_rounds.data.pubmedqa import (
    load_corpus,
    load_test_cases,
    row_to_chunks,
    row_to_test_case,
)

# Shaped exactly like one row of qiaojin/PubMedQA, trimmed to realistic length.
SAMPLE_ROW = {
    "pubid": 21645374,
    "question": "Do mitochondria play a role in remodelling lace plant leaves during PCD?",
    "context": {
        "contexts": [
            "Programmed cell death (PCD) is the regulated death of cells within an organism.",
            "Results depicted mitochondrial dynamics in vivo as PCD progresses.",
        ],
        "labels": ["BACKGROUND", "RESULTS"],
        "meshes": ["Alismataceae", "Apoptosis", "Mitochondria"],
    },
    "long_answer": (
        "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace "
        "plant, and highlight the correlation of this organelle with other organelles."
    ),
    "final_decision": "yes",
}


class TestRowToChunks:
    def test_one_chunk_per_context_passage(self):
        chunks = row_to_chunks(SAMPLE_ROW)
        assert len(chunks) == 2

    def test_chunk_ids_are_unique_and_pubid_prefixed(self):
        chunks = row_to_chunks(SAMPLE_ROW)
        assert [c.id for c in chunks] == ["21645374-0", "21645374-1"]

    def test_chunk_text_matches_source_context(self):
        chunks = row_to_chunks(SAMPLE_ROW)
        assert chunks[0].text == SAMPLE_ROW["context"]["contexts"][0]
        assert chunks[1].text == SAMPLE_ROW["context"]["contexts"][1]

    def test_section_label_carried_into_metadata(self):
        chunks = row_to_chunks(SAMPLE_ROW)
        assert chunks[0].metadata["section"] == "BACKGROUND"
        assert chunks[1].metadata["section"] == "RESULTS"

    def test_source_field_identifies_originating_abstract(self):
        chunks = row_to_chunks(SAMPLE_ROW)
        assert all(c.source == "pubmed:21645374" for c in chunks)

    def test_missing_labels_list_does_not_crash(self):
        row = {**SAMPLE_ROW, "context": {**SAMPLE_ROW["context"], "labels": []}}
        chunks = row_to_chunks(row)
        assert len(chunks) == 2
        assert chunks[0].metadata == {}


class TestRowToTestCase:
    def test_basic_fields(self):
        case = row_to_test_case(SAMPLE_ROW)
        assert case.id == "21645374"
        assert case.question == SAMPLE_ROW["question"]
        assert case.ground_truth_answer == SAMPLE_ROW["long_answer"]

    def test_ground_truth_chunk_ids_match_row_to_chunks_output(self):
        # the whole point of ground_truth_chunk_ids is that they resolve
        # against the corpus row_to_chunks would build from the same row
        case = row_to_test_case(SAMPLE_ROW)
        chunk_ids = {c.id for c in row_to_chunks(SAMPLE_ROW)}
        assert set(case.ground_truth_chunk_ids) == chunk_ids

    def test_defaults_to_single_hop(self):
        case = row_to_test_case(SAMPLE_ROW)
        assert case.question_type == QuestionType.SINGLE_HOP

    def test_final_decision_preserved_in_metadata(self):
        case = row_to_test_case(SAMPLE_ROW)
        assert case.metadata["final_decision"] == "yes"


@pytest.mark.integration
class TestRealDatasetLoading:
    """Hits HuggingFace Hub for real. Slow-ish (dataset download/cache) and
    requires network — run explicitly with `pytest -m integration`."""

    def test_load_test_cases_returns_real_labeled_examples(self):
        cases = load_test_cases(limit=5)
        assert len(cases) == 5
        assert all(c.question for c in cases)
        assert all(c.ground_truth_chunk_ids for c in cases)

    def test_load_corpus_includes_labeled_and_artificial_chunks(self):
        # small limit so this stays fast even though the full artificial
        # split is 211k examples
        chunks = load_corpus(limit=10, include_labeled=True)
        # 10 artificial-split rows (~4 passages each) + all 1000 labeled
        # rows' passages (~4 each) -> comfortably more than 10 chunks
        assert len(chunks) > 10
        sources = {c.source for c in chunks}
        assert len(sources) > 1  # more than one originating abstract
