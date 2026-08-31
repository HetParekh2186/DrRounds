"""Tests for doctor_rounds.data.scifact.

`_corpus_lookup` and `row_to_example` are pure and tested here with
hand-built fixture rows (the shape `pandas.DataFrame.to_dict("records")`
produces) — no network. `load_examples` does a real HuggingFace Hub read
and is covered separately by an integration test.
"""

import pytest

from doctor_rounds.data.scifact import _corpus_lookup, load_examples, row_to_example

CORPUS_ROWS = [
    {"doc_id": 111, "title": "T1", "abstract": ["Sentence one.", "Sentence two."], "structured": False},
    {"doc_id": 222, "title": "T2", "abstract": ["Another abstract sentence."], "structured": False},
]


class TestCorpusLookup:
    def test_joins_abstract_sentences(self):
        lookup = _corpus_lookup(CORPUS_ROWS)
        assert lookup["111"] == "Sentence one. Sentence two."

    def test_keys_are_strings(self):
        lookup = _corpus_lookup(CORPUS_ROWS)
        assert set(lookup.keys()) == {"111", "222"}


class TestRowToExample:
    def test_support_row_becomes_positive_example(self):
        row = {"claim": "Claim text.", "evidence_doc_id": "111", "evidence_label": "SUPPORT"}
        ex = row_to_example(row, _corpus_lookup(CORPUS_ROWS))
        assert ex is not None
        assert ex.claim == "Claim text."
        assert ex.context == "Sentence one. Sentence two."
        assert ex.label is True
        assert ex.source == "scifact"

    def test_contradict_row_becomes_negative_example(self):
        row = {"claim": "Claim text.", "evidence_doc_id": "222", "evidence_label": "CONTRADICT"}
        ex = row_to_example(row, _corpus_lookup(CORPUS_ROWS))
        assert ex is not None
        assert ex.label is False

    def test_unverifiable_row_returns_none(self):
        row = {"claim": "Claim text.", "evidence_doc_id": "", "evidence_label": ""}
        assert row_to_example(row, _corpus_lookup(CORPUS_ROWS)) is None

    def test_missing_evidence_doc_returns_none(self):
        # defensive: a SUPPORT/CONTRADICT row whose cited doc isn't in the
        # corpus lookup shouldn't crash the whole load
        row = {"claim": "Claim text.", "evidence_doc_id": "999", "evidence_label": "SUPPORT"}
        assert row_to_example(row, _corpus_lookup(CORPUS_ROWS)) is None


class TestLoadExamplesValidation:
    def test_rejects_test_split(self):
        with pytest.raises(ValueError, match="no public labels"):
            load_examples("test")

    def test_rejects_unknown_split(self):
        with pytest.raises(ValueError, match="made-up-split"):
            load_examples("made-up-split")


@pytest.mark.integration
class TestRealSciFactLoading:
    def test_loads_real_labeled_train_examples(self):
        examples = load_examples("train")
        assert len(examples) > 500
        assert any(ex.label is True for ex in examples)
        assert any(ex.label is False for ex in examples)
        assert all(ex.claim and ex.context for ex in examples)

    def test_loads_real_validation_examples(self):
        examples = load_examples("validation")
        assert len(examples) > 100
