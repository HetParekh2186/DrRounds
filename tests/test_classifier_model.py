"""Tests for doctor_rounds.classifier.model.

Requires torch/transformers (the "classifier" extra), which are
deliberately excluded from the default `dev` install and default CI
matrix — see pyproject.toml's `dev` extra comment — so this whole module
is skipped via `importorskip` where they aren't installed, rather than
failing.

Uses `hf-internal-testing/tiny-random-BertForSequenceClassification` — a
tiny (kilobytes, not gigabytes) test fixture the transformers team
publishes specifically for fast tests like this one — saved to a tmp_path
checkpoint so `LocalFaithfulnessClassifier` is exercised exactly the way
it will load a real fine-tuned checkpoint, without needing one.
"""

import pytest

pytest.importorskip("torch")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from doctor_rounds.classifier.model import (
    ClassifierJudge,
    LocalFaithfulnessClassifier,
)

_TINY_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"


@pytest.fixture(scope="module")
def tiny_checkpoint_dir(tmp_path_factory):
    """Downloads the tiny test model once per test session and saves it
    to a local directory — the same on-disk shape a real fine-tuned
    checkpoint from scripts/train_faithfulness_classifier.py would have."""
    out_dir = tmp_path_factory.mktemp("tiny_checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(_TINY_MODEL, num_labels=2)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


class FakeJudge:
    def score_faithfulness(self, answer: str, context: str) -> float:
        raise AssertionError("ClassifierJudge should never delegate faithfulness to the LLM judge")

    def score_relevance(self, answer: str, question: str) -> float:
        return 0.42


@pytest.mark.integration
class TestLocalFaithfulnessClassifier:
    def test_score_is_a_probability(self, tiny_checkpoint_dir):
        clf = LocalFaithfulnessClassifier(tiny_checkpoint_dir)
        score = clf.score_faithfulness("Metformin is first-line therapy.", "some retrieved context")
        assert 0.0 <= score <= 1.0

    def test_defaults_to_cpu_when_no_gpu_requested(self, tiny_checkpoint_dir):
        clf = LocalFaithfulnessClassifier(tiny_checkpoint_dir, device="cpu")
        assert clf.device == "cpu"

    def test_truncates_long_context_instead_of_erroring(self, tiny_checkpoint_dir):
        clf = LocalFaithfulnessClassifier(tiny_checkpoint_dir, max_length=32)
        long_context = "word " * 500
        score = clf.score_faithfulness("a short claim", long_context)
        assert 0.0 <= score <= 1.0


@pytest.mark.integration
class TestClassifierJudge:
    def test_faithfulness_uses_the_classifier(self, tiny_checkpoint_dir):
        clf = LocalFaithfulnessClassifier(tiny_checkpoint_dir)
        judge = ClassifierJudge(clf, FakeJudge())
        score = judge.score_faithfulness("a claim", "a context")
        assert 0.0 <= score <= 1.0  # would have raised via FakeJudge if misrouted

    def test_relevance_is_delegated_to_the_llm_judge(self, tiny_checkpoint_dir):
        clf = LocalFaithfulnessClassifier(tiny_checkpoint_dir)
        judge = ClassifierJudge(clf, FakeJudge())
        assert judge.score_relevance("an answer", "a question") == 0.42
