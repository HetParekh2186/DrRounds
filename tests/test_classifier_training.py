"""Tests for doctor_rounds.classifier.training.

`compute_classification_metrics` needs scikit-learn, which — unlike
torch/transformers — is a small, cheap dependency and is included in the
`dev` extra (see pyproject.toml), so these run in the default suite.
"""

import pytest

from doctor_rounds.classifier.training import (
    build_training_set,
    class_weights,
    compute_classification_metrics,
)
from doctor_rounds.classifier.types import FaithfulnessExample

SUPPORTED = FaithfulnessExample(claim="32% of patients responded.", context="ctx", label=True, source="scifact")
CONTRADICTED = FaithfulnessExample(claim="No patients responded.", context="ctx", label=False, source="scifact")


class TestBuildTrainingSet:
    def test_augments_with_synthetic_negatives_by_default(self):
        result = build_training_set([SUPPORTED, CONTRADICTED])
        # 2 real examples + 1 synthetic negative generated from SUPPORTED
        assert len(result) == 3
        sources = [ex.source for ex in result]
        assert sources.count("scifact") == 2
        assert sources.count("synthetic_corruption") == 1

    def test_augment_false_returns_only_real_examples(self):
        result = build_training_set([SUPPORTED, CONTRADICTED], augment_with_synthetic=False)
        assert result == [SUPPORTED, CONTRADICTED]

    def test_empty_input_returns_empty_list(self):
        assert build_training_set([]) == []

    def test_does_not_mutate_its_input(self):
        original = [SUPPORTED, CONTRADICTED]
        build_training_set(original)
        assert original == [SUPPORTED, CONTRADICTED]


class TestClassWeights:
    def test_balanced_labels_get_equal_weight(self):
        weight_neg, weight_pos = class_weights([0, 0, 1, 1])
        assert weight_neg == pytest.approx(1.0)
        assert weight_pos == pytest.approx(1.0)

    def test_minority_class_gets_higher_weight(self):
        # 3 negative, 1 positive -- the rare positive class should be weighted up
        weight_neg, weight_pos = class_weights([0, 0, 0, 1])
        assert weight_pos > weight_neg

    def test_weights_are_inversely_proportional_to_frequency(self):
        weight_neg, weight_pos = class_weights([0, 0, 0, 1])
        # n=4, n_neg=3, n_pos=1 -> weight_neg = 4/(2*3), weight_pos = 4/(2*1)
        assert weight_neg == pytest.approx(4 / 6)
        assert weight_pos == pytest.approx(4 / 2)


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        metrics = compute_classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_all_wrong_predictions(self):
        metrics = compute_classification_metrics([1, 0], [0, 1])
        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0

    def test_returns_plain_floats(self):
        metrics = compute_classification_metrics([1, 0], [1, 1])
        assert all(isinstance(v, float) for v in metrics.values())

    def test_no_positive_predictions_does_not_raise(self):
        # precision is undefined (0/0) when nothing is predicted positive
        # -- zero_division=0 should make this return 0.0, not raise
        metrics = compute_classification_metrics([1, 1], [0, 0])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
