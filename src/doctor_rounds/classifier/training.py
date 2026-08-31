"""Training-set assembly and metric computation for the faithfulness
classifier — factored out of `scripts/train_faithfulness_classifier.py`
so this logic is unit-tested without actually running a training loop
(the script itself, like `scripts/run_pubmedqa_benchmark.py`, is thin
orchestration over already-tested pieces and isn't unit tested itself).
"""

from __future__ import annotations

from doctor_rounds.classifier.corruption import generate_synthetic_negatives
from doctor_rounds.classifier.types import FaithfulnessExample


def build_training_set(
    scifact_examples: list[FaithfulnessExample],
    *,
    augment_with_synthetic: bool = True,
) -> list[FaithfulnessExample]:
    """Combines real SciFact examples with synthetic negatives generated
    from their supported subset (see `classifier.corruption` for why that
    second negative source matters).

    `scifact_examples` is a plain argument, not fetched here, so this
    stays a pure function of its inputs — the caller (the training
    script) owns the one real network call.
    """
    if not augment_with_synthetic:
        return list(scifact_examples)
    return list(scifact_examples) + generate_synthetic_negatives(scifact_examples)


def class_weights(labels: list[int]) -> tuple[float, float]:
    """Inverse-frequency class weights `(weight_for_0, weight_for_1)`:
    `weight[c] = n / (num_classes * n_c)`, so a rarer class's mistakes
    cost proportionally more during training.

    Plain floats (not a torch tensor) so this stays torch-free and
    unit-testable in the default suite — `scripts/train_faithfulness_classifier.py`
    wraps the result in `torch.tensor(...)` itself. Synthetic-negative
    augmentation (`build_training_set`) shifts the training set's class
    balance away from the real validation set's; an unweighted loss then
    just teaches "predict the majority training-set class more often" —
    found, not assumed, after an unweighted first training run scored
    *below* the eval set's own majority-class baseline (see the README's
    "Faithfulness classifier" section).
    """
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    return (n / (2 * n_neg), n / (2 * n_pos))


def compute_classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    """Accuracy/precision/recall/F1 for the binary faithfulness task.

    Plain float returns (not numpy scalars) so this is directly
    JSON-serializable and directly comparable across dict equality in
    tests — sklearn's `average="binary"` treats 1 (supported) as the
    positive class, matching `FaithfulnessExample.label`'s meaning.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
