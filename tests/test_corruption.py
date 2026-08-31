"""Tests for doctor_rounds.classifier.corruption.

All corruption functions are pure and deterministic (no seed needed) —
these tests exercise that directly rather than through any fake.
"""

from doctor_rounds.classifier.corruption import (
    corrupt_claim,
    generate_synthetic_negatives,
    negate_claim,
    perturb_number,
)
from doctor_rounds.classifier.types import FaithfulnessExample


class TestNegateClaim:
    def test_inserts_not_after_first_auxiliary(self):
        assert negate_claim("Metformin is first-line therapy.") == "Metformin is not first-line therapy."

    def test_inserts_not_after_first_of_multiple_auxiliaries(self):
        # only the first should be negated
        assert negate_claim("It is true and it was tested.") == "It is not true and it was tested."

    def test_falls_back_to_wrapping_when_no_auxiliary_present(self):
        assert negate_claim("Aspirin reduces inflammation.") == "It is not true that aspirin reduces inflammation."

    def test_is_deterministic(self):
        claim = "Metformin is first-line therapy."
        assert negate_claim(claim) == negate_claim(claim)

    def test_empty_claim_returned_unchanged(self):
        assert negate_claim("") == ""

    def test_whitespace_only_claim_returned_unchanged(self):
        assert negate_claim("   ") == "   "


class TestPerturbNumber:
    def test_doubles_integer_percentage(self):
        assert perturb_number("32% of patients responded.") == "64% of patients responded."

    def test_doubles_decimal_number(self):
        assert perturb_number("The rate was 1.5 per year.") == "The rate was 3 per year."

    def test_returns_none_when_no_number_present(self):
        assert perturb_number("Aspirin reduces inflammation.") is None

    def test_zero_is_perturbed_to_one_not_left_as_zero(self):
        assert perturb_number("The risk was 0 percent.") == "The risk was 1 percent."

    def test_only_perturbs_first_number(self):
        assert perturb_number("5 of 10 patients responded.") == "10 of 10 patients responded."


class TestCorruptClaim:
    def test_prefers_number_perturbation_when_a_number_is_present(self):
        assert corrupt_claim("32% of patients responded.") == "64% of patients responded."

    def test_falls_back_to_negation_when_no_number_present(self):
        assert corrupt_claim("Aspirin reduces inflammation.") == "It is not true that aspirin reduces inflammation."

    def test_result_differs_from_original(self):
        for claim in ["32% of patients responded.", "Aspirin reduces inflammation."]:
            assert corrupt_claim(claim) != claim


class TestGenerateSyntheticNegatives:
    def test_produces_one_negative_per_supported_example(self):
        examples = [
            FaithfulnessExample(claim="32% of patients responded.", context="ctx", label=True),
            FaithfulnessExample(claim="Aspirin reduces inflammation.", context="ctx", label=True),
        ]
        negatives = generate_synthetic_negatives(examples)
        assert len(negatives) == 2
        assert all(n.label is False for n in negatives)
        assert all(n.source == "synthetic_corruption" for n in negatives)

    def test_preserves_original_context(self):
        examples = [FaithfulnessExample(claim="32% of patients responded.", context="original context", label=True)]
        negatives = generate_synthetic_negatives(examples)
        assert negatives[0].context == "original context"

    def test_claim_text_actually_changed(self):
        examples = [FaithfulnessExample(claim="32% of patients responded.", context="ctx", label=True)]
        negatives = generate_synthetic_negatives(examples)
        assert negatives[0].claim != "32% of patients responded."

    def test_skips_already_negative_examples(self):
        examples = [FaithfulnessExample(claim="32% of patients responded.", context="ctx", label=False)]
        assert generate_synthetic_negatives(examples) == []

    def test_empty_input_returns_empty_list(self):
        assert generate_synthetic_negatives([]) == []

    def test_skips_example_when_corruption_produces_no_change(self):
        # an empty/malformed claim has no number to perturb and
        # negate_claim returns it unchanged (see its own empty-string
        # test) -- corrupted == original, so it must be skipped rather
        # than mislabeled as a negative example with identical text
        examples = [FaithfulnessExample(claim="", context="ctx", label=True)]
        assert generate_synthetic_negatives(examples) == []
