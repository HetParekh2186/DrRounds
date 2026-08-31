"""Synthetic negative examples: deterministically corrupt a claim known to
be supported by some context, producing a claim that no longer is.

SciFact's own CONTRADICT-labeled claims (`data/scifact.py`) are one source
of negative training examples, but they come from a narrower distribution
than a real RAG hallucination does — SciFact's annotators wrote claims
that directly contradict a specific cited paper, which tends to look
different from a generator model quietly getting a number or a fact wrong
while everything else about the sentence stays fluent and plausible. This
module adds that second, more RAG-realistic failure mode: take a claim
that *is* supported by its context, and mutate exactly one fact in it.

Both corruption strategies are pure functions of the claim text — no
randomness — so the same input always produces the same corrupted output.
That's deliberate: reproducible training data beats a `seed` parameter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from doctor_rounds.classifier.types import FaithfulnessExample

# Auxiliary/copula verbs common enough in short factual claims that
# inserting "not" right after one reliably produces a grammatical
# negation. Not exhaustive — see `negate_claim`'s fallback for claims
# that use none of these.
_NEGATION_WORDS = frozenset(
    ["is", "are", "was", "were", "can", "does", "do", "has", "have", "will", "did", "should"]
)

_NUMBER_PATTERN = re.compile(r"\d+(\.\d+)?")


def negate_claim(claim: str) -> str:
    """Inserts "not" after the claim's first auxiliary/copula verb, e.g.
    "Metformin is first-line therapy." -> "Metformin is not first-line
    therapy." Falls back to wrapping the whole claim in "It is not true
    that ..." when none of `_NEGATION_WORDS` appears — less fluent, but
    still a genuine factual negation rather than skipping the claim.
    """
    stripped = claim.strip()
    if not stripped:
        return claim

    words = stripped.split(" ")
    for i, word in enumerate(words):
        if word.lower().strip(".,;:") in _NEGATION_WORDS:
            return " ".join([*words[: i + 1], "not", *words[i + 1 :]])

    trimmed = stripped.rstrip(". ")
    return f"It is not true that {trimmed[0].lower()}{trimmed[1:]}."


def perturb_number(claim: str) -> str | None:
    """Doubles the first number in `claim` (e.g. "32% of patients..." ->
    "64% of patients..."), or returns `None` if the claim has no number.

    A wrong statistic is one of the most common and most convincing kinds
    of RAG hallucination — the sentence stays fluent, so this is a
    meaningfully different (and arguably harder) negative example than
    `negate_claim`'s output.
    """
    match = _NUMBER_PATTERN.search(claim)
    if match is None:
        return None

    original = match.group()
    doubled = float(original) * 2 or 1  # doubling zero is a no-op, so use 1 instead
    replacement = str(int(doubled)) if original.isdigit() else f"{doubled:g}"
    return claim[: match.start()] + replacement + claim[match.end() :]


def corrupt_claim(claim: str) -> str:
    """Returns a factually altered version of `claim` for use as a
    synthetic negative example. Prefers `perturb_number` (closer to a
    realistic hallucination — see its docstring), falling back to
    `negate_claim` for claims with no number to perturb.
    """
    return perturb_number(claim) or negate_claim(claim)


def generate_synthetic_negatives(examples: Sequence[FaithfulnessExample]) -> list[FaithfulnessExample]:
    """Builds one synthetic negative `FaithfulnessExample` per supported
    example in `examples`, pairing each corrupted claim with its
    *original* context — the context no longer supports the corrupted
    claim, which is exactly the "generator ignored/misstated its
    retrieved evidence" failure this classifier is meant to catch.

    Already-negative input examples are skipped (corrupting a claim
    that's already unsupported doesn't teach the classifier anything new)
    — as is the rare case where corruption doesn't actually change the
    claim text, to avoid mislabeling an unchanged claim as unsupported.
    """
    negatives = []
    for ex in examples:
        if not ex.label:
            continue
        corrupted = corrupt_claim(ex.claim)
        if corrupted == ex.claim:
            continue
        negatives.append(
            FaithfulnessExample(claim=corrupted, context=ex.context, label=False, source="synthetic_corruption")
        )
    return negatives
