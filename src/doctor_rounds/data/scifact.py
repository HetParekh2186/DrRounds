"""Real SciFact claim-verification data, loaded from HuggingFace, as
training/eval data for the faithfulness classifier.

SciFact (Wadden et al., 2020) pairs scientific claims with cited
abstracts, each annotated SUPPORT, CONTRADICT, or left unlabeled when the
cited document doesn't verify the claim either way. That's exactly the
(claim, context, label) shape faithfulness classification needs — a
"claim" here plays the role a RAG answer's assertion would, and the
abstract plays the role of retrieved context — so unlike PubMedQA (which
this project also uses, for retrieval), no reframing is required.

`allenai/scifact`'s Hub loading script predates the `datasets` library
dropping script-based datasets, so this reads the auto-converted parquet
files directly (`refs/convert/parquet`) via `pandas`/`huggingface_hub`
rather than `datasets.load_dataset` — see the module-level constants for
the exact paths.
"""

from __future__ import annotations

from typing import Any, Literal

from doctor_rounds.classifier.types import FaithfulnessExample

_HF_REPO = "allenai/scifact"
_HF_REVISION = "refs%2Fconvert%2Fparquet"  # url-encoded "refs/convert/parquet"

# SUPPORT/CONTRADICT rows have sentence-level evidence and are what this
# project trains/evaluates on; empty-label rows are claims SciFact's own
# annotators could not verify against the cited document either way —
# genuinely unlabeled, not a third class, so they're dropped rather than
# mapped to `label=False` (that would teach the classifier "unverifiable"
# and "contradicted" are the same signal, which they aren't).
_VERIFIABLE_LABELS = {"SUPPORT", "CONTRADICT"}


def _corpus_lookup(corpus_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Maps SciFact doc_id -> its abstract as one joined string.

    Pure and fixture-tested: takes plain rows (as `pandas.DataFrame.to_dict("records")`
    would produce), not a live dataset object.
    """
    return {str(row["doc_id"]): " ".join(row["abstract"]) for row in corpus_rows}


def row_to_example(claim_row: dict[str, Any], corpus_by_id: dict[str, str]) -> FaithfulnessExample | None:
    """Builds one `FaithfulnessExample` from a SciFact claim row, or
    returns `None` for a row this project doesn't train/eval on: an
    unverifiable claim (see `_VERIFIABLE_LABELS`), or — defensively, since
    real data occasionally has dangling references — a claim whose cited
    document isn't present in the corpus lookup.
    """
    label = claim_row.get("evidence_label")
    if label not in _VERIFIABLE_LABELS:
        return None

    context = corpus_by_id.get(str(claim_row.get("evidence_doc_id", "")))
    if context is None:
        return None

    return FaithfulnessExample(
        claim=claim_row["claim"],
        context=context,
        label=(label == "SUPPORT"),
        source="scifact",
    )


def load_examples(split: Literal["train", "validation"]) -> list[FaithfulnessExample]:
    """Loads real SciFact (claim, context, label) examples for `split`.

    Only "train" and "validation" are accepted: SciFact's "test" split
    (used for a since-closed leaderboard) has no public labels, so it's
    useless for training or benchmarking here — see
    https://huggingface.co/datasets/allenai/scifact.
    """
    if split not in ("train", "validation"):
        raise ValueError(
            f"Unsupported SciFact split: {split!r} (expected 'train' or 'validation' — "
            "'test' has no public labels, see this function's docstring)"
        )

    import pandas as pd

    claims = pd.read_parquet(
        f"hf://datasets/{_HF_REPO}@{_HF_REVISION}/claims/{split}/0000.parquet"
    ).to_dict("records")
    corpus_rows = pd.read_parquet(
        f"hf://datasets/{_HF_REPO}@{_HF_REVISION}/corpus/train/0000.parquet"
    ).to_dict("records")
    corpus_by_id = _corpus_lookup(corpus_rows)

    examples = (row_to_example(row, corpus_by_id) for row in claims)
    return [ex for ex in examples if ex is not None]
