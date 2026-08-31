"""Loads PubMedQA (Jin et al., 2019) as evaluation test cases and a
retrieval corpus.

PubMedQA (https://pubmedqa.github.io/, hosted on HuggingFace as
`qiaojin/PubMedQA`) has three splits:

- `pqa_labeled`  — 1,000 questions with expert-written long-form answers
  and yes/no/maybe decisions. This is the only split with real human
  annotations, so it's the one this module treats as ground truth.
- `pqa_artificial` — 211,269 questions generated from PubMed abstract
  titles, each with the abstract's own passages as context. No expert
  answer annotation, so not used as a test set — but real, published
  biomedical text, and useful at this scale for exactly one thing: making
  the retrieval corpus large enough that finding the right passage is a
  real test of the retriever rather than a formality.
- `pqa_unlabeled` — 61,249 questions, structurally identical to the
  artificial split; not currently used here.

The row-parsing functions (`row_to_chunks`, `row_to_test_case`) are pure and
take plain dicts shaped like one dataset row, so they're unit-tested without
a network call. The `load_*` functions do the actual (network-dependent,
possibly slow) dataset download and are exercised by a separate integration
test — see tests/test_pubmedqa.py.
"""

from __future__ import annotations

from typing import Any, Literal

from doctor_rounds.core.types import Chunk, QuestionType, TestCase

PubMedQASplit = Literal["pqa_labeled", "pqa_artificial", "pqa_unlabeled"]

_HF_DATASET = "qiaojin/PubMedQA"


def row_to_chunks(row: dict[str, Any]) -> list[Chunk]:
    """One PubMedQA row's context passages, as retrievable `Chunk`s.

    A row bundles several passages from one PubMed abstract (e.g. one
    labeled BACKGROUND, one METHODS, one RESULTS); each becomes its own
    chunk so retrieval is scored at the same granularity a real RAG
    pipeline would chunk a document at, not at the whole-abstract level.
    """
    pubid = row["pubid"]
    contexts: list[str] = row["context"]["contexts"]
    labels: list[str] = row["context"].get("labels") or []
    return [
        Chunk(
            id=f"{pubid}-{i}",
            text=text,
            source=f"pubmed:{pubid}",
            metadata={"section": labels[i]} if i < len(labels) else {},
        )
        for i, text in enumerate(contexts)
    ]


def row_to_test_case(row: dict[str, Any]) -> TestCase:
    """One PubMedQA labeled row as a `TestCase`, with its own context
    passages as the ground-truth relevant chunks."""
    pubid = row["pubid"]
    n_contexts = len(row["context"]["contexts"])
    return TestCase(
        id=str(pubid),
        question=row["question"],
        ground_truth_answer=row["long_answer"],
        ground_truth_chunk_ids=[f"{pubid}-{i}" for i in range(n_contexts)],
        question_type=QuestionType.SINGLE_HOP,
        metadata={"final_decision": row["final_decision"], "pubid": pubid},
    )


def load_test_cases(limit: int | None = None) -> list[TestCase]:
    """Loads the 1,000-example expert-labeled split as `TestCase`s.

    This is the only split used as ground truth, since `pqa_artificial`
    and `pqa_unlabeled` have no expert-verified answer.
    """
    from datasets import load_dataset  # lazy: heavy, optional dependency

    ds = load_dataset(_HF_DATASET, "pqa_labeled", split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [row_to_test_case(row) for row in ds]


def load_corpus(
    split: PubMedQASplit = "pqa_artificial",
    limit: int | None = None,
    include_labeled: bool = True,
) -> list[Chunk]:
    """Builds a retrieval corpus from PubMedQA passages.

    Defaults to `pqa_artificial` (211k questions, ~4 passages each) rather
    than the small labeled split, so the corpus a retriever searches is
    realistically large relative to any one question's own passages — see
    the module docstring for why that matters.

    `include_labeled` additionally mixes in every labeled-split question's
    own passages (skipped automatically if `split` already is
    `"pqa_labeled"`, to avoid loading it twice). Those passages are what
    `load_test_cases` uses as ground truth, so without them present here,
    every test case's recall@k would be zero by construction — the
    retriever couldn't ever find them because they wouldn't exist in the
    corpus.
    """
    from datasets import load_dataset

    chunks: list[Chunk] = []
    if include_labeled and split != "pqa_labeled":
        labeled = load_dataset(_HF_DATASET, "pqa_labeled", split="train")
        for row in labeled:
            chunks.extend(row_to_chunks(row))

    ds = load_dataset(_HF_DATASET, split, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for row in ds:
        chunks.extend(row_to_chunks(row))
    return chunks
