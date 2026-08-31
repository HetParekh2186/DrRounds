"""Synthetic test-set generation: turn a corpus of chunks nobody has
written questions for yet into `TestCase`s an evaluation run can use.

This is what makes Doctor Rounds usable against *your* corpus, not just
PubMedQA's pre-labeled questions — most real RAG pipelines are built over
documents that don't come with a labeled QA set attached. An `LLM` reads
one chunk at a time and is asked to write a question genuinely answerable
from it plus a ground-truth answer grounded in nothing else; the chunk's
own id becomes the ground-truth chunk id, so the resulting `TestCase`
plugs straight into `core.runner.run_evaluation`.

v1 only generates `QuestionType.SINGLE_HOP` cases (one chunk in, one
question out). Multi-hop generation (synthesizing a question that requires
two or more chunks together) is real future work, not a corner cut here —
it needs a chunk-pairing strategy this module doesn't have yet, tracked in
the repo roadmap rather than half-built.
"""

from __future__ import annotations

import random
import re
import uuid
from collections.abc import Sequence

from doctor_rounds.adapters.llm import LLM
from doctor_rounds.core.types import Chunk, QuestionType, TestCase

_GENERATION_PROMPT = """You are writing a test question for evaluating a medical question-answering \
system, the way an attending might quiz a resident on a single chart note.

Given the passage below, write ONE question that:
- can be answered using only the information in this passage (don't assume outside knowledge)
- has a clear, factual answer directly supported by the passage
- is phrased the way a clinician might actually ask it, not "what does the passage say about..."

Passage:
{passage}

Respond in exactly this format, nothing else:
QUESTION: <the question>
ANSWER: <the answer, 1-3 sentences, grounded only in the passage above>"""

_RESPONSE_PATTERN = re.compile(r"QUESTION:\s*(.+?)\s*\n+ANSWER:\s*(.+)", re.DOTALL | re.IGNORECASE)


def _parse_response(raw: str) -> tuple[str, str] | None:
    """Pulls (question, answer) out of a generation LLM's response.

    Returns `None` — rather than raising — on a malformed response: this
    is called once per chunk over a whole corpus, and one LLM hiccup
    shouldn't take down the rest of the batch. `generate_test_set` skips
    chunks that fail to parse.
    """
    match = _RESPONSE_PATTERN.search(raw)
    if not match:
        return None
    question, answer = match.group(1).strip(), match.group(2).strip()
    if not question or not answer:
        return None
    return question, answer


def generate_single_hop_test_case(chunk: Chunk, llm: LLM) -> TestCase | None:
    """Generates one `TestCase` answerable from `chunk` alone.

    Returns `None` if the LLM's response couldn't be parsed into a
    question/answer pair — the caller decides whether to retry or skip.
    """
    if not chunk.text.strip():
        raise ValueError(f"Chunk {chunk.id!r} has no text to generate a question from")

    parsed = _parse_response(llm.generate(_GENERATION_PROMPT.format(passage=chunk.text)))
    if parsed is None:
        return None
    question, answer = parsed

    return TestCase(
        id=f"synth-{uuid.uuid4().hex[:12]}",
        question=question,
        ground_truth_answer=answer,
        ground_truth_chunk_ids=[chunk.id],
        question_type=QuestionType.SINGLE_HOP,
        metadata={"generator": "llm_single_hop", "source_chunk_id": chunk.id},
    )


def generate_test_set(
    chunks: Sequence[Chunk],
    llm: LLM,
    *,
    n: int,
    seed: int | None = None,
) -> list[TestCase]:
    """Samples up to `n` chunks and generates one `TestCase` per chunk.

    Sampling (not just taking the first `n`) matters for a corpus that's
    grouped or ordered by document — an unsampled prefix would skew the
    resulting test set toward whichever document happens to sort first.
    `seed` makes that sampling reproducible; the LLM's own output is only
    as deterministic as the LLM itself.

    The returned list can be shorter than `n`: chunks whose generation
    didn't parse are skipped rather than retried or substituted, so a
    batch of bad LLM output doesn't silently swap in different chunks
    than the ones actually sampled.
    """
    if n <= 0 or not chunks:
        return []

    sample = random.Random(seed).sample(list(chunks), k=min(n, len(chunks)))

    test_cases = []
    for chunk in sample:
        test_case = generate_single_hop_test_case(chunk, llm)
        if test_case is not None:
            test_cases.append(test_case)
    return test_cases
