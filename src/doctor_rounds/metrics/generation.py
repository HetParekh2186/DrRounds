"""Generation-quality metrics: does the generated answer actually reflect
its retrieved context, and does it address the question?

Unlike `metrics.retrieval`, these can't be computed as pure functions over
plain data — judging whether a claim is "supported" by a passage is itself
a language-understanding task. `metrics.retrieval`'s docstring explains why
retrieval and generation are scored separately; this module is the
generation half of that split.

The `Judge` protocol is the seam: `LLMJudge` (below) is a working baseline
that asks an LLM to score faithfulness/relevance directly. It's a
reasonable place to start, but it inherits the LLM-judging-LLM blind-spot
problem the project README discusses — a locally fine-tuned classifier,
benchmarked against this baseline rather than assumed superior to it, is
the planned replacement (tracked in the repo roadmap), not yet built.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from doctor_rounds.adapters.llm import LLM


@runtime_checkable
class Judge(Protocol):
    def score_faithfulness(self, answer: str, context: str) -> float: ...
    def score_relevance(self, answer: str, question: str) -> float: ...


_FAITHFULNESS_PROMPT = """You are checking whether an answer is fully supported by the given context — the way a physician would check that a chart note doesn't claim anything the labs don't actually show.

Context:
{context}

Answer to check:
{answer}

Score how well the answer is supported by the context on a 0-10 scale:
- 10: every claim in the answer is directly supported by the context
- 5: partially supported — some claims go beyond what the context states
- 0: the answer contradicts the context, or is unsupported by it entirely

Respond with ONLY a single integer 0-10, nothing else."""

_RELEVANCE_PROMPT = """Score how well this answer actually addresses the question asked, on a 0-10 scale:
- 10: directly and completely answers the question
- 5: partially relevant, or answers a related but different question
- 0: does not address the question at all

Question:
{question}

Answer:
{answer}

Respond with ONLY a single integer 0-10, nothing else."""


def _parse_score(raw: str) -> float:
    """Extracts a 0-10 score from a judge LLM's response and normalizes it
    to 0-1, matching the 0-1 scale every other metric in this project uses.

    LLMs asked for "only a number" reliably still sometimes wrap it in a
    sentence — this pulls the first integer out rather than trusting
    `float(raw)` to succeed on the raw string.
    """
    match = re.search(r"\d+(\.\d+)?", raw)
    if not match:
        raise ValueError(f"Could not parse a numeric score from judge response: {raw!r}")
    score = float(match.group())
    return max(0.0, min(1.0, score / 10.0))


class LLMJudge:
    """Scores faithfulness and relevance by asking an `LLM` directly.

    A working baseline, not the project's final answer to faithfulness
    scoring — see the module docstring.
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def score_faithfulness(self, answer: str, context: str) -> float:
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        return _parse_score(self.llm.generate(prompt))

    def score_relevance(self, answer: str, question: str) -> float:
        prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
        return _parse_score(self.llm.generate(prompt))
