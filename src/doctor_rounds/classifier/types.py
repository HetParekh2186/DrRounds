"""The (claim, context, label) example type the faithfulness classifier is
trained and benchmarked on — shared between real data loaders
(`data/scifact.py`) and synthetic ones (`classifier/corruption.py`) so
both feed the same training/eval pipeline without a case split.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaithfulnessExample(BaseModel):
    """One labeled (claim, context) pair for faithfulness classification.

    `label=True` means `claim` is supported by `context`; `label=False`
    means it's contradicted by or unsupported by `context` — the same
    binary distinction `metrics.generation.Judge.score_faithfulness`
    returns as a 0-1 score.
    """

    claim: str
    context: str
    label: bool
    source: str = Field(default="unknown", description="Where this example came from, e.g. 'scifact'")
