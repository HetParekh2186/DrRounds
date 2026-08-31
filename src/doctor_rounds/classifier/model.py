"""Loads a fine-tuned local faithfulness classifier and exposes it as
faithfulness scoring compatible with `metrics.generation.Judge`.

Requires the "classifier" extra (torch + transformers) — a genuinely
heavy, GPU-shaped dependency this project deliberately keeps out of the
default install and default CI matrix (see the `dev` extra's comment in
pyproject.toml). Only import this module where you actually use it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from doctor_rounds.metrics.generation import Judge


class LocalFaithfulnessClassifier:
    """Scores faithfulness with a locally fine-tuned sequence-pair
    classifier instead of an LLM judge call.

    `model_dir` is a directory produced by
    `scripts/train_faithfulness_classifier.py` (or any
    `AutoModelForSequenceClassification`-compatible checkpoint trained the
    same way: label 1 = "claim is supported by context", matching
    `classifier.types.FaithfulnessExample.label`).
    """

    def __init__(self, model_dir: str | Path, *, max_length: int = 384, device: str | None = None) -> None:
        self.model_dir = str(model_dir)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def score_faithfulness(self, answer: str, context: str) -> float:
        """Probability that `answer` is supported by `context`, per the
        fine-tuned classifier — a plain forward pass, no generation."""
        inputs = self.tokenizer(
            answer, context, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())


class ClassifierJudge:
    """A full `Judge`: faithfulness from the local classifier (fast,
    offline, no per-call API cost), relevance from an LLM judge.

    The classifier isn't trained for relevance — it only ever sees
    (claim, context) pairs, not (answer, question) pairs, a different
    comparison — so relevance is delegated rather than approximated.
    """

    def __init__(self, classifier: LocalFaithfulnessClassifier, relevance_judge: Judge) -> None:
        self.classifier = classifier
        self.relevance_judge = relevance_judge

    def score_faithfulness(self, answer: str, context: str) -> float:
        return self.classifier.score_faithfulness(answer, context)

    def score_relevance(self, answer: str, question: str) -> float:
        return self.relevance_judge.score_relevance(answer, question)
