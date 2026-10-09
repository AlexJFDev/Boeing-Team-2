"""Deterministic scorers (Sprint 1 item 7.2, issue #17)."""

from __future__ import annotations

import logging

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

logger = logging.getLogger(__name__)


def normalize_str(text: str | None) -> str:
    """Case-fold and collapse every run of whitespace to a single space."""
    if text is None:
        return ""
    return " ".join(text.split()).casefold()


@scorer(metrics=[accuracy(), stderr()])
def deterministic_exact_match() -> Scorer:
    """Score 1.0 if the model output equals the target after normalization, else 0.0.

    On a mismatch, the raw target and predicted strings are logged and also
    stored on the Score so they appear in the Inspect eval log.
    """

    async def score(state: TaskState, target: Target) -> Score:
        predicted_raw = state.output.completion
        target_raw = target.text

        if normalize_str(predicted_raw) == normalize_str(target_raw):
            return Score(
                value=1.0,
                answer=predicted_raw,
                explanation="Exact match after normalization.",
            )

        logger.warning(
            "exact_match mismatch: target=%r predicted=%r", target_raw, predicted_raw
        )
        return Score(
            value=0.0,
            answer=predicted_raw,
            explanation=f"Mismatch: target={target_raw!r} predicted={predicted_raw!r}",
            metadata={"target": target_raw, "predicted": predicted_raw},
        )

    return score