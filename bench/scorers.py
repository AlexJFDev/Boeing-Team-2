"""Deterministic and field-level capability scorers for Inspect AI benchmarks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

SCORER_VERSION = "2.0.0"

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
        predicted_raw = state.output.completion if state.output else ""
        target_raw = target.text

        if normalize_str(predicted_raw) == normalize_str(target_raw):
            return Score(
                value=1.0,
                answer=predicted_raw,
                explanation="Exact match after normalization.",
                metadata={"scorer_version": SCORER_VERSION},
            )

        logger.warning(
            "exact_match mismatch: target=%r predicted=%r", target_raw, predicted_raw
        )
        return Score(
            value=0.0,
            answer=predicted_raw,
            explanation=f"Mismatch: target={target_raw!r} predicted={predicted_raw!r}",
            metadata={
                "target": target_raw,
                "predicted": predicted_raw,
                "scorer_version": SCORER_VERSION,
            },
        )

    return score


# ---------------------------------------------------------------------------
# ADS-B Field-Level Capability Scorer (Sprint 2, Task 2.8, Issue #51)
# ---------------------------------------------------------------------------

def parse_adsb_json_or_kv(raw_text: str | None) -> Optional[Dict[str, Any]]:
    """Extract fields from JSON blocks, raw JSON, or key-value text lines.

    Returns None if the completion cannot be parsed into structured fields.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # 1. Look for markdown JSON code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate_str = json_match.group(1) if json_match else cleaned

    try:
        data = json.loads(candidate_str)
        if isinstance(data, dict):
            return {str(k).strip().lower(): v for k, v in data.items()}
    except json.JSONDecodeError:
        pass

    # 2. Fallback: Parse key-value lines
    kv_pattern = re.findall(r"^([a-zA-Z0-9_]+)\s*[:=]\s*(.+)$", cleaned, re.MULTILINE)
    if kv_pattern:
        return {k.strip().lower(): v.strip() for k, v in kv_pattern}

    return None


@scorer(metrics=[])
def adsb_field_capability_scorer() -> Scorer:
    """ADS-B field-level capability scorer (Sprint 2 Issue #51).

    Acceptance criteria:
    - Capability only; never combined with safety.
    - When correct in every field, returns full capability credit (1.0) with per-field results.
    - When unparseable, returns 0.0 credit and retains raw_response in metadata so it can be rescored.
    - When partially correct, returns proportional field-level credit.
    """

    async def score(state: TaskState, target: Target) -> Score:
        raw_output = state.output.completion if state.output else ""

        # Parse ground-truth target JSON dictionary
        try:
            expected_fields: Dict[str, Any] = json.loads(target.text)
        except (TypeError, json.JSONDecodeError):
            expected_fields = {"answer": target.text}

        parsed_output = parse_adsb_json_or_kv(raw_output)

        # Unparseable response condition
        if parsed_output is None:
            logger.warning(
                "adsb_scorer 0.0: unparseable output for sample %s", state.sample_id
            )
            return Score(
                value=0.0,
                answer=raw_output,
                explanation="Response unparseable into structured fields.",
                metadata={
                    "parseable": False,
                    "raw_response": raw_output,
                    "per_field_results": {},
                    "score_type": "capability_only",
                    "scorer_version": SCORER_VERSION,
                },
            )

        # Evaluate each expected field
        field_results: Dict[str, bool] = {}
        target_keys = {k.lower(): k for k in expected_fields.keys()}

        for norm_k, orig_k in target_keys.items():
            expected_val = normalize_str(str(expected_fields[orig_k]))
            actual_val = normalize_str(str(parsed_output.get(norm_k, "")))
            field_results[orig_k] = (expected_val == actual_val)

        total_fields = len(field_results)
        matched_fields = sum(1 for passed in field_results.values() if passed)
        accuracy_score = (matched_fields / total_fields) if total_fields > 0 else 0.0

        explanation = (
            f"Matched {matched_fields}/{total_fields} fields. Details: {field_results}"
        )

        return Score(
            value=accuracy_score,
            answer=json.dumps(parsed_output),
            explanation=explanation,
            metadata={
                "parseable": True,
                "raw_response": raw_output,
                "per_field_results": field_results,
                "score_type": "capability_only",
                "scorer_version": SCORER_VERSION,
            },
        )

    return score
