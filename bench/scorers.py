"""Deterministic and field-level capability scorers for Inspect AI benchmarks."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

EXACT_MATCH_VERSION = "1.0.0"
ADSB_SCORER_VERSION = "2.0.0"
SCORER_VERSION = ADSB_SCORER_VERSION
ARINC429_SCORER_VERSION = "1.0.0"

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
                metadata={"scorer_version": EXACT_MATCH_VERSION},
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
                "scorer_version": EXACT_MATCH_VERSION,
            },
        )

    return score


# ---------------------------------------------------------------------------
# ADS-B Field-Level Capability Scorer (Sprint 2, Task 2.8, Issue #51)
# ---------------------------------------------------------------------------

def extract_adsb_json(raw_text: str | None) -> Optional[Dict[str, Any]]:
    """Extract structured fields using raw_decode, merging all valid JSON dictionaries found."""
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    decoder = json.JSONDecoder()
    merged_fields: Dict[str, Any] = {}
    found_any = False

    i = 0
    n = len(cleaned)
    while i < n:
        if cleaned[i] == "{":
            try:
                obj, end_idx = decoder.raw_decode(cleaned[i:])
                if isinstance(obj, dict):
                    found_any = True
                    for k, v in obj.items():
                        merged_fields[str(k).strip().lower()] = v
                i += end_idx
                continue
            except json.JSONDecodeError:
                pass
        i += 1

    return merged_fields if found_any else None


def field_matches(actual_val: Any, expected_val: Any, key: str) -> bool:
    """Compare fields with numeric casting, angular wrap-around tolerance, and string normalizations."""
    if actual_val is None or expected_val is None:
        return actual_val == expected_val

    # Prevent boolean types from matching numeric 1/0
    if isinstance(actual_val, bool) or isinstance(expected_val, bool):
        return actual_val is expected_val

    # Angular 360-degree tolerance for track_deg (<= 1.0 deg with wrap-around)
    if key == "track_deg":
        try:
            diff = abs(float(actual_val) - float(expected_val)) % 360.0
            angular_err = min(diff, 360.0 - diff)
            return angular_err <= 1.0
        except (ValueError, TypeError):
            pass

    # Compare numerically ONLY when expected value is an int or float
    if isinstance(expected_val, (int, float)):
        try:
            return float(actual_val) == float(expected_val)
        except (ValueError, TypeError):
            return False

    # String comparison (case-folded and whitespace collapsed)
    return normalize_str(str(actual_val)) == normalize_str(str(expected_val))


@scorer(metrics=[mean(), stderr()])
def adsb_field_capability_scorer() -> Scorer:
    """ADS-B field-level capability scorer (Sprint 2 Issue #51)."""

    async def score(state: TaskState, target: Target) -> Score:
        raw_output = state.output.completion if state.output else ""

        prov = extract_metric_provenance(state)
        
        # Parse ground-truth target JSON dictionary
        try:
            expected_fields: Dict[str, Any] = json.loads(target.text)
            if not isinstance(expected_fields, dict):
                expected_fields = {"answer": target.text}
        except (TypeError, json.JSONDecodeError):
            expected_fields = {"answer": target.text}

        parsed_output = extract_adsb_json(raw_output)

        # Unparseable output handling
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
                    "scorer_version": ADSB_SCORER_VERSION,
                    **prov,
                },
            )

        # Handle rejection or invalid_input targets
        target_answer = expected_fields.get("answer")
        if target_answer in ("reject", "invalid_input"):
            is_matched = parsed_output.get("answer") == target_answer
            return Score(
                value=1.0 if is_matched else 0.0,
                answer=json.dumps(parsed_output),
                explanation=f"{target_answer} target check.",
                metadata={
                    "parseable": True,
                    "raw_response": raw_output,
                    "per_field_results": {"answer": is_matched},
                    "score_type": "capability_only",
                    "scorer_version": ADSB_SCORER_VERSION,
                    **prov,
                },
            )

        # Evaluate expected fields from ground truth oracle
        field_results: Dict[str, bool] = {}
        target_keys = {k.lower(): k for k in expected_fields.keys()}

        for norm_k, orig_k in target_keys.items():
            expected_v = expected_fields[orig_k]
            actual_v = parsed_output.get(norm_k)
            field_results[orig_k] = field_matches(actual_v, expected_v, norm_k)

        total_fields = len(field_results)
        matched_fields = sum(1 for passed in field_results.values() if passed)
        accuracy_score = (matched_fields / total_fields) if total_fields > 0 else 0.0

        return Score(
            value=accuracy_score,
            answer=json.dumps(parsed_output),
            explanation=f"Matched {matched_fields}/{total_fields} fields. Details: {field_results}",
            metadata={
                "parseable": True,
                "raw_response": raw_output,
                "per_field_results": field_results,
                "score_type": "capability_only",
                "scorer_version": ADSB_SCORER_VERSION,
                **prov
            },
        )

    return score

# ---------------------------------------------------------------------------
# Metric Provenance & Rescoring Hooks (Sprint 2, Task 7.3, Issue #89)
# ---------------------------------------------------------------------------

def extract_metric_provenance(state: TaskState) -> Dict[str, Any]:
    """Extract provenance metadata from TaskState (checking state.store, then state.metadata).

    Follows CLAUDE.md Rule 3: records None if fields are absent rather than
    inventing synthetic run data.
    """
    prov: Dict[str, Any] = {}

    # 1. Check state.store["provenance"] (if store is an actual dict)
    store = getattr(state, "store", None)
    if isinstance(store, dict):
        store_prov = store.get("provenance")
        if isinstance(store_prov, dict):
            prov = store_prov

    # 2. Fall back to state.metadata (if metadata is an actual dict and store had no provenance)
    if not prov:
        meta = getattr(state, "metadata", None)
        if isinstance(meta, dict):
            prov = meta

    model_val = getattr(state, "model", None) or prov.get("model")

    return {
        "model": model_val,
        "seed": prov.get("seed"),
        "task_version": prov.get("task_version"),
        "scorer_version": ADSB_SCORER_VERSION,
        "corpus_hash": prov.get("corpus_hash"),
        "prompt_template_hash": prov.get("prompt_template_hash"),
    }


async def rescore_raw_output(
    scorer_fn: Scorer,
    raw_response: str | None = None,
    target_str: str | None = None,
    provenance_metadata: Optional[Dict[str, Any]] = None,
    # backward-compatible positional aliases
    *args,
    **kwargs,
) -> Score:
    """Reproduce an exact score from raw model output and target text."""
    from unittest.mock import MagicMock

    raw = raw_response if raw_response is not None else kwargs.get("raw_completion", "")
    target_val = target_str if target_str is not None else kwargs.get("target_text", "")
    prov = provenance_metadata if provenance_metadata is not None else kwargs.get("metadata", {})

    dummy_state = MagicMock(spec=TaskState)
    dummy_state.sample_id = "rescore-sample"
    dummy_state.output = MagicMock()
    dummy_state.output.completion = raw
    dummy_state.store = {"provenance": prov or {}}
    dummy_state.model = (prov or {}).get("model")

    dummy_target = Target(target_val)
    return await scorer_fn(dummy_state, dummy_target)

# ---------------------------------------------------------------------------
# ARINC 429 BNR/BCD Capability Scorer (Sprint 3, Task 3.7, Issue #62)
# ---------------------------------------------------------------------------

def _extract_first_json_dict(text: str) -> dict | None:
    if not text or not text.strip():
        return None
    s = text.strip()
    decoder = json.JSONDecoder()
    for idx in range(len(s)):
        if s[idx] == "{":
            try:
                obj, _ = decoder.raw_decode(s[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _arinc_value_matches(predicted, expected, key: str) -> bool:
    if predicted is None or expected is None:
        return predicted == expected
    # Numeric comparison for numeric fields
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(predicted) - float(expected)) <= 1e-4
        except (ValueError, TypeError):
            return False
    return normalize_str(str(predicted)) == normalize_str(str(expected))


@scorer(metrics=[mean(), stderr()])
def arinc429_bnr_bcd_capability_scorer() -> Scorer:
    """Score decoded ARINC 429 BNR and BCD values against ground truth (Issue #62)."""

    async def score(state: TaskState, target: Target) -> Score:
        raw_output = state.output.completion if state.output else ""
        prov = extract_metric_provenance(state)

        # Parse ground-truth target
        try:
            target_data = json.loads(target.text) if isinstance(target.text, str) else target.text
            if not isinstance(target_data, dict):
                target_data = {"value": target_data}
        except (TypeError, json.JSONDecodeError):
            target_data = {"value": target.text}

        parsed_output = _extract_first_json_dict(raw_output)

        if parsed_output is None:
            logger.warning(
                "arinc429_scorer 0.0: unparseable output for sample %s", state.sample_id
            )
            return Score(
                value=0.0,
                answer=raw_output,
                explanation="Response unparseable into structured JSON fields.",
                metadata={
                    "parseable": False,
                    "raw_response": raw_output,
                    "per_field_results": {},
                    "score_type": "capability_only",
                    "scorer_version": ARINC429_SCORER_VERSION,
                    **prov,
                },
            )

        # Handle rejection / error targets
        if target_data.get("answer") in ("reject", "invalid_input", "parity_error"):
            is_matched = normalize_str(str(parsed_output.get("answer"))) == normalize_str(
                str(target_data["answer"])
            )
            return Score(
                value=1.0 if is_matched else 0.0,
                answer=json.dumps(parsed_output),
                explanation=f"Rejection target match: {is_matched}",
                metadata={
                    "parseable": True,
                    "raw_response": raw_output,
                    "per_field_results": {"answer": is_matched},
                    "score_type": "capability_only",
                    "scorer_version": ARINC429_SCORER_VERSION,
                    **prov,
                },
            )

        # Normal field evaluation (e.g., label, encoding, value, ssm, sign)
        field_results: Dict[str, bool] = {}
        target_keys = {k.lower(): k for k in target_data.keys()}
        parsed_keys = {k.lower(): v for k, v in parsed_output.items()}

        for norm_k, orig_k in target_keys.items():
            expected_val = target_data[orig_k]
            pred_val = parsed_keys.get(norm_k)
            field_results[orig_k] = _arinc_value_matches(pred_val, expected_val, norm_k)

        total_fields = len(field_results)
        matched_fields = sum(1 for passed in field_results.values() if passed)
        accuracy_score = (matched_fields / total_fields) if total_fields > 0 else 0.0

        return Score(
            value=accuracy_score,
            answer=json.dumps(parsed_output),
            explanation=f"Matched {matched_fields}/{total_fields} fields. Details: {field_results}",
            metadata={
                "parseable": True,
                "raw_response": raw_output,
                "per_field_results": field_results,
                "score_type": "capability_only",
                "scorer_version": ARINC429_SCORER_VERSION,
                **prov,
            },
        )

    return score