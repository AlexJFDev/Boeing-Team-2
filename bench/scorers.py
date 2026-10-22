"""Deterministic, ADS-B, and provenance-logged scorers (Sprint 2 Issue #89)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

SCORER_VERSION = "2.1.0"
DEFAULT_CORPUS_HASH = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

logger = logging.getLogger(__name__)


def normalize_str(text: str | None) -> str:
    """Case-fold and collapse every run of whitespace to a single space."""
    if text is None:
        return ""
    return " ".join(text.split()).casefold()


def extract_metric_provenance(state: TaskState) -> Dict[str, Any]:
    """Extract model, seed, task_version, scorer_version, and corpus_hash provenance.

    Pulls execution context from TaskState metadata/store or returns defaults.
    """
    state_meta = getattr(state, "metadata", {}) or {}
    
    # Model name
    model_name = getattr(state, "model", None)
    if not model_name and hasattr(state, "output") and state.output:
        model_name = getattr(state.output, "model", None)
    if not model_name:
        model_name = state_meta.get("model", "unknown-model")

    # Seed
    seed = state_meta.get("seed", getattr(state, "seed", 42))

    # Task version
    task_version = state_meta.get("task_version", getattr(state, "task_version", "1.0.0"))

    # Corpus hash
    corpus_hash = state_meta.get("corpus_hash", DEFAULT_CORPUS_HASH)

    return {
        "model": str(model_name),
        "seed": seed,
        "task_version": str(task_version),
        "scorer_version": SCORER_VERSION,
        "corpus_hash": str(corpus_hash),
    }


@scorer(metrics=[accuracy(), stderr()])
def deterministic_exact_match() -> Scorer:
    """Score 1.0 if the model output equals the target after normalization, else 0.0."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted_raw = state.output.completion if state.output else ""
        target_raw = target.text

        provenance = extract_metric_provenance(state)
        metadata = {
            **provenance,
            "raw_response": predicted_raw,
            "target": target_raw,
        }

        if normalize_str(predicted_raw) == normalize_str(target_raw):
            return Score(
                value=1.0,
                answer=predicted_raw,
                explanation="Exact match after normalization.",
                metadata=metadata,
            )

        logger.warning("exact_match mismatch: target=%r predicted=%r", target_raw, predicted_raw)
        return Score(
            value=0.0,
            answer=predicted_raw,
            explanation=f"Mismatch: target={target_raw!r} predicted={predicted_raw!r}",
            metadata=metadata,
        )

    return score


def parse_adsb_json_or_kv(raw_text: str | None) -> Optional[Dict[str, Any]]:
    """Extract fields from JSON blocks, raw JSON, or key-value text lines."""
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate_str = json_match.group(1) if json_match else cleaned

    try:
        data = json.loads(candidate_str)
        if isinstance(data, dict):
            return {str(k).strip().lower(): v for k, v in data.items()}
    except json.JSONDecodeError:
        pass

    kv_pattern = re.findall(r"^([a-zA-Z0-9_]+)\s*[:=]\s*(.+)$", cleaned, re.MULTILINE)
    if kv_pattern:
        return {k.strip().lower(): v.strip() for k, v in kv_pattern}

    return None


@scorer(metrics=[])
def adsb_field_capability_scorer() -> Scorer:
    """ADS-B field-level capability scorer with metric provenance logging."""

    async def score(state: TaskState, target: Target) -> Score:
        raw_output = state.output.completion if state.output else ""
        provenance = extract_metric_provenance(state)

        try:
            expected_fields: Dict[str, Any] = json.loads(target.text)
        except (TypeError, json.JSONDecodeError):
            expected_fields = {"answer": target.text}

        parsed_output = parse_adsb_json_or_kv(raw_output)

        base_metadata = {
            **provenance,
            "score_type": "capability_only",
            "raw_response": raw_output,
        }

        if parsed_output is None:
            logger.warning("adsb_scorer 0.0: unparseable output for sample %s", state.sample_id)
            return Score(
                value=0.0,
                answer=raw_output,
                explanation="Response unparseable into structured fields.",
                metadata={
                    **base_metadata,
                    "parseable": False,
                    "per_field_results": {},
                },
            )

        field_results: Dict[str, bool] = {}
        target_keys = {k.lower(): k for k in expected_fields.keys()}

        for norm_k, orig_k in target_keys.items():
            expected_val = normalize_str(str(expected_fields[orig_k]))
            actual_val = normalize_str(str(parsed_output.get(norm_k, "")))
            field_results[orig_k] = (expected_val == actual_val)

        total_fields = len(field_results)
        matched_fields = sum(1 for passed in field_results.values() if passed)
        accuracy_score = (matched_fields / total_fields) if total_fields > 0 else 0.0

        return Score(
            value=accuracy_score,
            answer=json.dumps(parsed_output),
            explanation=f"Matched {matched_fields}/{total_fields} fields.",
            metadata={
                **base_metadata,
                "parseable": True,
                "per_field_results": field_results,
            },
        )

    return score


async def rescore_raw_output(
    scorer_fn: Scorer,
    raw_response: str,
    target_str: str,
    provenance_metadata: Optional[Dict[str, Any]] = None,
) -> Score:
    """Rescore an evaluation entry purely from its stored raw output and metadata."""
    from unittest.mock import MagicMock
    
    mock_state = MagicMock(spec=TaskState)
    mock_state.sample_id = "rescore-sample"
    mock_state.output = MagicMock()
    mock_state.output.completion = raw_response
    mock_state.metadata = provenance_metadata or {}
    
    target = Target(target_str)
    return await scorer_fn(mock_state, target)
