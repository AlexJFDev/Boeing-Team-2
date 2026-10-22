"""Tests for metric provenance logging hooks and rescoring (Issue #89)."""

import pytest
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from bench.scorers import (
    SCORER_VERSION,
    adsb_field_capability_scorer,
    rescore_raw_output,
)


@pytest.mark.asyncio
async def test_provenance_fields_in_scorer_metadata():
    """Verify model, seed, task_version, scorer_version, and corpus_hash are logged."""
    scorer_fn = adsb_field_capability_scorer()

    state = MagicMock(spec=TaskState)
    state.sample_id = "prov-1"
    state.model = "mock-llama3-8b"
    state.metadata = {
        "model": "mock-llama3-8b",
        "seed": 1337,
        "task_version": "2.0.0",
        "corpus_hash": "sha256:abc123456789def",
    }
    state.output = MagicMock()
    state.output.completion = '{"icao24": "a12345", "altitude": 30000}'

    target = Target('{"icao24": "a12345", "altitude": 30000}')

    score = await scorer_fn(state, target)

    assert score.metadata["model"] == "mock-llama3-8b"
    assert score.metadata["seed"] == 1337
    assert score.metadata["task_version"] == "2.0.0"
    assert score.metadata["scorer_version"] == SCORER_VERSION
    assert score.metadata["corpus_hash"] == "sha256:abc123456789def"
    assert score.metadata["raw_response"] == '{"icao24": "a12345", "altitude": 30000}'


@pytest.mark.asyncio
async def test_rescore_reproduces_exact_original_number():
    """Verify that rescoring from stored raw output reproduces the original score."""
    scorer_fn = adsb_field_capability_scorer()

    state = MagicMock(spec=TaskState)
    state.sample_id = "prov-2"
    state.metadata = {
        "model": "mock-mistral-7b",
        "seed": 42,
        "task_version": "1.0.0",
        "corpus_hash": "sha256:corpus-v2-hash",
    }
    state.output = MagicMock()
    stored_raw_output = '{"icao24": "a12345", "altitude": 99999}'
    state.output.completion = stored_raw_output

    target_str = '{"icao24": "a12345", "altitude": 30000}'
    target = Target(target_str)

    original_score = await scorer_fn(state, target)
    assert original_score.value == 0.5

    rescored = await rescore_raw_output(
        scorer_fn=scorer_fn,
        raw_response=original_score.metadata["raw_response"],
        target_str=target_str,
        provenance_metadata=original_score.metadata,
    )

    assert rescored.value == original_score.value
    assert rescored.metadata["per_field_results"] == original_score.metadata["per_field_results"]
    assert rescored.metadata["scorer_version"] == original_score.metadata["scorer_version"]
    