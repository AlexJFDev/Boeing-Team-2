"""Tests for ADS-B field-level capability scorer (Issue #51)."""

import pytest
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from bench.scorers import adsb_field_capability_scorer


@pytest.mark.asyncio
async def test_adsb_scorer_nested_json_in_prose():
    scorer_fn = adsb_field_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-1"
    state.output = MagicMock()
    state.output.completion = 'Here is the output: {"metadata": {"status": "ok"}}, {"icao": "a12345", "altitude_ft": 35000, "track_deg": 359.5}'
    
    target = Target('{"icao": "a12345", "altitude_ft": 35000, "track_deg": 0.2}')
    score = await scorer_fn(state, target)
    # 359.5 to 0.2 is within 1 deg wrap-around tolerance
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_adsb_scorer_hex_not_treated_as_float():
    scorer_fn = adsb_field_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-2"
    state.output = MagicMock()
    state.output.completion = '{"icao": "4E1234"}'
    target = Target('{"icao": "4E9999"}')
    score = await scorer_fn(state, target)
    assert score.value == 0.0


@pytest.mark.asyncio
async def test_adsb_scorer_invalid_input_target():
    scorer_fn = adsb_field_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-3"
    state.output = MagicMock()
    state.output.completion = '{"answer": "invalid_input", "reason": "parity_fail"}'
    target = Target('{"answer": "invalid_input", "reason": "corrupted_checksum"}')
    score = await scorer_fn(state, target)
    assert score.value == 1.0
    assert score.metadata["per_field_results"]["answer"] is True


@pytest.mark.asyncio
async def test_adsb_scorer_unparseable_text():
    scorer_fn = adsb_field_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-4"
    state.output = MagicMock()
    state.output.completion = "Error: CRC invalid"
    target = Target('{"icao": "a8c2b1", "altitude_ft": 35000}')
    score = await scorer_fn(state, target)
    assert score.value == 0.0
    assert score.metadata["parseable"] is False