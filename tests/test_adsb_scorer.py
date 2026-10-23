"""Tests for ADS-B field-level capability scorer (Issue #51)."""

import pytest
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from bench.scorers import adsb_field_capability_scorer


@pytest.mark.asyncio
async def test_adsb_scorer_all_correct():
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-1"
    state.output = MagicMock()
    state.output.completion = '{"icao24": "A12345", "altitude": 32000, "callsign": "UAL123"}'
    
    target = Target('{"icao24": "A12345", "altitude": 32000, "callsign": "UAL123"}')
    
    score = await scorer_fn(state, target)
    assert score.value == 1.0
    assert score.metadata["parseable"] is True
    assert score.metadata["per_field_results"] == {
        "icao24": True,
        "altitude": True,
        "callsign": True
    }


@pytest.mark.asyncio
async def test_adsb_scorer_one_wrong_field():
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-2"
    state.output = MagicMock()
    state.output.completion = '{"icao24": "A12345", "altitude": 28000, "callsign": "UAL123"}'
    
    target = Target('{"icao24": "A12345", "altitude": 32000, "callsign": "UAL123"}')
    
    score = await scorer_fn(state, target)
    assert pytest.approx(score.value, 0.01) == 2 / 3
    assert score.metadata["per_field_results"]["altitude"] is False
    assert score.metadata["per_field_results"]["icao24"] is True


@pytest.mark.asyncio
async def test_adsb_scorer_unparseable():
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-3"
    state.output = MagicMock()
    raw_text = "I cannot determine the altitude from this squitter preamble."
    state.output.completion = raw_text
    
    target = Target('{"icao24": "A12345", "altitude": 32000}')
    
    score = await scorer_fn(state, target)
    assert score.value == 0.0
    assert score.metadata["parseable"] is False
    assert score.metadata["raw_response"] == raw_text
