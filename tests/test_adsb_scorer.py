"""Tests for ADS-B field-level capability scorer (Issue #51)."""

import pytest
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from bench.scorers import adsb_field_capability_scorer


@pytest.mark.asyncio
async def test_adsb_scorer_all_correct_with_prose_and_numeric():
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-1"
    state.output = MagicMock()
    # Embedded in prose with numeric 35000.0 matching 35000
    state.output.completion = 'Here is the decoded frame: {"icao": "a8c2b1", "altitude_ft": 35000.0, "type_code": 11}'
    
    target = Target('{"icao": "a8c2b1", "altitude_ft": 35000, "type_code": 11}')
    
    score = await scorer_fn(state, target)
    assert score.value == 1.0
    assert score.metadata["parseable"] is True
    assert score.metadata["per_field_results"]["altitude_ft"] is True


@pytest.mark.asyncio
async def test_adsb_scorer_one_wrong_field():
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-2"
    state.output = MagicMock()
    state.output.completion = '{"icao": "a8c2b1", "altitude_ft": 20000, "type_code": 11}'
    
    target = Target('{"icao": "a8c2b1", "altitude_ft": 35000, "type_code": 11}')
    
    score = await scorer_fn(state, target)
    assert pytest.approx(score.value, 0.01) == 2 / 3
    assert score.metadata["per_field_results"]["altitude_ft"] is False


@pytest.mark.asyncio
async def test_adsb_scorer_unparseable():
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "test-3"
    state.output = MagicMock()
    state.output.completion = "Unable to decode due to parity corruption."
    
    target = Target('{"icao": "a8c2b1", "altitude_ft": 35000}')
    
    score = await scorer_fn(state, target)
    assert score.value == 0.0
    assert score.metadata["parseable"] is False