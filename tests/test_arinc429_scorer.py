"""Tests for ARINC 429 BNR/BCD capability scorer (Issue #62)."""

import pytest
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from bench.scorers import arinc429_bnr_bcd_capability_scorer


@pytest.mark.asyncio
async def test_arinc429_bnr_all_correct():
    scorer_fn = arinc429_bnr_bcd_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "bnr-1"
    state.output = MagicMock()
    state.output.completion = 'Decoded word: {"label": "310", "encoding": "BNR", "value": 142.5, "sign": "plus"}'
    
    target = Target('{"label": "310", "encoding": "BNR", "value": 142.5, "sign": "plus"}')
    score = await scorer_fn(state, target)
    assert score.value == 1.0
    assert score.metadata["parseable"] is True
    assert score.metadata["per_field_results"]["value"] is True
    assert score.metadata["per_field_results"]["encoding"] is True


@pytest.mark.asyncio
async def test_arinc429_bcd_partial_match():
    scorer_fn = arinc429_bnr_bcd_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "bcd-1"
    state.output = MagicMock()
    # Wrong value: 12500 vs target 13200
    state.output.completion = '{"label": "203", "encoding": "BCD", "value": 12500}'
    
    target = Target('{"label": "203", "encoding": "BCD", "value": 13200}')
    score = await scorer_fn(state, target)
    assert pytest.approx(score.value, 0.01) == 2 / 3
    assert score.metadata["per_field_results"]["value"] is False
    assert score.metadata["per_field_results"]["encoding"] is True


@pytest.mark.asyncio
async def test_arinc429_parity_rejection():
    scorer_fn = arinc429_bnr_bcd_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "err-1"
    state.output = MagicMock()
    state.output.completion = '{"answer": "parity_error", "note": "Bit 32 odd parity failed"}'
    
    target = Target('{"answer": "parity_error"}')
    score = await scorer_fn(state, target)
    assert score.value == 1.0
    assert score.metadata["per_field_results"]["answer"] is True


@pytest.mark.asyncio
async def test_arinc429_unparseable():
    scorer_fn = arinc429_bnr_bcd_capability_scorer()
    state = MagicMock(spec=TaskState)
    state.sample_id = "unparse-1"
    state.output = MagicMock()
    raw = "Unable to process the word bits."
    state.output.completion = raw
    
    target = Target('{"label": "310", "value": 100}')
    score = await scorer_fn(state, target)
    assert score.value == 0.0
    assert score.metadata["parseable"] is False
    assert score.metadata["raw_response"] == raw
