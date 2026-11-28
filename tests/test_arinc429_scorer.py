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


# --- 1.1.0: backlog 3.7 negative case, and B3 diagnostics -------------------------


def _score(completion: str, target: str) -> tuple[float, dict]:
    """(score value, metadata) for one completion."""
    import asyncio

    state = MagicMock(spec=TaskState)
    state.sample_id, state.metadata, state.store, state.model = "s", {}, {}, None
    state.output = MagicMock()
    state.output.completion = completion
    score = asyncio.run(arinc429_bnr_bcd_capability_scorer()(state, Target(target)))
    assert score is not None and score.metadata is not None
    return float(score.value), score.metadata  # type: ignore[arg-type]


B1_TARGET = '{"label":"203","sdi":1,"ssm":"11","value":1000}'


def test_wrong_ssm_scores_the_value_zero_regardless_of_magnitude() -> None:
    # Backlog 3.7: "When the sign or SSM is wrong, the value scores 0 regardless of magnitude."
    value, meta = _score('{"label":"203","sdi":1,"ssm":"00","value":1000}', B1_TARGET)
    assert meta["per_field_results"] == {"label": True, "sdi": True, "ssm": False, "value": False}
    assert value == 0.5
    assert meta["value_zeroed_by"] == ["ssm"]


def test_wrong_sign_scores_the_value_zero() -> None:
    target = '{"label": "310", "encoding": "BNR", "value": 142.5, "sign": "plus"}'
    value, meta = _score('{"label": "310", "encoding": "BNR", "value": 142.5, "sign": "minus"}', target)
    assert meta["per_field_results"]["value"] is False
    assert meta["value_zeroed_by"] == ["sign"]
    assert value == 0.5


def test_a_signed_value_with_the_wrong_sign_is_already_wrong() -> None:
    _, meta = _score('{"label":"203","sdi":1,"ssm":"11","value":-1000}', B1_TARGET)
    assert meta["per_field_results"]["value"] is False
    assert "value_zeroed_by" not in meta  # the magnitude test caught it


def test_right_ssm_and_value_score_one_with_no_gate() -> None:
    value, meta = _score(B1_TARGET, B1_TARGET)
    assert value == 1.0 and "value_zeroed_by" not in meta


B3_TARGET = '{"violations":[[5,"word_timing"],[13,"rate_fast"],[15,"rate_fast"]]}'


def test_b3_score_is_still_the_whole_list() -> None:
    assert _score(B3_TARGET, B3_TARGET)[0] == 1.0
    assert _score('{"violations":[[5,"word_timing"],[13,"rate_fast"],[9,"gap"]]}', B3_TARGET)[0] == 0.0


def test_b3_records_precision_and_recall_without_changing_the_score() -> None:
    value, meta = _score('Found: {"violations":[[5,"word_timing"],[13,"Rate_Fast"],[9,"gap"]]}', B3_TARGET)
    assert value == 0.0
    assert meta["violation_overlap"] == {
        "parseable": True, "expected": 3, "reported": 3, "found": 2, "missed": 1, "false_alarms": 1,
        "precision": 2 / 3, "recall": 2 / 3}
    value, meta = _score('{"violations":[]}', '{"violations":[]}')
    assert value == 1.0
    assert meta["violation_overlap"]["precision"] is None and meta["violation_overlap"]["recall"] is None
    _, meta = _score('{"violations":"none"}', B3_TARGET)
    assert meta["violation_overlap"] == {"parseable": False, "expected": 3}


def test_non_b3_targets_carry_no_overlap() -> None:
    assert "violation_overlap" not in _score(B1_TARGET, B1_TARGET)[1]
