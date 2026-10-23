"""Unit tests for deterministic exact-match scorer (Issue #17)."""

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from bench.scorers import deterministic_exact_match, normalize_str


def test_normalize_str():
    assert normalize_str("  DF17  ") == "df17"
    assert normalize_str("ARINC   429\n") == "arinc 429"
    assert normalize_str(None) == ""


@pytest.mark.asyncio
async def test_scorer_exact_match_positive():
    scorer_fn = deterministic_exact_match()
    state = TaskState(
        model="mockllm/model",
        sample_id=1,
        epoch=1,
        input="Decode frame",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content="  DF17 \n"),
    )
    target = Target("df17")

    result = await scorer_fn(state, target)
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_scorer_mismatch_negative_logs(caplog):
    scorer_fn = deterministic_exact_match()
    state = TaskState(
        model="mockllm/model",
        sample_id=2,
        epoch=1,
        input="Decode frame",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content="DF18"),
    )
    target = Target("DF17")

    with caplog.at_level("WARNING"):
        result = await scorer_fn(state, target)

    assert result.value == 0.0
    assert "target='DF17'" in caplog.text
    assert "predicted='DF18'" in caplog.text