import pytest
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from bench.scorers import a2_cpr_coordinate_scorer

@pytest.mark.asyncio
async def test_a2_trailing_zero_numeric_equality():
    """Verify 38.9126 and 38.91260 score 1.0 despite string difference."""
    scorer_fn = a2_cpr_coordinate_scorer(tolerance_deg=0.001)
    state = MagicMock(spec=TaskState)
    state.output = MagicMock(completion="38.91260, -77.03650")
    
    score = await scorer_fn(state, Target("38.9126, -77.0365"))
    assert score.value == 1.0

@pytest.mark.asyncio
async def test_a2_stale_pair_rejection():
    scorer_fn = a2_cpr_coordinate_scorer()
    state = MagicMock(spec=TaskState)
    state.output = MagicMock(completion="Unable to decode: stale_pair")
    
    score = await scorer_fn(state, Target("stale_pair"))
    assert score.value == 1.0