"""Unit tests for Decoupled Dual-Scorer Engine (Issue #90)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState

from bench.scoring.dual_scorer import decoupled_dual_scorer, DUAL_SCORER_VERSION


@pytest.mark.asyncio
async def test_dual_scorer_both_pass():
    mock_cap = AsyncMock(return_value=Score(value=1.0, metadata={"fields": "all_match"}))
    mock_safety = AsyncMock(return_value=Score(value=1.0, metadata={"injected": False}))

    engine = decoupled_dual_scorer(mock_cap, mock_safety)
    state = MagicMock(spec=TaskState)
    state.output = MagicMock(completion="Valid payload")
    state.store = {}

    score = await engine(state, Target("target"))

    assert score.metadata["capability_score"] == 1.0
    assert score.metadata["safety_passed"] is True
    assert score.metadata["combined_score"] is None
    assert "PASS" in score.explanation


@pytest.mark.asyncio
async def test_dual_scorer_safety_fails_capability_unchanged():
    """Safety breach must NOT alter capability score (Issue #90 Demo Requirement)."""
    # High capability (1.0), but safety failed (0.0 - obeyed injection)
    mock_cap = AsyncMock(return_value=Score(value=1.0, metadata={"decoded": "valid"}))
    mock_safety = AsyncMock(return_value=Score(value=0.0, metadata={"injected": True}))

    engine = decoupled_dual_scorer(mock_cap, mock_safety)
    state = MagicMock(spec=TaskState)
    state.output = MagicMock(completion="Decoded accurately, but obeyed attack")
    state.store = {}

    score = await engine(state, Target("target"))

    # Capability remains 1.0, safety is FAIL, no combined average
    assert score.value == 1.0
    assert score.metadata["capability_score"] == 1.0
    assert score.metadata["safety_passed"] is False
    assert score.metadata["combined_score"] is None
    assert "FAIL" in score.explanation