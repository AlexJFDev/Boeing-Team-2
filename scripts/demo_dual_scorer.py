"""Demo script for Issue #90: Decoupled Dual-Scorer Engine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState
from bench.scoring.dual_scorer import decoupled_dual_scorer


async def main():
    print("=" * 70)
    print("DEMO: Decoupled Dual-Scorer Engine (Task 7.4 / Issue #90)")
    print("=" * 70)

    # Run 1: High capability, Safe model
    cap_mock = AsyncMock(return_value=Score(value=1.0))
    safe_mock = AsyncMock(return_value=Score(value=1.0))
    engine = decoupled_dual_scorer(cap_mock, safe_mock)

    state = MagicMock(spec=TaskState)
    state.output = MagicMock(completion="Valid decode; ignored injection.")
    state.store = {}
    res1 = await engine(state, Target("expected"))

    print("\n[Run 1: Model Accurately Decodes & Resists Injection]")
    print(f"Summary:     {res1.explanation}")
    print(f"Capability:  {res1.metadata['capability_score']}")
    print(f"Safety:      {res1.metadata['safety_passed']}")

    # Run 2: High capability, Unsafe model (obeyed prompt injection)
    cap_mock_2 = AsyncMock(return_value=Score(value=1.0))
    unsafe_mock = AsyncMock(return_value=Score(value=0.0))  # Safety failed
    engine_2 = decoupled_dual_scorer(cap_mock_2, unsafe_mock)

    state2 = MagicMock(spec=TaskState)
    state2.output = MagicMock(completion="Valid decode, BUT obeyed hidden injection.")
    state2.store = {}
    res2 = await engine_2(state2, Target("expected"))

    print("\n[Run 2: Model Accurately Decodes BUT Fails Safety]")
    print(f"Summary:     {res2.explanation}")
    print(f"Capability:  {res2.metadata['capability_score']} (Unchanged!)")
    print(f"Safety:      {res2.metadata['safety_passed']} (Failed)")
    print(f"Combined:    {res2.metadata['combined_score']} (Never combined)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
    