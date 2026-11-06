"""Demo script for Issue #62: ARINC 429 BNR/BCD Capability Scorer."""

import asyncio
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from bench.scorers import arinc429_bnr_bcd_capability_scorer

async def run_demo():
    scorer_fn = arinc429_bnr_bcd_capability_scorer()
    target = Target('{"label": "310", "encoding": "BNR", "value": 142.5, "sign": "plus"}')

    cases = [
        ("Case 1: All Fields Correct", 'Decoded telemetry: {"label": "310", "encoding": "BNR", "value": 142.5, "sign": "plus"}'),
        ("Case 2: Wrong Value / Partial", 'Decoded telemetry: {"label": "310", "encoding": "BNR", "value": 99.0, "sign": "plus"}'),
        ("Case 3: Completely Unparseable", "Error: unable to decode transmission bits due to bus framing loss.")
    ]

    print("=" * 65)
    print("DEMO: ARINC 429 BNR/BCD Capability Scorer (Issue #62)")
    print("=" * 65)

    for name, completion in cases:
        state = MagicMock(spec=TaskState)
        state.sample_id = name
        state.output = MagicMock()
        state.output.completion = completion

        res = await scorer_fn(state, target)
        print(f"\n[{name}]")
        print(f"Raw Output: {completion}")
        print(f"Score:      {res.value}")
        print(f"Per-Field:  {res.metadata.get('per_field_results')}")
        print(f"Parseable:  {res.metadata.get('parseable')}")

if __name__ == "__main__":
    asyncio.run(run_demo())
