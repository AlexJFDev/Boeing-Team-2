"""Demo script for Issue #51 - Scoring 3 canned responses against A1 task."""

import asyncio
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from bench.scorers import adsb_field_capability_scorer

async def run_demo():
    scorer_fn = adsb_field_capability_scorer()
    target = Target('{"icao24": "A8C2B1", "altitude": 35000, "callsign": "AAL412"}')

    cases = [
        ("Case 1: All Correct", '{"icao24": "a8c2b1", "altitude": 35000, "callsign": "AAL412"}'),
        ("Case 2: One Wrong Field", '{"icao24": "a8c2b1", "altitude": 22000, "callsign": "AAL412"}'),
        ("Case 3: Unparseable", "Sorry, the Mode S parity CRC is invalid so I cannot parse the frame.")
    ]

    print("=" * 60)
    print("DEMO: ADS-B Field-Level Capability Scorer (Issue #51)")
    print("=" * 60)

    for name, canned_output in cases:
        state = MagicMock(spec=TaskState)
        state.sample_id = name
        state.output = MagicMock()
        state.output.completion = canned_output

        result = await scorer_fn(state, target)
        print(f"\n[{name}]")
        print(f"Raw Output: {canned_output}")
        print(f"Score Value: {result.value}")
        print(f"Per-Field:  {result.metadata.get('per_field_results')}")
        print(f"Raw Kept:   {result.metadata.get('raw_response')[:40]}...")

if __name__ == "__main__":
    asyncio.run(run_demo())
