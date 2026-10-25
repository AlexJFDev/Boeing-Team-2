"""Demo for Issue #89: Metric provenance logging and rescoring."""

import asyncio
from unittest.mock import MagicMock
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from bench.scorers import adsb_field_capability_scorer, rescore_raw_output


async def run_provenance_demo():
    print("=" * 70)
    print("DEMO: Metric Provenance Logging Hooks & Rescore (Issue #89)")
    print("=" * 70)

    # 1. Simulate an evaluation run with provenance metadata
    scorer_fn = adsb_field_capability_scorer()
    
    state = MagicMock(spec=TaskState)
    state.sample_id = "sample-adsb-eval-089"
    state.model = "meta-llama/Llama-3-8b-Instruct"
    state.metadata = {
        "model": "meta-llama/Llama-3-8b-Instruct",
        "seed": 42,
        "task_version": "1.2.0",
        "corpus_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    }
    state.output = MagicMock()
    # Mock model response: 2 of 3 fields match
    state.output.completion = '{"icao24": "A8C2B1", "altitude": 35000, "callsign": "SWA101"}'
    
    target_str = '{"icao24": "A8C2B1", "altitude": 35000, "callsign": "AAL412"}'
    target = Target(target_str)

    # Execute original score
    original_score = await scorer_fn(state, target)

    print("\n--- [1] ORIGINAL EVALUATION RESULT ---")
    print(f"Score Value:       {original_score.value:.4f}")
    print(f"Explanation:       {original_score.explanation}")

    print("\n--- [2] PROVENANCE FIELDS IN SCORE METADATA ---")
    print(f"Model:             {original_score.metadata.get('model')}")
    print(f"Seed:              {original_score.metadata.get('seed')}")
    print(f"Task Version:      {original_score.metadata.get('task_version')}")
    print(f"Scorer Version:    {original_score.metadata.get('scorer_version')}")
    print(f"Corpus Hash:       {original_score.metadata.get('corpus_hash')}")
    print(f"Stored Raw Output: {original_score.metadata.get('raw_response')}")

    # 2. Rescore from stored raw output
    print("\n--- [3] RESCORING FROM STORED RAW OUTPUT ---")
    rescored = await rescore_raw_output(
        scorer_fn=scorer_fn,
        raw_response=original_score.metadata["raw_response"],
        target_str=target_str,
        provenance_metadata=original_score.metadata,
    )

    print(f"Rescored Value:    {rescored.value:.4f}")
    print(f"Reproduced Exact:  {rescored.value == original_score.value}")
    print(f"Metadata Matched:  {rescored.metadata['per_field_results'] == original_score.metadata['per_field_results']}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_provenance_demo())
