"""Mock task that exercises the EvalLog provenance fields (issue #16).

Runs with the built-in mock model. Run it from the repo root so `bench` imports:

    PYTHONPATH=. inspect eval tasks/provenance_mock.py --model mockllm/model
"""

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, prompt_template

from bench.provenance import sha256_text
from bench.provenance_solver import record_provenance

PROMPT_TEMPLATE = "Reply with the single word: {prompt}"
SAMPLES = [Sample(input="ready", target="ready")]


@task
def provenance_mock():
    # The mock corpus is the inline samples. A real task would use
    # corpus_hash_from_manifest("data/adsb/MANIFEST.json") instead.
    corpus = json.dumps(
        [{"input": sample.input, "target": sample.target} for sample in SAMPLES],
        sort_keys=True,
    )
    return Task(
        dataset=SAMPLES,
        solver=[
            record_provenance(sha256_text(corpus), PROMPT_TEMPLATE),
            prompt_template(PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=includes(),
    )