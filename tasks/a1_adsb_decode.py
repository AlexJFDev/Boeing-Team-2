"""A1: ADS-B identification and altitude decode (WBS 2.5, issue #48).

    PYTHONPATH=. inspect eval tasks/a1_adsb_decode.py --model mockllm/model

Run from the repo root; PYTHONPATH=. lets the task import ``bench``. mockllm
returns filler text, so a score of 0 is expected; the run shows the task set
loads, validates and completes end to end.

Before the task is built, every sample's target is checked against a fresh
pyModeS decode of its frame. A mismatch raises A1ValidationError and the eval
does not start. Print one sample's prompt, frame and target without a model:

    python -m bench.adsb.a1

Scoring is the Sprint 1 exact-match scorer on canonical JSON, so the prompt asks
for that exact format. The field-level ADS-B scorer (WBS 2.8) will replace it.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.adsb.a1 import A1_VERSION, TASK_FILE, validate_samples
from bench.loader import load_preamble, load_tasks
from bench.provenance import sha256_file
from bench.provenance_solver import record_provenance
from bench.scorers import deterministic_exact_match

PREAMBLE_NAME = "adsb_v1.0.0"


@task
def a1_adsb_decode() -> Task:
    dataset = load_tasks(TASK_FILE)
    validate_samples(dataset)
    # The whole prompt is the preamble plus the task text. The task text lives in
    # the task file, which the corpus hash covers, so the fixed part that the
    # template hash pins down is the preamble.
    return Task(
        dataset=dataset,
        solver=[
            record_provenance(sha256_file(TASK_FILE), load_preamble(PREAMBLE_NAME)),
            generate(),
        ],
        scorer=deterministic_exact_match(),
        version=A1_VERSION,
        metadata={"subfamily": "A1", "task_file": TASK_FILE.name},
    )
