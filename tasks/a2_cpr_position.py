"""A2: ADS-B CPR position decode (WBS 2.6, issue #49).

    PYTHONPATH=. inspect eval tasks/a2_cpr_position.py --model mockllm/model

Run from the repo root (PowerShell: ``$env:PYTHONPATH = "."`` first). mockllm
returns filler text, so a score of 0 is expected.

Before the task is built, every sample's target is re-derived from the CPR
verifier (bench/adsb/cpr.py). A mismatch raises A2ValidationError and the eval
does not start. Show a global, a local and a stale-pair sample without a model:

    python -m bench.adsb.a2

Scoring is the Sprint 1 exact-match scorer on canonical JSON with positions
rounded to 5 decimal places. The field-level ADS-B scorer (WBS 2.8) will replace
it with a distance tolerance using metadata["expected_position"].
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.adsb.a2 import A2_VERSION, PREAMBLE, TASK_FILE, validate_samples
from bench.loader import load_preamble, load_tasks
from bench.provenance import sha256_file
from bench.provenance_solver import record_provenance
from bench.scorers import deterministic_exact_match


@task
def a2_cpr_position() -> Task:
    dataset = load_tasks(TASK_FILE)
    validate_samples(dataset)
    # The task text and per-task premises live in the task file, which the corpus
    # hash covers; the fixed part of every prompt is the preamble.
    return Task(
        dataset=dataset,
        solver=[
            record_provenance(sha256_file(TASK_FILE), load_preamble(PREAMBLE)),
            generate(),
        ],
        scorer=deterministic_exact_match(),
        version=A2_VERSION,
        metadata={"subfamily": "A2", "task_file": TASK_FILE.name},
    )
