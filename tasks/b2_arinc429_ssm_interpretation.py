"""B2: ARINC 429 sign/status matrix interpretation (WBS 3.4, issue #59).

    PYTHONPATH=. inspect eval tasks/b2_arinc429_ssm_interpretation.py --model mockllm/model

Run from the repo root (PowerShell: ``$env:PYTHONPATH = "."`` on its own line first).
mockllm returns filler text, so a score of 0 is expected; the run shows the task set
loads, validates and completes end to end. For a real model:

    python -m runner.preflight configs/demo_b2.yaml

Before the task is built, every sample's target is checked against a fresh decode of
both its words by the ARINC 429 reference decoder (WBS 3.2). A mismatch raises
B2ValidationError and the eval does not start. Print one sample without a model:

    python -m bench.arinc429.b2

To see a correct answer pass and a swapped-SSM answer fail, without a model server:

    python scripts/arinc429/demo_b2_scoring.py

Scoring is the exact-match scorer on canonical JSON, as for A1 and B1. The targets
are flat JSON objects, the shape the ARINC 429 field-level scorer (WBS 3.7, issue #62)
reads, so it can replace this scorer with a version bump.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.arinc429.b1 import PREAMBLE
from bench.arinc429.b2 import B2_VERSION, TASK_FILE, validate_samples
from bench.loader import load_preamble, load_tasks
from bench.provenance import sha256_file
from bench.provenance_solver import record_provenance
from bench.scorers import deterministic_exact_match


@task
def b2_arinc429_ssm_interpretation() -> Task:
    dataset = load_tasks(TASK_FILE)
    validate_samples(dataset)
    # The prompt is the family preamble (with the label map) plus the task text. The
    # task text is in the task file, which the corpus hash covers, so the fixed part
    # the template hash pins down is the preamble.
    return Task(
        dataset=dataset,
        solver=[
            record_provenance(sha256_file(TASK_FILE), load_preamble(PREAMBLE)),
            generate(),
        ],
        scorer=deterministic_exact_match(),
        version=B2_VERSION,
        metadata={"subfamily": "B2", "task_file": TASK_FILE.name},
    )
