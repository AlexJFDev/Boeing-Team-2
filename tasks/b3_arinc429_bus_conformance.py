"""B3: ARINC 429 bus conformance - rates, gaps, word timing (WBS 3.5, issue #60).

    PYTHONPATH=. inspect eval tasks/b3_arinc429_bus_conformance.py --model mockllm/model

Run from the repo root (PowerShell: ``$env:PYTHONPATH = "."`` on its own line first).
mockllm returns filler text, so a score of 0 is expected; the run shows the task set
loads, validates and completes end to end. For a real model:

    python -m runner.preflight configs/demo_b3.yaml

Each prompt is the family preamble, then the task's own premises (bus speed, word
timing, gap and each label's transmission rate), then the timed word stream. Before
the task is built, every sample's target is checked against a fresh run of the
conformance checker (bench/arinc429/conformance.py) over its stream and stated rules.
A mismatch raises B3ValidationError and the eval does not start. Print one sample,
with the checker's measurements, without a model:

    python -m bench.arinc429.b3 --show b3-03

To see the verdict follow the stated rules and an answer built on a contradicting
rate score 0, without a model server:

    python scripts/arinc429/demo_b3_scoring.py

Scoring is the ARINC 429 field-level scorer (bench/scorers.py, WBS 3.7, issue #62).
The target has one field, "violations", so the whole list must match (sorted, as the
prompt asks): 1.0 or 0, with no partial credit for finding some violations. It reads
the JSON in the answer, so spacing and surrounding prose don't matter. Task version
1.0.1 (file v1.0.0) switched to it from exact match.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.arinc429.b1 import PREAMBLE
from bench.arinc429.b3 import B3_TASK_VERSION, TASK_FILE, validate_samples
from bench.loader import load_preamble, load_tasks
from bench.provenance import sha256_file
from bench.provenance_solver import record_provenance
from bench.scorers import arinc429_bnr_bcd_capability_scorer


@task
def b3_arinc429_bus_conformance() -> Task:
    dataset = load_tasks(TASK_FILE)
    validate_samples(dataset)
    # The per-task premises and stream are in the task file, which the corpus hash
    # covers, so the fixed part the template hash pins down is the family preamble.
    return Task(
        dataset=dataset,
        solver=[
            record_provenance(sha256_file(TASK_FILE), load_preamble(PREAMBLE)),
            generate(),
        ],
        scorer=arinc429_bnr_bcd_capability_scorer(),
        version=B3_TASK_VERSION,
        metadata={"subfamily": "B3", "task_file": TASK_FILE.name},
    )
