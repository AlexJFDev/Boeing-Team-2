"""B1: ARINC 429 word decode and the label bit order (WBS 3.3, issue #58).

    PYTHONPATH=. inspect eval tasks/b1_arinc429_word_decode.py --model mockllm/model

Run from the repo root (PowerShell: ``$env:PYTHONPATH = "."`` first). mockllm
returns filler text, so a score of 0 is expected; the run shows the task set loads,
validates and completes end to end. For a real model:

    python -m runner.preflight configs/demo_b1.yaml

Before the task is built, every sample's target is checked against a fresh decode of
its word by the ARINC 429 reference decoder (WBS 3.2). A mismatch raises
B1ValidationError and the eval does not start. Print one sample without a model:

    python -m bench.arinc429.b1

To see answers scored against the decoder's targets, without a model server:

    python scripts/arinc429/demo_b1_scoring.py

Scoring is the ARINC 429 field-level scorer (bench/scorers.py, WBS 3.7, issue #62):
credit per field (label, SDI, SSM, value), with numbers within 0.0001 of the target. A
wrong SSM also zeroes the value (backlog 3.7). A parity failure scores 1.0 only for
"answer": "reject". Task version 1.0.1 (file v1.0.0)
switched to it from exact match.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.arinc429.b1 import B1_TASK_VERSION, PREAMBLE, TASK_FILE, validate_samples
from bench.loader import load_preamble, load_tasks
from bench.provenance import sha256_file
from bench.provenance_solver import record_provenance
from bench.scorers import arinc429_bnr_bcd_capability_scorer


@task
def b1_arinc429_word_decode() -> Task:
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
        scorer=arinc429_bnr_bcd_capability_scorer(),
        version=B1_TASK_VERSION,
        metadata={"subfamily": "B1", "task_file": TASK_FILE.name},
    )
