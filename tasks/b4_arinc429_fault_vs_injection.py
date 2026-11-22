"""B4: ARINC 429 hardware fault versus deliberate injection (WBS 3.6, issue #61).

    PYTHONPATH=. inspect eval tasks/b4_arinc429_fault_vs_injection.py --model mockllm/model

Run from the repo root (PowerShell: ``$env:PYTHONPATH = "."`` on its own line first).
mockllm returns filler text, so a capability score of 0 is expected; the run shows the
task set loads, validates and completes end to end. For a real model:

    python -m runner.preflight configs/demo_b4.yaml

Each prompt is the family preamble, then the task's own premises (the transmitter's
label, SDI, SSM and rate, the largest change per transmission, and what counts as a
fault or an injection), then the word stream. Before the task is built, every sample's
target and unsafe-if clauses are checked against a fresh run of the triage reference
(bench/arinc429/triage.py). A mismatch raises B4ValidationError and the eval does not
start. Print one sample, with per-word findings, without a model:

    python -m bench.arinc429.b4 --show b4-01

To see a correct "injection" verdict get full credit and a "benign sensor noise"
answer flagged unsafe, without a model server:

    python scripts/arinc429/demo_b4_scoring.py

Scoring is two separate scorers, never combined (CLAUDE.md rule 2). Capability: the
B4 triage scorer (1.0 for the right verdict and words, 0.5 for the right verdict only).
Safety: the rule-based unsafe-if scorer (bench/safety.py), which flags an answer that
dismisses an injected word as noise, a glitch or a hardware fault.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.arinc429.b1 import PREAMBLE
from bench.arinc429.b4 import B4_VERSION, TASK_FILE, b4_triage_scorer, validate_samples
from bench.loader import load_preamble, load_tasks
from bench.provenance import sha256_file
from bench.provenance_solver import record_provenance
from bench.safety import unsafe_if_rule_scorer


@task
def b4_arinc429_fault_vs_injection() -> Task:
    dataset = load_tasks(TASK_FILE)
    validate_samples(dataset)
    return Task(
        dataset=dataset,
        solver=[
            record_provenance(sha256_file(TASK_FILE), load_preamble(PREAMBLE)),
            generate(),
        ],
        # Capability and safety are separate scorers, reported side by side and never
        # combined: a safety failure leaves the capability score unchanged.
        scorer=[b4_triage_scorer(), unsafe_if_rule_scorer()],
        version=B4_VERSION,
        metadata={"subfamily": "B4", "task_file": TASK_FILE.name},
    )
