"""Loader demo (WBS 1.13, issue #21): ADS-B tasks with the family preamble injected.

    PYTHONPATH=. inspect eval tasks/loader_demo.py --model mockllm/model

Run from the repo root; PYTHONPATH=. lets the task import ``bench``.
mockllm returns filler, so a score of 0 is expected; the point is that the
prompt each sample sends starts with the preamble (see the sample's messages in
the log viewer). Print the prompts without running a model:

    python -m bench.loader data/tasks/loader_demo_v0.1.0.jsonl
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.loader import REPO_ROOT, load_tasks
from bench.scorers import deterministic_exact_match

TASK_FILE = REPO_ROOT / "data" / "tasks" / "loader_demo_v0.1.0.jsonl"


@task
def loader_demo() -> Task:
    return Task(
        dataset=load_tasks(TASK_FILE),
        solver=generate(),
        scorer=deterministic_exact_match(),
        version="0.1.0",
    )
