"""Loader demo (WBS 1.13, issue #21): ADS-B tasks with the family preamble injected.

    PYTHONPATH=. inspect eval tasks/loader_demo.py --model mockllm/model

Run from the repo root; PYTHONPATH=. lets the task import ``bench``.
mockllm returns filler, so a score of 0 is expected. The point is that each
prompt starts with the preamble (see the sample's messages in the log viewer)
and that the corpus hash is recorded in the log as eval.metadata.corpus_hash.
Print the prompts without running a model:

    python -m bench.loader data/tasks/loader_demo_v0.1.0.jsonl
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from bench.corpus import task_corpus
from bench.loader import REPO_ROOT, load_tasks
from bench.scorers import deterministic_exact_match

TASK_FILE = REPO_ROOT / "data" / "tasks" / "loader_demo_v0.1.0.jsonl"


@task
def loader_demo() -> Task:
    dataset = load_tasks(TASK_FILE)  # refuses a task file or preamble that fails its manifest
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=deterministic_exact_match(),
        version="0.1.0",
        # corpus_hash, corpus_files and corpus_hash_scheme land in EvalLog.eval.metadata
        metadata=task_corpus(TASK_FILE),
    )
