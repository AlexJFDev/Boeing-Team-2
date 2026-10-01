"""Smoke test for the Inspect AI install.

Runs with the built-in mock model, so it needs no GPU, API key or Ollama server:

    inspect eval tasks/smoke_test.py --model mockllm/model

To exercise a real local model instead, see docs/setup.md.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def smoke_test():
    return Task(
        dataset=[
            Sample(
                input="Reply with the single word: ready",
                target="ready",
            )
        ],
        solver=generate(),
        scorer=includes(),
    )
