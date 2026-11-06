"""Regression task set (WBS 1.25, issue #33).

Every A1 and A2 task answered by fixed canned responses and scored by the A-family
scorers. No model call is made; a replay solver supplies the responses, so the
scores are deterministic and comparable with the recorded baseline.

    PYTHONPATH=. inspect eval tasks/regression_set.py --model mockllm/model
    python -m bench.regression check --log logs/<that run>.eval

``python -m bench.regression check`` does both steps in one command.
"""

from inspect_ai import Task, task

from bench.regression import regression_task


@task
def regression_set() -> Task:
    return regression_task()
