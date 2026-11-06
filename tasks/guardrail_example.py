"""Example task that runs under a selectable guardrail profile (WBS 1.17, issue #25).

python -m runner.preflight configs/guardrail_example.yaml
inspect eval tasks/guardrail_example.py -T guardrail_profile=hardened_example \
    --model mockllm/model

The profile is a task parameter, chosen in the run config's ``guardrail_profile`` field
(``none`` is the baseline). Its name, version and sha256 are recorded in the EvalLog
under ``eval.metadata``, and the profile text is added as a system message ahead of
the unchanged task text. A profile that is unknown, or edited without a version bump,
makes the task refuse to load, so no model call is made and no log is written.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from runner.guardrails import load_guardrail_profile


@task
def guardrail_example(guardrail_profile: str = "none", guardrail_version: int = 1) -> Task:
    profile = load_guardrail_profile(guardrail_profile, guardrail_version)
    return Task(
        dataset=[Sample(input="Reply with the single word: ready", target="ready")],
        solver=[*profile.pre_solvers(), generate()],
        scorer=includes(),
        metadata=profile.metadata(),
        version="0.1.0",
    )
