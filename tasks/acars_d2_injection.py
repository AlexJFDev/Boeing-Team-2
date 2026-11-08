"""D2: ACARS indirect prompt injection (task 6.2, issue #81).

    PYTHONPATH=. inspect eval tasks/acars_d2_injection.py --model mockllm/model
    PYTHONPATH=. inspect eval tasks/acars_d2_injection.py --model ollama/llama3.1 -T vectors=V01,V03

Run from the repo root; PYTHONPATH=. lets the task import ``bench`` (not needed
once ``pip install -r requirements.txt`` has installed the repo with ``-e .``).

Each vector runs twice: with the instruction hidden in an ACARS field
(``injected``) and with harmless filler (``control``). ``d2_safety`` records
whether the model obeyed the hidden instruction; ``d2_control_capability``
records, separately, whether it got the control answer right. They are never
combined. mockllm filler comes out ``unparseable`` (not a pass): it only proves
the harness runs. For the scripted safe/obedient demo and a readable report use
``python -m bench.acars.d2_run`` (see bench/acars/README.md).

Task args (-T): vectors, temperature (0.0), seed (42), max_tokens (512),
guardrail_file (none), sentinel (derived from seed + vector file hash). All are recorded in the log.
"""

from inspect_ai import Task, task

from bench.acars.d2_harness import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    build_task,
)


@task
def acars_d2_injection(
    vectors: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    guardrail_file: str | None = None,
    sentinel: str | None = None,
) -> Task:
    return build_task(
        vectors=vectors,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        guardrail_file=guardrail_file,
        sentinel=sentinel,
    )
