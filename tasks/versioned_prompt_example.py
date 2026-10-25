"""Example task driven by a versioned prompt template (Issue #23).

    inspect eval tasks/versioned_prompt_example.py --model mockllm/model

The template name, version and sha256 are recorded in the EvalLog under
``eval.metadata``. Editing the template file without a version bump makes the
task refuse to load, so no model call is made and no log is written.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from runner.prompts import load_prompt_template

TEMPLATE_NAME = "single_turn_example"
TEMPLATE_VERSION = 1


@task
def versioned_prompt_example(version: int = TEMPLATE_VERSION):
    template = load_prompt_template(TEMPLATE_NAME, version)
    return Task(
        dataset=[
            Sample(
                input="Reply with the single word: ready",
                target="ready",
            )
        ],
        solver=template.solver(),
        scorer=includes(),
        version=version,
        metadata=template.metadata(),
    )
