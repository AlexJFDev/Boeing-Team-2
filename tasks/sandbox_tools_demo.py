"""Sandbox tools demo task (WBS 1.10, issue #19).

One ADS-B decode task, run with or without the sandbox tools:

    PYTHONPATH=. inspect eval tasks/sandbox_tools_demo.py --model ollama/llama3.1
    PYTHONPATH=. inspect eval tasks/sandbox_tools_demo.py -T tools=false --model ollama/llama3.1

With tools on, the model gets python() and adsb_decode() inside the Docker sandbox
(Docker must be running). With tools off there is no sandbox and no tools. The
EvalLog's task metadata records which, so the two are never mixed up in analysis.

For a scripted run that needs no real model, see ``python -m bench.tools``.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools

from bench.tools import SANDBOX, sandbox_tools, toolset_metadata


@task
def sandbox_tools_demo(tools: bool = True) -> Task:
    return Task(
        dataset=[
            Sample(
                id="tools-demo-001",
                input=("What callsign does this ADS-B message carry? "
                       "8DF00A112369A6B1C31820ABE595. Reply with the callsign only."),
                target="ZZZ101",
            )
        ],
        solver=[use_tools(sandbox_tools()), generate()] if tools else [generate()],
        sandbox=SANDBOX if tools else None,
        scorer=includes(),
        metadata=toolset_metadata(tools),
        version="0.1.0",
    )
