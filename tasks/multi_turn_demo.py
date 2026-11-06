"""Multi-turn tool-calling demo task (WBS 1.16, issue #24).

One ADS-B decode task solved by Inspect's react() loop with the sandbox tools, so the
model can call tools over several turns and finish with submit(). ``max_turns`` caps
the loop; a model that never submits is stopped at the limit and the sample is scored
on what it has (an incorrect answer), with the limit recorded in the EvalLog.

    PYTHONPATH=. inspect eval tasks/multi_turn_demo.py --model ollama/llama3.1
    PYTHONPATH=. inspect eval tasks/multi_turn_demo.py -T max_turns=3 --model ollama/llama3.1

Docker must be running (the tools execute in the sandbox). For a scripted run that
needs no real model, see ``python -m bench.agent_loop``.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from bench.agent_loop import DEFAULT_MAX_TURNS, bench_solver, loop_metadata
from bench.tools import SANDBOX, sandbox_tools, toolset_metadata


@task
def multi_turn_demo(max_turns: int = DEFAULT_MAX_TURNS) -> Task:
    return Task(
        dataset=[
            Sample(
                id="multi-turn-demo-001",
                input=("What callsign does this ADS-B message carry? "
                       "8DF00A112369A6B1C31820ABE595. Submit the callsign only."),
                target="ZZZ101",
            )
        ],
        solver=bench_solver(sandbox_tools()),
        sandbox=SANDBOX,
        scorer=includes(),
        turn_limit=max_turns,
        metadata={**toolset_metadata(True), **loop_metadata(max_turns)},
        version="0.1.0",
    )
