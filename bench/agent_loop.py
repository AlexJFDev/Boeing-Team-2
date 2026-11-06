"""Multi-turn tool-calling solver loop (WBS 1.16, issue #24).

The loop itself is Inspect AI's ``react()`` agent: the model generates, Inspect runs
any tool calls it made, the results go back to the model, and this repeats until the
model calls ``submit`` with its answer. This module only fixes the configuration so
every task uses the same loop, and records it in the EvalLog.

A runaway loop is stopped by a turn limit set on the Task (``turn_limit``, counted in
model turns). When the limit is reached Inspect ends the sample with a limit error,
keeps every message so far, and the run continues with the next sample. Tasks take
``max_turns`` as a parameter and record it with ``loop_metadata`` so results with
different limits are never compared as if they were the same configuration.

Command-line demo (needs Docker). A scripted mock model calls a sandboxed tool over
several turns and submits an answer, then a second mock model never stops calling
the tool and the turn limit ends it.

    python -m bench.agent_loop
"""

from __future__ import annotations

from typing import Any

import inspect_ai
from inspect_ai.agent import AgentPrompt, AgentSubmit, react
from inspect_ai.solver import Solver, solver
from inspect_ai.tool import Tool

LOOP_NAME = "react_loop"
LOOP_VERSION = "1.0.0"
DEFAULT_MAX_TURNS = 10

PROMPT = AgentPrompt(
    instructions=(
        "Use the tools to work out the answer. When you are sure, call submit() "
        "with the answer only."),
    assistant_prompt=None,
)


@solver
def bench_solver(tools: list[Tool]) -> Solver:
    """The benchmark's multi-turn loop over the given tools."""
    return react(
        prompt=PROMPT,
        tools=tools,
        submit=AgentSubmit(name="submit", description="Submit the final answer."),
        truncation="disabled",
    )


def loop_metadata(max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, Any]:
    """What to record in Task metadata so the EvalLog says how the loop was run."""
    return {
        "loop": LOOP_NAME,
        "loop_version": LOOP_VERSION,
        "max_turns": max_turns,
        "inspect_ai_version": inspect_ai.__version__,
    }


def main() -> None:
    from collections.abc import Iterator

    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
    from inspect_ai.scorer import includes

    from bench.tools import SANDBOX, sandbox_tools

    frame = "8DF00A112369A6B1C31820ABE595"  # synthetic, data/adsb adsb-0001

    def call(name: str, args: dict[str, str]) -> ModelOutput:
        return ModelOutput.for_tool_call("mockllm/model", name, args)

    def runaway() -> Iterator[ModelOutput]:
        while True:
            yield call("adsb_decode", {"frame": frame})

    scenarios: list[tuple[str, Any]] = [
        ("answers after several turns", [
            call("adsb_decode", {"frame": frame}),
            call("python", {"code": f"import pyModeS as pms\nprint(pms.decode('{frame}')['callsign'])"}),
            call("adsb_decode", {"frame": frame}),
            call("submit", {"answer": "ZZZ101"}),
        ]),
        ("runaway loop hits the turn limit", runaway()),
    ]
    max_turns = 5
    for title, outputs in scenarios:
        task = Task(
            dataset=[Sample(input="What callsign does this frame carry? " + frame,
                            target="ZZZ101")],
            solver=bench_solver(sandbox_tools()),
            sandbox=SANDBOX,
            scorer=includes(),
            turn_limit=max_turns,
            metadata=loop_metadata(max_turns),
        )
        [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                             display="none")
        if log.status != "success" or not log.samples:
            raise SystemExit(f"run failed: {log.error}")
        sample = log.samples[0]
        tool_msgs = [m for m in sample.messages if isinstance(m, ChatMessageTool)]
        print(f"=== {title}: {len(tool_msgs)} tool results, "
              f"stopped by {sample.limit.type + ' limit' if sample.limit else 'submit()'}")
        for message in tool_msgs:
            print(f"--- {message.function}: {message.text.strip()[:80]}")


if __name__ == "__main__":
    main()
