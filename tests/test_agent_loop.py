"""Multi-turn tool-calling solver loop (WBS 1.16, issue #24).

Demo (issue): a mock task where the model calls a sandboxed tool over several turns
until it answers, and a turn limit stops a runaway loop.

The loop tests use a plain stub tool so they need no Docker. One test goes through
the real sandbox tools and is skipped when Docker is not running.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.tool import Tool, tool

from bench.agent_loop import DEFAULT_MAX_TURNS, bench_solver, loop_metadata
from bench.tools import SANDBOX, sandbox_tools
from tasks.multi_turn_demo import multi_turn_demo
from tests.test_sandbox_tools import DEMO_FRAME, needs_docker

ANSWER = "ZZZ101"


@tool
def lookup() -> Tool:
    async def execute(key: str) -> str:
        """
        Look up a value.

        Args:
          key: The key to look up.

        Returns:
          The value.
        """
        return f"value-of-{key}"

    return execute


def call(name: str, args: dict[str, str], label: str | None = None) -> ModelOutput:
    return ModelOutput.for_tool_call("mockllm/model", name, args, tool_call_id=label)


def run_loop(outputs: Any, tmp_path: Path, max_turns: int = DEFAULT_MAX_TURNS,
             tools: list[Tool] | None = None, sandbox: Any = None) -> Any:
    task = Task(
        dataset=[Sample(input="Find the answer.", target=ANSWER)],
        solver=bench_solver(tools or [lookup()]),
        sandbox=sandbox,
        scorer=includes(),
        turn_limit=max_turns,
        metadata=loop_metadata(max_turns),
    )
    [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                         log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    assert log.samples
    return log


def tool_results(log: Any) -> list[ChatMessageTool]:
    return [m for m in log.samples[0].messages if isinstance(m, ChatMessageTool)]


def runaway() -> Iterator[ModelOutput]:
    while True:
        yield call("lookup", {"key": "again"})


# --- When the model calls a tool over several turns, it can then answer ------


def test_model_calls_tool_over_several_turns_then_answers(tmp_path: Path) -> None:
    outputs = [call("lookup", {"key": "a"}, "t1"), call("lookup", {"key": "b"}, "t2"),
               call("lookup", {"key": "c"}, "t3"), call("submit", {"answer": ANSWER}, "done")]
    log = run_loop(outputs, tmp_path)
    sample = log.samples[0]
    lookups = [m for m in tool_results(log) if m.function == "lookup"]
    assert [m.text for m in lookups] == ["value-of-a", "value-of-b", "value-of-c"]
    assert sample.limit is None  # stopped by submit(), not a limit
    assert sample.scores["includes"].value == "C"


# --- Negative case: a runaway loop is stopped by the turn limit --------------


def test_turn_limit_stops_a_runaway_loop(tmp_path: Path) -> None:
    log = run_loop(runaway(), tmp_path, max_turns=4)
    sample = log.samples[0]
    assert sample.limit is not None
    assert sample.limit.type == "turn"
    assert sample.limit.limit == 4
    # Bounded: the loop did not run on, and every tool result before the stop is kept.
    assert len(tool_results(log)) <= 4
    assert sample.scores["includes"].value == "I"  # no answer was submitted


def test_lower_limit_stops_sooner(tmp_path: Path) -> None:
    short = run_loop(runaway(), tmp_path / "short", max_turns=2)
    long = run_loop(runaway(), tmp_path / "long", max_turns=6)
    assert len(tool_results(short)) < len(tool_results(long))


# --- Reproducibility: the loop configuration is recorded ---------------------


def test_loop_config_is_recorded_in_the_log(tmp_path: Path) -> None:
    log = run_loop([call("submit", {"answer": ANSWER})], tmp_path, max_turns=7)
    meta = log.eval.metadata
    assert meta["loop"] == "react_loop"
    assert meta["max_turns"] == 7
    assert meta["inspect_ai_version"]


def test_demo_task_declares_limit_and_version() -> None:
    task = multi_turn_demo(max_turns=3)
    assert task.version == "0.1.0"
    assert task.metadata is not None and task.metadata["max_turns"] == 3


# --- The same loop with the real sandbox tools -------------------------------


@needs_docker
def test_loop_runs_sandbox_tools_over_several_turns(tmp_path: Path) -> None:
    outputs = [call("adsb_decode", {"frame": DEMO_FRAME}, "decode"),
               call("python", {"code": "print(1 + 1)"}, "py"),
               call("submit", {"answer": ANSWER}, "done")]
    log = run_loop(outputs, tmp_path, tools=sandbox_tools(), sandbox=SANDBOX)
    by_id = {m.tool_call_id: m for m in tool_results(log)}
    assert ANSWER in by_id["decode"].text
    assert by_id["py"].text.strip() == "2"
    assert log.samples[0].scores["includes"].value == "C"
