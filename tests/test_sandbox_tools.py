"""pyModeS and Python as sandbox tools (WBS 1.10, issue #19).

Acceptance criteria (issue):
- When the model calls the Python tool, pyModeS decodes a frame inside the sandbox
  and returns the result.
- When code in the sandbox attempts a network call or a host write, it fails.

From the product backlog, also covered:
- When the model calls the decode tool with valid hex, the result is returned from
  inside the sandbox.
- When the tool raises an exception, a sanitized error is returned to the model and
  the run continues.

WBS 1.9 already proves the sandbox is isolated by running probes with
sandbox().exec(). These tests go through the path the model uses: a scripted mock
model makes tool calls, Inspect executes them in the Docker sandbox, and the results
come back as tool messages. The Docker tests are skipped when Docker is not running.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolError

from bench import tools
from bench.tools import (
    DECODE_SCRIPT,
    SANDBOX,
    TOOL_NAMES,
    adsb_decode,
    check_frame,
    decode_command,
    sandbox_image,
    sandbox_tools,
    toolset_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO_ROOT / "data" / "adsb" / "MANIFEST.json").read_text())

DEMO_FRAME = "8DF00A112369A6B1C31820ABE595"  # adsb-0001, callsign ZZZ101
CORRUPT_FRAME = "8DF00A112361A6B1C31820ABE595"  # adsb-0065, parity fails
CANARY = f"canary-{uuid.uuid4().hex}"
FINAL_ANSWER = "ZZZ101"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker daemon is not running")


# ---------------------------------------------------------------------------
# No Docker needed: registration, input checks, metadata
# ---------------------------------------------------------------------------


def test_tool_set_is_python_and_adsb_decode() -> None:
    from inspect_ai.tool._tool_def import ToolDef

    names = tuple(ToolDef(t).name for t in sandbox_tools())
    assert names == TOOL_NAMES == ("python", "adsb_decode")


def test_tools_run_in_the_repo_sandbox() -> None:
    assert SANDBOX == ("docker", str(REPO_ROOT / "sandbox" / "compose.yaml"))


def test_metadata_records_the_tool_set_and_image() -> None:
    meta = toolset_metadata(True)
    assert meta["tool_set"] == tools.TOOLSET_NAME
    assert meta["tool_set_version"] == tools.TOOLSET_VERSION
    assert meta["tools"] == list(TOOL_NAMES)
    assert meta["sandbox_image"] == sandbox_image() == "boeing-team-2-sandbox:1.0.0"
    assert toolset_metadata(False) == {"tool_set": "none"}


def test_sandbox_decoder_is_the_reference_decoder_version() -> None:
    """The tool decodes with the same pyModeS version that produced the ground truth."""
    lock = (REPO_ROOT / "sandbox" / "requirements.lock").read_text()
    assert MANIFEST["reference_decoder"] in lock


def test_frame_is_passed_as_an_argument_not_code() -> None:
    cmd = decode_command(f"  {DEMO_FRAME.lower()} ")
    assert cmd == ["python3", "-c", DECODE_SCRIPT, DEMO_FRAME]
    assert DEMO_FRAME not in DECODE_SCRIPT


@pytest.mark.parametrize("bad, message", [
    ("8DF00A11", "28 hexadecimal characters"),
    (DEMO_FRAME + "00", "28 hexadecimal characters"),
    ("'); import os; os.system('id'); ('", "only hexadecimal"),
    ("8DF00A11 2369A6B1C31820ABE595", "only hexadecimal"),
    ("", "only hexadecimal"),
])
def test_bad_frame_is_refused_before_the_sandbox(bad: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        check_frame(bad)


@pytest.mark.anyio
async def test_decode_tool_refuses_bad_input_without_running_anything() -> None:
    """No sandbox exists in this test, so reaching sandbox().exec() would raise something else."""
    tool = adsb_decode()
    with pytest.raises(ToolError, match="hexadecimal"):
        await tool("not-a-frame")

def test_tools_off_means_no_tools_and_no_sandbox() -> None:
    from tasks.sandbox_tools_demo import sandbox_tools_demo

    off = sandbox_tools_demo(tools=False)
    on = sandbox_tools_demo(tools=True)
    assert off.sandbox is None and off.metadata == {"tool_set": "none"}
    assert on.sandbox is not None and on.metadata["tool_set"] == tools.TOOLSET_NAME


# ---------------------------------------------------------------------------
# Docker: a scripted model calls the tools inside the sandbox
# ---------------------------------------------------------------------------

# (label, tool, arguments), in the order the mock model calls them.
CALLS: list[tuple[str, str, dict[str, str]]] = [
    ("decode_tool", "adsb_decode", {"frame": DEMO_FRAME}),
    ("decode_tool_corrupt", "adsb_decode", {"frame": CORRUPT_FRAME}),
    ("python_decode", "python", {"code": (
        "import os, pyModeS as pms\n"
        f"print(pms.decode('{DEMO_FRAME}')['callsign'])\n"
        "print('uid', os.getuid(), 'container', os.path.exists('/.dockerenv'))")}),
    ("python_network", "python", {"code": (
        "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=5)")}),
    ("python_dns", "python", {"code": "import socket\nsocket.getaddrinfo('example.com', 80)"}),
    ("python_write_etc", "python", {"code": f"open('/etc/{CANARY}', 'w').write('x')"}),
    ("python_write_repo", "python", {"code": f"open('{REPO_ROOT}/{CANARY}', 'w').write('x')"}),
    ("python_exception", "python", {"code": "1 / 0"}),
    ("decode_tool_bad_input", "adsb_decode", {"frame": "8DF00A11"}),
    ("after_errors", "python", {"code": "print('still running')"}),
]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, dict[str, ChatMessageTool]]:
    outputs = [ModelOutput.for_tool_call("mockllm/model", name, args, tool_call_id=label)
               for label, name, args in CALLS]
    outputs.append(ModelOutput.from_content("mockllm/model", FINAL_ANSWER))
    task = Task(
        dataset=[Sample(input="Decode the frame with the tools.", target=FINAL_ANSWER)],
        solver=[use_tools(sandbox_tools()), generate()],
        sandbox=SANDBOX,
        metadata=toolset_metadata(True),
    )
    [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                         log_dir=str(tmp_path_factory.mktemp("logs")), display="none")
    assert log.status == "success", log.error
    assert log.samples
    results = {m.tool_call_id: m for m in log.samples[0].messages
               if isinstance(m, ChatMessageTool) and m.tool_call_id}
    return log, results


def text(message: ChatMessageTool) -> str:
    return message.error.message if message.error else message.text


# --- Acceptance criterion 1: decode inside the sandbox ----------------------


@needs_docker
def test_python_tool_decodes_with_pymodes_inside_the_sandbox(run: Any) -> None:
    _, results = run
    out = results["python_decode"]
    assert out.error is None
    lines = out.text.strip().splitlines()
    assert lines[0] == "ZZZ101"
    assert lines[1] == "uid 1000 container True"  # the sandbox user, in a container


@needs_docker
def test_decode_tool_returns_pymodes_fields_from_the_sandbox(run: Any) -> None:
    _, results = run
    out = results["decode_tool"]
    assert out.error is None
    decoded = json.loads(out.text)
    frame = json.loads((REPO_ROOT / "data" / "adsb" / MANIFEST["file"]).read_text()
                       .splitlines()[0])
    assert decoded == frame["expected"]  # same as the stored ground truth


@needs_docker
def test_decode_tool_reports_a_failed_parity_check(run: Any) -> None:
    _, results = run
    decoded = json.loads(results["decode_tool_corrupt"].text)
    assert decoded["crc_valid"] is False


# --- Acceptance criterion 2: network calls and host writes fail -------------


@needs_docker
def test_network_call_from_the_python_tool_fails(run: Any) -> None:
    _, results = run
    assert "Network is unreachable" in text(results["python_network"])
    assert "socket.gaierror" in text(results["python_dns"])


@needs_docker
def test_write_from_the_python_tool_fails(run: Any) -> None:
    _, results = run
    assert "Read-only file system" in text(results["python_write_etc"])
    # The repo path does not exist in the container, so the write cannot reach the host.
    assert "No such file or directory" in text(results["python_write_repo"])
    assert not (REPO_ROOT / CANARY).exists()


# --- Backlog: errors go back to the model and the run continues -------------


@needs_docker
def test_exception_is_returned_to_the_model(run: Any) -> None:
    _, results = run
    out = text(results["python_exception"])
    assert "ZeroDivisionError: division by zero" in out
    assert str(REPO_ROOT) not in out  # traceback is from the container, not the host


@needs_docker
def test_bad_frame_is_a_tool_error_for_the_model(run: Any) -> None:
    _, results = run
    out = results["decode_tool_bad_input"]
    assert out.error is not None
    assert "28 hexadecimal characters" in out.error.message


@needs_docker
def test_run_continues_after_errors(run: Any) -> None:
    log, results = run
    assert results["after_errors"].text.strip() == "still running"
    assert log.samples[0].output.completion == FINAL_ANSWER
    assert len(results) == len(CALLS)


@needs_docker
def test_evallog_records_the_tool_set(run: Any) -> None:
    log, _ = run
    assert log.eval.metadata == toolset_metadata(True)
