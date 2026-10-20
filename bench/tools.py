"""pyModeS and Python as Inspect AI sandbox tools (WBS 1.10, issue #19).

Two tools, both executed inside the Docker sandbox from WBS 1.8 (no network, read-only
root, unprivileged user; isolation proven by WBS 1.9):

    python       Inspect's built-in python() tool. pyModeS and scapy are installed
                 in the sandbox image, so the model can decode with them directly.
    adsb_decode  One ADS-B / Mode S frame decoded by the sandbox's pyModeS, returned
                 as JSON. The frame is checked on the host first and passed to the
                 sandbox as a command-line argument, never spliced into code.

Errors go back to the model and the sample keeps running: a Python exception comes
back as its traceback (the python tool returns stderr), a bad frame or a decoder
failure as a ToolError, and a timeout as Inspect's own tool error. Tracebacks come
from inside the container, so they show container paths, never host paths.

Tool access is a run variable. With tools on, the model can run the same pyModeS
version that produced the ADS-B ground truth, so a decode task then measures tool
use rather than protocol knowledge. Tasks take ``tools`` as a parameter and record
the tool set in the EvalLog (``toolset_metadata``), so results with and without
tools are never compared as if they were the same configuration.

Command-line demo (needs Docker). A scripted mock model calls the tools: a decode,
an outbound connection, a write outside the workspace, an exception and a bad frame.

    python -m bench.tools
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.tool import Tool, ToolError, python, tool
from inspect_ai.util import sandbox

TOOLSET_NAME = "adsb_sandbox"
TOOLSET_VERSION = "1.0.0"
TOOL_TIMEOUT_S = 30

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "sandbox" / "compose.yaml"
SANDBOX = ("docker", str(COMPOSE_FILE))

FRAME_HEX_LENGTH = 28  # 112-bit extended squitter
HEX_DIGITS = frozenset(string.hexdigits)

# Runs inside the sandbox. The frame arrives as argv[1]; nothing from the model is
# ever formatted into this source.
DECODE_SCRIPT = (
    "import json, sys\n"
    "import pyModeS as pms\n"
    "print(json.dumps(dict(pms.decode(sys.argv[1])), sort_keys=True, default=str))\n"
)


def sandbox_image() -> str:
    """The sandbox image tag from compose.yaml, which carries SANDBOX_VERSION."""
    services = yaml.safe_load(COMPOSE_FILE.read_text())["services"]
    return str(services["default"]["image"])


def check_frame(frame: str) -> str:
    """Normalize a frame or raise ToolError saying what is wrong with it."""
    if not isinstance(frame, str):
        raise ToolError("frame must be a string of 28 hexadecimal characters")
    text = frame.strip().upper()
    if not text or not set(text) <= HEX_DIGITS:
        raise ToolError("frame must contain only hexadecimal characters (0-9, A-F)")
    if len(text) != FRAME_HEX_LENGTH:
        raise ToolError(
            f"frame must be {FRAME_HEX_LENGTH} hexadecimal characters (112 bits), "
            f"got {len(text)}")
    return text


def decode_command(frame: str) -> list[str]:
    """The argv run in the sandbox for one decode."""
    return ["python3", "-c", DECODE_SCRIPT, check_frame(frame)]


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "no error output"


@tool
def adsb_decode(timeout: int = TOOL_TIMEOUT_S) -> Tool:
    async def execute(frame: str) -> str:
        """
        Decode one ADS-B / Mode S message with pyModeS and return its fields as JSON.

        The result includes crc_valid. pyModeS decodes the fields even when the
        parity check fails, so check crc_valid before trusting them.

        Args:
          frame: The 112-bit message as 28 hexadecimal characters.

        Returns:
          The decoded fields as a JSON object.
        """
        cmd = decode_command(frame)  # validate before touching the sandbox
        result = await sandbox().exec(cmd, timeout=timeout)
        if not result.success:
            raise ToolError(f"decoder error: {_last_line(result.stderr)}")
        try:
            json.loads(result.stdout)
        except ValueError:
            raise ToolError("decoder returned output that is not JSON") from None
        return result.stdout.strip()

    return execute


def sandbox_tools(timeout: int = TOOL_TIMEOUT_S) -> list[Tool]:
    """The tool set given to the model when tools are on."""
    return [python(timeout=timeout), adsb_decode(timeout=timeout)]


TOOL_NAMES = ("python", "adsb_decode")


def toolset_metadata(enabled: bool) -> dict[str, Any]:
    """What to record in Task metadata so the EvalLog says which tools the model had."""
    if not enabled:
        return {"tool_set": "none"}
    return {
        "tool_set": TOOLSET_NAME,
        "tool_set_version": TOOLSET_VERSION,
        "tools": list(TOOL_NAMES),
        "tool_timeout_s": TOOL_TIMEOUT_S,
        "sandbox_image": sandbox_image(),
    }


def main() -> None:
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
    from inspect_ai.solver import generate, use_tools

    frame = "8DF00A112369A6B1C31820ABE595"  # synthetic, data/adsb adsb-0001
    calls: list[tuple[str, dict[str, str]]] = [
        ("adsb_decode", {"frame": frame}),
        ("python", {"code": "import pyModeS as pms, os\n"
                            f"print(pms.decode('{frame}')['callsign'], 'uid', os.getuid())"}),
        ("python", {"code": "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=5)"}),
        ("python", {"code": "open('/etc/written-by-model', 'w').write('x')"}),
        ("python", {"code": "1 / 0"}),
        ("adsb_decode", {"frame": "not-a-frame"}),
    ]
    outputs = [ModelOutput.for_tool_call("mockllm/model", name, args) for name, args in calls]
    outputs.append(ModelOutput.from_content("mockllm/model", "done"))
    task = Task(
        dataset=[Sample(input="Decode the frame with the tools.", target="ZZZ101")],
        solver=[use_tools(sandbox_tools()), generate()],
        sandbox=SANDBOX,
        metadata=toolset_metadata(True),
    )
    [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs))
    if log.status != "success" or not log.samples:
        raise SystemExit(f"run failed: {log.error}")
    for message in log.samples[0].messages:
        if isinstance(message, ChatMessageTool):
            body = message.error.message if message.error else message.text
            print(f"--- {message.function} -> {'ERROR ' if message.error else ''}{body.strip()}")
    print(f"--- final answer: {log.samples[0].output.completion}")


if __name__ == "__main__":
    main()
