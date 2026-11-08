"""tshark and scapy as Inspect AI sandbox tools (WBS 1.11, issue #20).

Two tools that read a capture file in the task directory inside the Docker sandbox
from WBS 1.8 (no network, read-only root, unprivileged user; isolation proven by
WBS 1.9):

    tshark      Wireshark's dissector on one capture: a one-line summary per packet,
                chosen fields per packet, or the full dissection tree, optionally
                narrowed by a display filter.
    scapy_read  scapy's parse of a range of packets: a summary, the stored field
                values layer by layer, or a hex dump.

Neither tool passes model text to a shell or into code. The model picks from an
allow-list of options; everything else is fixed:

- tshark only ever runs as ``tshark -r - -n`` plus a display filter (``-Y``), field
  names (``-e``) and one of three output modes. Options that capture (``-i``, which
  also reaches extcap interfaces such as sshdump and udpdump), write files (``-w``),
  load Lua (``-X``) or change preferences (``-o``) cannot be expressed.
- The file must be a .pcap, .pcapng or .cap file inside the task directory
  (``TASK_DIR``, where Inspect copies a sample's ``files``). The path is checked on
  the host, then again inside the sandbox after the file is opened, by resolving the
  open descriptor. So ``..``, absolute paths elsewhere, and symlinks the model makes
  with another tool that point outside the task directory are all refused, and the
  path cannot be swapped between the check and the read: tshark reads the already
  opened file on stdin.

The allow-list keeps the tools to what they are for; it is not what keeps the host
safe. If a task also gives the model ``python()``, the model can read any file in
the container with it. Isolation is the container's job (WBS 1.8 / 1.9).

Errors go back to the model as ToolError and the sample keeps running. Output is
capped at ``OUTPUT_LIMIT`` characters and marked when cut, so a big capture cannot
flood the context.

``scapy_read`` reports the values stored in the file, so a wrong UDP checksum or a
bad length field (the AFDX "bus fault" traces plant both) reaches the model as it is.
A test checks this with a frame whose stored checksum is wrong.

Command-line demo (needs Docker). A scripted mock model reads a synthetic AFDX
capture with both tools, tries a path outside the task directory, then tries a live
capture, a network connection and a host write.

    python -m bench.packet_tools
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Any, Literal

from inspect_ai.tool import Tool, ToolError, python, tool
from inspect_ai.util import sandbox

from bench.tools import SANDBOX, TOOL_TIMEOUT_S, _last_line, sandbox_image

TOOLSET_NAME = "pcap_sandbox"
TOOLSET_VERSION = "1.0.0"
TOOL_NAMES = ("tshark", "scapy_read")

# Inspect copies a sample's files into the container's working directory. That is
# WORKDIR in sandbox/Dockerfile and a tmpfs in sandbox/compose.yaml; a test checks
# all three agree.
TASK_DIR = "/workspace"
CAPTURE_SUFFIXES = (".pcap", ".pcapng", ".cap")
MAX_PATH_CHARS = 255

OUTPUT_LIMIT = 20_000  # characters returned to the model per call
MAX_FILTER_CHARS = 1_000
MAX_FIELDS = 32
FIELD_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,99}")
TSHARK_MODES = ("summary", "fields", "verbose")
SCAPY_DETAILS = ("summary", "fields", "hex")
MAX_SCAPY_COUNT = 200

REFUSED_EXIT = 3  # the sandbox script's exit code for a refused file

# Runs inside the sandbox. argv: task_dir, path, then the tool's own arguments.
# Opens the file first and checks where the open descriptor really points, so a
# symlink, or a swap after the check, cannot redirect the read.
_OPEN_TASK_FILE = r'''
import os, stat, sys

def refuse(msg):
    sys.stderr.write(msg + "\n")
    sys.exit(3)

def open_task_file(task_dir, rel):
    root = os.path.realpath(task_dir)
    try:
        fd = os.open(os.path.join(root, rel), os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    except FileNotFoundError:
        refuse(f"no such file in the task directory: {rel}")
    except OSError as e:
        refuse(f"cannot open {rel}: {e.strerror}")
    real = os.readlink(f"/proc/self/fd/{fd}")
    if os.path.commonpath([root, real]) != root:
        refuse(f"{rel} resolves outside the task directory {task_dir}")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        refuse(f"{rel} is not a regular file")
    os.set_blocking(fd, True)
    return fd
'''

TSHARK_SCRIPT = _OPEN_TASK_FILE + r'''
import subprocess, tempfile
task_dir, rel, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
fd = open_task_file(task_dir, rel)
with tempfile.TemporaryFile() as err:
    proc = subprocess.Popen(["tshark", "-r", "-", "-n", *sys.argv[4:]],
                            stdin=fd, stdout=subprocess.PIPE, stderr=err)
    out = proc.stdout.read(limit + 1)
    cut = len(out) > limit
    if cut:
        proc.kill()
    proc.stdout.close()
    rc = proc.wait()
    err.seek(0)
    stderr = err.read().decode(errors="replace")
text = out[:limit].decode(errors="replace")
if cut:
    text += f"\n[output cut at {limit} characters; narrow the display filter, e.g. frame.number <= 50]"
elif rc != 0:
    if not text.strip():
        sys.stderr.write(stderr)
        sys.exit(1)  # never 3: that code means "refused"
    text += "\n[tshark warning] " + (stderr.strip().splitlines() or ["exit " + str(rc)])[-1]
sys.stdout.write(text)
'''

SCAPY_SCRIPT = _OPEN_TASK_FILE + r'''
import json
task_dir, rel, first, count, detail, limit = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
fd = open_task_file(task_dir, rel)
from scapy.all import PcapReader, hexdump  # scapy.all registers the link layers; utils alone reads Raw
packets, more, size = [], False, 0
try:
    with PcapReader(os.fdopen(fd, "rb")) as reader:
        for number, pkt in enumerate(reader, start=1):
            if number < first:
                continue
            if len(packets) == count:
                more = True
                break
            row = {"number": number, "time": str(pkt.time), "length": len(bytes(pkt))}
            if detail == "summary":
                row["summary"] = pkt.summary()
            elif detail == "fields":
                row["fields"] = pkt.show(dump=True)   # values as stored in the file
            else:
                row["hex"] = hexdump(pkt, dump=True)
            size += len(json.dumps(row))
            if size > limit:
                more = True
                break
            packets.append(row)
except Exception as e:
    sys.stderr.write(f"scapy could not read {rel}: {type(e).__name__}: {e}\n")
    sys.exit(2)
print(json.dumps({"file": rel, "first": first, "returned": len(packets),
                  "more_packets": more, "packets": packets}, indent=1))
'''


def check_capture_path(path: Any) -> str:
    """Return the path relative to TASK_DIR, or raise ToolError saying why it is refused.

    This is the host-side check. The sandbox script checks again after opening the
    file, which also catches symlinks.
    """
    if not isinstance(path, str) or not path.strip():
        raise ToolError("file must be the name of a capture file in the task directory")
    text = path.strip()
    if len(text) > MAX_PATH_CHARS or any(ord(c) < 32 for c in text):
        raise ToolError("file must be a plain path of at most 255 printable characters")
    full = posixpath.normpath(posixpath.join(TASK_DIR, text))
    if not full.startswith(TASK_DIR + "/"):
        raise ToolError(f"file must be inside the task directory {TASK_DIR}; refused: {text}")
    if not full.lower().endswith(CAPTURE_SUFFIXES):
        raise ToolError(f"file must be a capture file ({', '.join(CAPTURE_SUFFIXES)}); refused: {text}")
    return posixpath.relpath(full, TASK_DIR)


def check_display_filter(display_filter: Any) -> str:
    if not isinstance(display_filter, str):
        raise ToolError("display_filter must be a string")
    text = display_filter.strip()
    if len(text) > MAX_FILTER_CHARS or any(c in text for c in "\x00\r\n"):
        raise ToolError(f"display_filter must be one line of at most {MAX_FILTER_CHARS} characters")
    return text


def check_fields(fields: Any) -> list[str]:
    if not isinstance(fields, list) or not fields:
        raise ToolError("fields must be a non-empty list of Wireshark field names, e.g. ['ip.dst']")
    if len(fields) > MAX_FIELDS:
        raise ToolError(f"at most {MAX_FIELDS} fields per call")
    for name in fields:
        if not isinstance(name, str) or not FIELD_NAME.fullmatch(name):
            raise ToolError(f"not a Wireshark field name: {name!r}")
    return list(fields)


def tshark_args(mode: str, display_filter: str = "", fields: list[str] | None = None) -> list[str]:
    """The tshark options for one call, after `tshark -r - -n`. Only these can be produced."""
    if mode not in TSHARK_MODES:
        raise ToolError(f"mode must be one of {', '.join(TSHARK_MODES)}")
    args: list[str] = []
    flt = check_display_filter(display_filter)
    if flt:
        args += ["-Y", flt]
    if mode == "fields":
        args += ["-T", "fields", "-E", "header=y", "-E", "separator=/t",
                 "-E", "occurrence=a", "-E", "aggregator=,"]
        for name in check_fields(fields):
            args += ["-e", name]
    elif fields:
        raise ToolError("fields are only used with mode='fields'")
    elif mode == "verbose":
        args.append("-V")
    return args


def tshark_command(file: str, mode: str = "summary", display_filter: str = "",
                   fields: list[str] | None = None) -> list[str]:
    """The argv run in the sandbox for one tshark call."""
    rel = check_capture_path(file)
    return ["python3", "-c", TSHARK_SCRIPT, TASK_DIR, rel, str(OUTPUT_LIMIT),
            *tshark_args(mode, display_filter, fields)]


def scapy_command(file: str, first: int = 1, count: int = 20, detail: str = "summary") -> list[str]:
    """The argv run in the sandbox for one scapy_read call."""
    rel = check_capture_path(file)
    if isinstance(first, bool) or not isinstance(first, int) or first < 1:
        raise ToolError("first must be a packet number, starting at 1")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_SCAPY_COUNT:
        raise ToolError(f"count must be between 1 and {MAX_SCAPY_COUNT}")
    if detail not in SCAPY_DETAILS:
        raise ToolError(f"detail must be one of {', '.join(SCAPY_DETAILS)}")
    return ["python3", "-c", SCAPY_SCRIPT, TASK_DIR, rel, str(first), str(count), detail,
            str(OUTPUT_LIMIT)]


async def _run(cmd: list[str], timeout: int, what: str) -> str:
    result = await sandbox().exec(cmd, timeout=timeout)
    if result.returncode == REFUSED_EXIT:
        raise ToolError(f"refused: {_last_line(result.stderr)}")
    if not result.success:
        raise ToolError(f"{what} error: {_last_line(result.stderr)}")
    return result.stdout


@tool
def tshark(timeout: int = TOOL_TIMEOUT_S) -> Tool:
    async def execute(file: str, mode: Literal["summary", "fields", "verbose"] = "summary",
                      display_filter: str = "", fields: list[str] | None = None) -> str:
        """
        Dissect a capture file in the task directory with Wireshark's tshark.

        Name resolution is off, so addresses are shown as numbers. Output over
        20,000 characters is cut; narrow it with a display filter such as
        "frame.number <= 50".

        Args:
          file: Capture file (.pcap, .pcapng or .cap) in the task directory, e.g. "capture.pcap".
          mode: "summary" for one line per packet, "fields" for the chosen fields as tab-separated columns with a header, "verbose" for the full dissection tree.
          display_filter: Optional Wireshark display filter, e.g. "udp.dstport == 5001".
          fields: Field names for mode "fields", e.g. ["frame.number", "eth.dst", "ip.dst"].

        Returns:
          tshark's output.
        """
        cmd = tshark_command(file, mode, display_filter, fields)  # validate before the sandbox
        return await _run(cmd, timeout, "tshark")

    return execute


@tool
def scapy_read(timeout: int = TOOL_TIMEOUT_S) -> Tool:
    async def execute(file: str, first: int = 1, count: int = 20,
                      detail: Literal["summary", "fields", "hex"] = "summary") -> str:
        """
        Parse packets from a capture file in the task directory with scapy.

        Field values are shown as stored in the file. Checksums and lengths are not
        recomputed, so a wrong checksum in the capture appears as it is.

        Args:
          file: Capture file (.pcap, .pcapng or .cap) in the task directory, e.g. "capture.pcap".
          first: Number of the first packet to return, starting at 1.
          count: How many packets to return (1 to 200).
          detail: "summary" for one line per packet, "fields" for every layer's field values, "hex" for a hex dump.

        Returns:
          JSON with the packets read and whether more follow.
        """
        cmd = scapy_command(file, first, count, detail)  # validate before the sandbox
        out = await _run(cmd, timeout, "scapy")
        try:
            json.loads(out)
        except ValueError:
            raise ToolError("scapy returned output that is not JSON") from None
        return out.strip()

    return execute


def packet_tools(timeout: int = TOOL_TIMEOUT_S, include_python: bool = False) -> list[Tool]:
    """The tool set given to the model for capture tasks."""
    tools = [tshark(timeout=timeout), scapy_read(timeout=timeout)]
    return [python(timeout=timeout), *tools] if include_python else tools


def toolset_metadata(enabled: bool, include_python: bool = False) -> dict[str, Any]:
    """What to record in Task metadata so the EvalLog says which tools the model had."""
    if not enabled:
        return {"tool_set": "none"}
    return {
        "tool_set": TOOLSET_NAME,
        "tool_set_version": TOOLSET_VERSION,
        "tools": (["python"] if include_python else []) + list(TOOL_NAMES),
        "tool_timeout_s": TOOL_TIMEOUT_S,
        "task_dir": TASK_DIR,
        "output_limit_chars": OUTPUT_LIMIT,
        "sandbox_image": sandbox_image(),
    }


# ---------------------------------------------------------------------------
# Synthetic capture for the demo and tests
# ---------------------------------------------------------------------------

DEMO_VL = {
    "vl_id": 100, "name": "vl_100", "bag_ms": 16, "lmax_bytes": 512, "skew_max_ms": 2,
    "src_ip": "10.1.2.3", "src_mac_user_id": 0x1234, "interface_id": 1,
    "udp_src_port": 5000, "udp_dst_port": 5001, "app_payload_bytes": 64,
    "networks": ["A", "B"], "initial_sn": 1,
}
DEMO_FRAMES_PER_VL = 3
DEMO_T0 = 1000.0
DEMO_B_OFFSET_S = 0.00001  # B copies arrive 10 us after A, well inside SkewMax


def write_demo_capture(path: Path) -> list[dict[str, Any]]:
    """Write a small synthetic AFDX capture (both networks interleaved) with the
    team's frame builder. Returns what was written, frame by frame, as ground truth."""
    from scapy.utils import wrpcap

    from bench.afdx import afdx_synth as afdx

    cfg = {"schema_version": "1.0",
           "redundancy": {"integrity_check": True, "resync_rule": "drop_with_resync"},
           "virtual_links": [DEMO_VL]}
    packets, truth = [], []
    for record in afdx.generate_frames(cfg, DEMO_FRAMES_PER_VL):
        pkt = record.packet
        pkt.time = DEMO_T0 + record.t_s + (DEMO_B_OFFSET_S if record.network == "B" else 0.0)
        packets.append(pkt)
        truth.append({"network": record.network, "vl_id": record.vl_id, "sn": record.sn,
                      **afdx.parse_frame(pkt)})
    wrpcap(str(path), packets)
    return truth


def main() -> None:
    import tempfile

    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
    from inspect_ai.solver import generate, use_tools

    from bench.tools import REPO_ROOT

    canary = REPO_ROOT / "written-by-model.txt"
    calls: list[tuple[str, dict[str, Any]]] = [
        ("tshark", {"file": "capture.pcap", "mode": "fields", "display_filter": "udp",
                    "fields": ["frame.number", "frame.time_relative", "eth.dst", "ip.dst",
                               "udp.dstport"]}),
        ("scapy_read", {"file": "capture.pcap", "count": 2}),
        ("tshark", {"file": "../etc/passwd.pcap"}),
        ("python", {"code": "import os\nos.symlink('/etc/os-release', 'link.pcap')"}),
        ("scapy_read", {"file": "link.pcap"}),
        ("python", {"code": "import subprocess\nprint(subprocess.run(['tshark', '-i', 'any', "
                            "'-c', '1', '-a', 'duration:3'], capture_output=True, text=True)"
                            ".stderr.splitlines()[1])"}),
        ("python", {"code": "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=5)"}),
        ("python", {"code": f"open('{canary}', 'w').write('x')"}),
    ]
    outputs = [ModelOutput.for_tool_call("mockllm/model", name, args) for name, args in calls]
    outputs.append(ModelOutput.from_content("mockllm/model", "VL 100"))
    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / "capture.pcap"
        write_demo_capture(capture)
        task = Task(
            dataset=[Sample(input="Which virtual link carries the frames in capture.pcap?",
                            target="100", files={"capture.pcap": str(capture)})],
            solver=[use_tools(packet_tools(include_python=True)), generate()],
            sandbox=SANDBOX,
            metadata=toolset_metadata(True, include_python=True),
        )
        [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs))
    if log.status != "success" or not log.samples:
        raise SystemExit(f"run failed: {log.error}")
    for message in log.samples[0].messages:
        if isinstance(message, ChatMessageTool):
            body = (message.error.message if message.error else message.text).strip()
            print(f"--- {message.function} -> {'ERROR ' if message.error else ''}{body}")
    print(f"--- host check: {canary} exists: {canary.exists()}")
    print(f"--- final answer: {log.samples[0].output.completion}")


if __name__ == "__main__":
    main()
