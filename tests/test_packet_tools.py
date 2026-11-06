"""tshark and scapy sandbox tools (WBS 1.11, issue #20).

Acceptance criteria (product backlog):
- When the model calls the tshark tool on a task pcap, dissected fields are returned.
- When the model passes a file path outside the task directory, the tool refuses the call.

Sprint 3 backlog demo, also covered:
- A model-issued tshark call and a scapy call on a synthetic capture inside the sandbox
  return parsed output; the sandbox has no network and writes nothing to the host.

The tshark fields are checked against the AFDX frame builder's own record of what it
wrote (bench.afdx.afdx_synth), so the dissector is graded against a different decoder,
not against itself. The Docker tests run a scripted mock model through Inspect's tool
path, like tests/test_sandbox_tools.py, and are skipped when Docker is not running.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolError

from bench import packet_tools as pt
from bench.packet_tools import (
    OUTPUT_LIMIT,
    SCAPY_SCRIPT,
    TASK_DIR,
    TOOL_NAMES,
    TSHARK_SCRIPT,
    check_capture_path,
    packet_tools,
    scapy_command,
    scapy_read,
    tshark,
    tshark_args,
    tshark_command,
    toolset_metadata,
    write_demo_capture,
)
from bench.tools import SANDBOX, sandbox_image

REPO_ROOT = Path(__file__).resolve().parents[1]
CANARY = f"canary-{uuid.uuid4().hex}"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker daemon is not running")


# ---------------------------------------------------------------------------
# No Docker needed: registration, metadata, the allow-list and the path check
# ---------------------------------------------------------------------------


def test_tool_set_is_tshark_and_scapy_read() -> None:
    from inspect_ai.tool._tool_def import ToolDef

    assert tuple(ToolDef(t).name for t in packet_tools()) == TOOL_NAMES == ("tshark", "scapy_read")
    with_python = tuple(ToolDef(t).name for t in packet_tools(include_python=True))
    assert with_python == ("python", "tshark", "scapy_read")


def test_metadata_records_the_tool_set_task_dir_and_image() -> None:
    meta = toolset_metadata(True)
    assert meta["tool_set"] == pt.TOOLSET_NAME
    assert meta["tool_set_version"] == pt.TOOLSET_VERSION
    assert meta["tools"] == list(TOOL_NAMES)
    assert meta["task_dir"] == TASK_DIR
    assert meta["output_limit_chars"] == OUTPUT_LIMIT
    assert meta["sandbox_image"] == sandbox_image()
    assert toolset_metadata(True, include_python=True)["tools"] == ["python", *TOOL_NAMES]
    assert toolset_metadata(False) == {"tool_set": "none"}


def test_task_dir_is_the_sandbox_working_dir_and_a_tmpfs() -> None:
    """Inspect copies sample files into the working dir; it must be TASK_DIR and in memory."""
    dockerfile = (REPO_ROOT / "sandbox" / "Dockerfile").read_text()
    assert re.search(rf"^WORKDIR {re.escape(TASK_DIR)}$", dockerfile, re.M)
    service = yaml.safe_load((REPO_ROOT / "sandbox" / "compose.yaml").read_text())["services"]["default"]
    assert any(entry.split(":")[0] == TASK_DIR for entry in service["tmpfs"])


@pytest.mark.parametrize("path, rel", [
    ("capture.pcap", "capture.pcap"),
    ("  capture.pcap ", "capture.pcap"),
    ("./capture.pcap", "capture.pcap"),
    ("traces/net_a.pcapng", "traces/net_a.pcapng"),
    ("traces/../capture.pcap", "capture.pcap"),
    (f"{TASK_DIR}/capture.pcap", "capture.pcap"),
    ("OLD.CAP", "OLD.CAP"),
])
def test_paths_inside_the_task_directory_are_accepted(path: str, rel: str) -> None:
    assert check_capture_path(path) == rel


@pytest.mark.parametrize("path, message", [
    ("../etc/passwd.pcap", "inside the task directory"),
    ("../../capture.pcap", "inside the task directory"),
    ("/etc/capture.pcap", "inside the task directory"),
    ("/tmp/capture.pcap", "inside the task directory"),
    (f"{TASK_DIR}/../etc/capture.pcap", "inside the task directory"),
    (f"{TASK_DIR}x/capture.pcap", "inside the task directory"),   # prefix, not a parent
    (TASK_DIR, "inside the task directory"),
    (".", "inside the task directory"),
    ("capture.txt", "capture file"),
    ("/etc/passwd", "inside the task directory"),
    ("capture.pcap\n-w /tmp/x", "printable"),
    ("x" * 300 + ".pcap", "at most 255"),
    ("", "name of a capture file"),
    (None, "name of a capture file"),
])
def test_paths_outside_the_task_directory_are_refused(path: Any, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        check_capture_path(path)


def test_tshark_options_come_only_from_the_allow_list() -> None:
    assert tshark_args("summary") == []
    assert tshark_args("verbose", "udp") == ["-Y", "udp", "-V"]
    assert tshark_args("fields", "", ["ip.dst", "_ws.col.info"]) == [
        "-T", "fields", "-E", "header=y", "-E", "separator=/t", "-E", "occurrence=a",
        "-E", "aggregator=,", "-e", "ip.dst", "-e", "_ws.col.info"]
    cmd = tshark_command("capture.pcap", "fields", "udp", ["ip.dst"])
    assert cmd[:6] == ["python3", "-c", TSHARK_SCRIPT, TASK_DIR, "capture.pcap", str(OUTPUT_LIMIT)]
    # Every option the model can cause sits at a flag position from this set; filter text
    # and field names only ever appear as the value after -Y or -e.
    flags = cmd[6::2]
    assert set(flags) <= {"-Y", "-T", "-E", "-e"}


@pytest.mark.parametrize("mode, flt, fields, message", [
    ("pdml", "", None, "mode must be one of"),
    ("fields", "", None, "non-empty list"),
    ("fields", "", [], "non-empty list"),
    ("fields", "", ["-w"], "not a Wireshark field name"),
    ("fields", "", ["ip.dst -w /tmp/x"], "not a Wireshark field name"),
    ("fields", "", ["ip.dst", 7], "not a Wireshark field name"),
    ("fields", "", ["ip.dst"] * 33, "at most 32"),
    ("summary", "", ["ip.dst"], "only used with mode='fields'"),
    ("summary", "udp\n-w /tmp/x", None, "one line"),
    ("summary", "x" * 1001, None, "one line"),
    ("summary", 5, None, "must be a string"),
])
def test_tshark_refuses_options_outside_the_allow_list(mode: str, flt: Any, fields: Any,
                                                       message: str) -> None:
    with pytest.raises(ToolError, match=message):
        tshark_args(mode, flt, fields)


def test_model_text_is_passed_as_arguments_not_code() -> None:
    path = "x'); import os; os.system('id'); ('.pcap"
    cmd = tshark_command(path, "summary", "udp or ip.src == 'x'")
    assert path not in TSHARK_SCRIPT and path in cmd
    assert scapy_command(path)[2] == SCAPY_SCRIPT


@pytest.mark.parametrize("first, count, detail, message", [
    (0, 20, "summary", "first must be"),
    (True, 20, "summary", "first must be"),
    ("1", 20, "summary", "first must be"),
    (1, 0, "summary", "count must be"),
    (1, 201, "summary", "count must be"),
    (1, 20, "show2", "detail must be one of"),
])
def test_scapy_read_refuses_bad_arguments(first: Any, count: Any, detail: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        scapy_command("capture.pcap", first, count, detail)


@pytest.mark.anyio
async def test_tools_refuse_a_bad_path_without_running_anything() -> None:
    """No sandbox exists in this test, so reaching sandbox().exec() would raise something else."""
    with pytest.raises(ToolError, match="inside the task directory"):
        await tshark()("../etc/passwd.pcap")
    with pytest.raises(ToolError, match="inside the task directory"):
        await scapy_read()("/etc/capture.pcap")


def test_demo_capture_is_synthetic_and_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.pcap", tmp_path / "b.pcap"
    truth = write_demo_capture(a)
    write_demo_capture(b)
    assert a.read_bytes() == b.read_bytes()
    assert [(f["network"], f["sn"]) for f in truth] == [
        ("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3), ("B", 3)]
    assert {f["dst_ip"] for f in truth} == {"224.224.0.100"}   # VL 100, AFDX multicast


# ---------------------------------------------------------------------------
# Docker: a scripted model calls the tools inside the sandbox
# ---------------------------------------------------------------------------

FIELDS = ["frame.number", "frame.time_relative", "eth.dst", "ip.dst", "udp.dstport", "data.data"]

# (label, tool, arguments), in the order the mock model calls them.
CALLS: list[tuple[str, str, dict[str, Any]]] = [
    # Acceptance criterion 1 and the sprint demo: parsed output from both tools
    ("tshark_fields", "tshark", {"file": "capture.pcap", "mode": "fields", "fields": FIELDS}),
    ("tshark_summary", "tshark", {"file": "capture.pcap"}),
    ("tshark_verbose", "tshark", {"file": "capture.pcap", "mode": "verbose",
                                  "display_filter": "frame.number == 1"}),
    ("tshark_filter", "tshark", {"file": f"{TASK_DIR}/capture.pcap", "mode": "fields",
                                 "display_filter": "frame.number <= 2", "fields": ["frame.number"]}),
    ("tshark_bad_filter", "tshark", {"file": "capture.pcap", "display_filter": "bogus.field == 1"}),
    ("tshark_truncated_file", "tshark", {"file": "cut.pcap"}),
    ("tshark_big", "tshark", {"file": "big.pcap", "mode": "verbose"}),
    ("scapy_summary", "scapy_read", {"file": "capture.pcap"}),
    ("scapy_range", "scapy_read", {"file": "capture.pcap", "first": 5, "count": 1}),
    ("scapy_fields", "scapy_read", {"file": "badsum.pcap", "detail": "fields", "count": 1}),
    ("scapy_hex", "scapy_read", {"file": "capture.pcap", "detail": "hex", "count": 1}),
    # Acceptance criterion 2: paths outside the task directory are refused
    ("tshark_dotdot", "tshark", {"file": "../etc/passwd.pcap"}),
    ("scapy_absolute", "scapy_read", {"file": "/tmp/outside.pcap"}),
    ("make_links", "python", {"code": (
        "import os, shutil\n"
        "shutil.copy('capture.pcap', '/tmp/outside.pcap')\n"
        "os.symlink('/tmp/outside.pcap', 'link.pcap')\n"     # a real capture, but outside
        "os.symlink('/etc', 'etc')\n"
        "os.mkdir('dir.pcap')\n"
        "os.mkfifo('fifo.pcap')\n"
        "print('ok')")}),
    ("tshark_symlink", "tshark", {"file": "link.pcap"}),
    ("scapy_symlink", "scapy_read", {"file": "link.pcap"}),
    ("tshark_symlink_dir", "tshark", {"file": "etc/hosts.pcap"}),
    ("tshark_missing", "tshark", {"file": "missing.pcap"}),
    ("tshark_directory", "tshark", {"file": "dir.pcap"}),
    ("scapy_fifo", "scapy_read", {"file": "fifo.pcap"}),
    # Sprint demo: no network, no host writes
    ("live_capture", "python", {"code": (
        "import subprocess\n"
        "r = subprocess.run(['tshark', '-i', 'any', '-c', '1', '-a', 'duration:3'],\n"
        "                   capture_output=True, text=True, timeout=20)\n"
        "print(r.returncode, r.stderr)")}),
    ("network", "python", {"code": "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=5)"}),
    ("write_repo", "python", {"code": f"open('{REPO_ROOT}/{CANARY}', 'w').write('x')"}),
    ("write_etc", "python", {"code": f"open('/etc/{CANARY}', 'w').write('x')"}),
    ("after_errors", "tshark", {"file": "capture.pcap", "mode": "fields", "fields": ["frame.number"],
                                "display_filter": "frame.number == 6"}),
]
FINAL_ANSWER = "100"
BAD_UDP_CHECKSUM = 0x1234


def write_fixtures(directory: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    from scapy.layers.inet import UDP
    from scapy.layers.l2 import Ether
    from scapy.utils import rdpcap, wrpcap

    from bench.afdx import afdx_synth as afdx

    capture = directory / "capture.pcap"
    truth = write_demo_capture(capture)

    # One frame whose stored UDP checksum is wrong, as in the AFDX "bus fault" traces.
    frame = Ether(bytes(rdpcap(str(capture))[0]))
    frame[UDP].chksum = BAD_UDP_CHECKSUM
    wrpcap(str(directory / "badsum.pcap"), [Ether(bytes(frame))])

    # A capture cut off mid-packet.
    (directory / "cut.pcap").write_bytes(capture.read_bytes()[:-20])

    # Enough packets that the verbose dissection exceeds the output limit.
    cfg = {"schema_version": "1.0",
           "redundancy": {"integrity_check": True, "resync_rule": "drop_with_resync"},
           "virtual_links": [pt.DEMO_VL]}
    wrpcap(str(directory / "big.pcap"), [r.packet for r in afdx.generate_frames(cfg, 100)])

    names = ("capture.pcap", "badsum.pcap", "cut.pcap", "big.pcap")
    return {name: str(directory / name) for name in names}, truth


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, dict[str, ChatMessageTool], list[dict[str, Any]]]:
    files, truth = write_fixtures(tmp_path_factory.mktemp("captures"))
    outputs = [ModelOutput.for_tool_call("mockllm/model", name, args, tool_call_id=label)
               for label, name, args in CALLS]
    outputs.append(ModelOutput.from_content("mockllm/model", FINAL_ANSWER))
    task = Task(
        dataset=[Sample(input="Which virtual link carries the frames in capture.pcap?",
                        target=FINAL_ANSWER, files=files)],
        solver=[use_tools(packet_tools(include_python=True)), generate()],
        sandbox=SANDBOX,
        metadata=toolset_metadata(True, include_python=True),
    )
    [log] = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                         log_dir=str(tmp_path_factory.mktemp("logs")), display="none")
    assert log.status == "success", log.error
    assert log.samples
    results = {m.tool_call_id: m for m in log.samples[0].messages
               if isinstance(m, ChatMessageTool) and m.tool_call_id}
    return log, results, truth


def ok(results: dict[str, ChatMessageTool], label: str) -> str:
    message = results[label]
    assert message.error is None, message.error
    return message.text


def refused(results: dict[str, ChatMessageTool], label: str) -> str:
    message = results[label]
    assert message.error is not None, f"{label} was not refused: {message.text[:200]}"
    return message.error.message


def text(results: dict[str, ChatMessageTool], label: str) -> str:
    """python() returns a traceback as ordinary tool output, not as a tool error."""
    message = results[label]
    return message.error.message if message.error else message.text


def rows(tsv: str) -> list[dict[str, str]]:
    header, *lines = tsv.strip().splitlines()
    return [dict(zip(header.split("\t"), line.split("\t"))) for line in lines]


# --- Acceptance criterion 1: tshark returns dissected fields ---------------


@needs_docker
def test_tshark_fields_match_the_frame_builder(run: Any) -> None:
    """tshark's dissection agrees with what the AFDX builder says it wrote."""
    _, results, truth = run
    got = rows(ok(results, "tshark_fields"))
    assert [int(r["frame.number"]) for r in got] == list(range(1, len(truth) + 1))
    for row, frame in zip(got, truth):
        assert row["eth.dst"] == frame["dst_mac"] == "03:00:00:00:00:64"   # VL ID 100 in the MAC
        assert row["ip.dst"] == frame["dst_ip"]
        assert int(row["udp.dstport"]) == frame["udp_dport"]
        assert int(row["data.data"][-2:], 16) == frame["sn"]           # SN is the last byte
    offsets = [round(float(r["frame.time_relative"]), 6) for r in got]
    assert offsets == [0.0, 0.00001, 0.016, 0.01601, 0.032, 0.03201]


@needs_docker
def test_tshark_summary_and_verbose_modes(run: Any) -> None:
    _, results, truth = run
    summary = ok(results, "tshark_summary").strip().splitlines()
    assert len(summary) == len(truth)
    assert all("10.1.2.3" in line and "224.224.0.100" in line and "UDP" in line for line in summary)
    verbose = ok(results, "tshark_verbose")
    assert "User Datagram Protocol, Src Port: 5000, Dst Port: 5001" in verbose
    assert "Frame 2:" not in verbose  # the display filter applied


@needs_docker
def test_display_filter_and_absolute_task_path(run: Any) -> None:
    _, results, _ = run
    assert [r["frame.number"] for r in rows(ok(results, "tshark_filter"))] == ["1", "2"]


@needs_docker
def test_tshark_errors_come_back_to_the_model(run: Any) -> None:
    _, results, _ = run
    assert "tshark error" in refused(results, "tshark_bad_filter")
    cut = ok(results, "tshark_truncated_file")   # what could be read, plus the warning
    assert "[tshark warning]" in cut and "cut short" in cut
    lines = [line for line in cut.splitlines() if line.strip()]
    assert len(lines) == 6 and lines[-1].startswith("[tshark warning]")  # 5 packets, then the warning


@needs_docker
def test_big_output_is_cut_at_the_limit(run: Any) -> None:
    _, results, _ = run
    out = ok(results, "tshark_big")
    assert "[output cut at 20000 characters" in out
    assert len(out) < OUTPUT_LIMIT + 200


# --- Sprint demo: scapy returns parsed output ------------------------------


@needs_docker
def test_scapy_read_parses_the_capture(run: Any) -> None:
    _, results, truth = run
    summary = json.loads(ok(results, "scapy_summary"))
    assert summary["returned"] == len(truth) and summary["more_packets"] is False
    assert [p["number"] for p in summary["packets"]] == [1, 2, 3, 4, 5, 6]
    assert all(p["summary"].startswith("Ether / IP / UDP 10.1.2.3:5000 > 224.224.0.100:5001")
               for p in summary["packets"])
    assert summary["packets"][1]["time"] == "1000.000010"
    window = json.loads(ok(results, "scapy_range"))
    assert [p["number"] for p in window["packets"]] == [5] and window["more_packets"] is True
    hexdump = json.loads(ok(results, "scapy_hex"))["packets"][0]["hex"]
    assert hexdump.startswith("0000  03 00 00 00 00 64")


@needs_docker
def test_scapy_shows_the_stored_checksum(run: Any) -> None:
    """A wrong checksum in the file must reach the model as it is (AFDX bus-fault traces)."""
    _, results, _ = run
    fields = json.loads(ok(results, "scapy_fields"))["packets"][0]["fields"]
    assert re.search(rf"chksum\s*=\s*{BAD_UDP_CHECKSUM:#x}", fields)


# --- Acceptance criterion 2: paths outside the task directory are refused --


@needs_docker
def test_paths_outside_are_refused_before_the_sandbox(run: Any) -> None:
    _, results, _ = run
    assert "inside the task directory" in refused(results, "tshark_dotdot")
    assert "inside the task directory" in refused(results, "scapy_absolute")


@needs_docker
def test_symlinks_out_of_the_task_directory_are_refused(run: Any) -> None:
    _, results, _ = run
    assert ok(results, "make_links").strip() == "ok"
    for label in ("tshark_symlink", "scapy_symlink"):
        assert "resolves outside the task directory" in refused(results, label)
    assert "refused" in refused(results, "tshark_symlink_dir")


@needs_docker
def test_missing_files_directories_and_fifos_are_refused(run: Any) -> None:
    _, results, _ = run
    assert "no such file in the task directory: missing.pcap" in refused(results, "tshark_missing")
    assert "dir.pcap is not a regular file" in refused(results, "tshark_directory")
    assert "fifo.pcap is not a regular file" in refused(results, "scapy_fifo")  # and no hang


# --- Sprint demo: no network, no host writes; run continues ----------------


@needs_docker
def test_sandbox_cannot_capture_or_reach_the_network(run: Any) -> None:
    _, results, _ = run
    capture = ok(results, "live_capture")
    assert not capture.startswith("0 ")   # tshark exited non-zero
    assert "permission to capture" in capture
    assert "Network is unreachable" in text(results, "network")


@needs_docker
def test_sandbox_writes_nothing_to_the_host(run: Any) -> None:
    _, results, _ = run
    assert "No such file or directory" in text(results, "write_repo")
    assert "Read-only file system" in text(results, "write_etc")
    assert not (REPO_ROOT / CANARY).exists()


@needs_docker
def test_run_continues_after_errors_and_logs_the_tool_set(run: Any) -> None:
    log, results, _ = run
    assert rows(ok(results, "after_errors")) == [{"frame.number": "6"}]
    assert len(results) == len(CALLS)
    assert log.samples[0].output.completion == FINAL_ANSWER
    assert log.eval.metadata == toolset_metadata(True, include_python=True)
