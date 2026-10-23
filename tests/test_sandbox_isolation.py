"""Sandbox isolation tests (WBS 1.9, issue #9).

Acceptance criteria:
- When code inside the sandbox tries to write to the host file system, the
  write fails.
- When code inside the sandbox attempts an outbound network connection, the
  connection is blocked.

A container cannot name a host path, so "write to the host file system" is
tested as three things that together rule it out: nothing from the host is
mounted into the container, the container's own root filesystem is read-only,
and a file written to the one writable location never appears on the host.

Inspect AI only disables networking by itself when there is no compose file.
We ship sandbox/compose.yaml, so the settings are ours to keep. The tests are
in three layers:

1. Config guard: reads compose.yaml and fails if an isolation setting is
   missing. Needs no Docker.
2. Behaviour: starts the sandbox through Inspect AI, the way the runner will,
   and runs probes inside it. Also inspects the running container from the host.
3. Controls: show the probes can succeed (so a failure means "blocked", not
   "broken probe") and that a weakened sandbox is detected.

Layers 2 and 3 are skipped when the Docker daemon is not running.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DIR = REPO_ROOT / "sandbox"
COMPOSE_FILE = SANDBOX_DIR / "compose.yaml"

CANARY_NAME = f"canary-{uuid.uuid4().hex}"
PROBE_TIMEOUT_S = 30

# A synthetic frame from data/adsb (adsb-0001); decodes to callsign ZZZ101.
DEMO_FRAME = "8DF00A112369A6B1C31820ABE595"

TCP_PROBE = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=5).close()"
DNS_PROBE = "import socket; socket.getaddrinfo('example.com', 80)"
HTTP_PROBE = "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)"

# Prints "name:operstate" for every network device. The kernel gives each
# network namespace placeholder tunnel devices (tunl0, sit0, gre0, ...), so
# the device list is not just "lo"; what matters is that none of them is up.
INTERFACES_PROBE = (
    "import pathlib\n"
    "for d in sorted(pathlib.Path('/sys/class/net').iterdir()):\n"
    "    f = d / 'operstate'\n"
    "    if f.is_file(): print(d.name + ':' + f.read_text().strip())"
)

# The exception each network probe must die with when there is no network.
EXPECTED_NETWORK_ERROR = {
    "net_tcp": "Network is unreachable",
    "net_dns": "socket.gaierror",
    "net_http": "urllib.error.URLError",
}

# name -> command run inside the sandbox
PROBES: dict[str, list[str]] = {
    "uid": ["id", "-u"],
    "interfaces": ["python", "-c", INTERFACES_PROBE],
    "routes": ["cat", "/proc/net/route"],
    "net_tcp": ["python", "-c", TCP_PROBE],
    "net_dns": ["python", "-c", DNS_PROBE],
    "net_http": ["python", "-c", HTTP_PROBE],
    "write_root": ["sh", "-c", f"echo x > /{CANARY_NAME}"],
    "write_etc": ["sh", "-c", f"echo x > /etc/{CANARY_NAME}"],
    "write_usr": ["sh", "-c", f"echo x > /usr/local/bin/{CANARY_NAME}"],
    "write_home": ["sh", "-c", f"echo x > /home/sandbox/{CANARY_NAME}"],
    "docker_socket": ["test", "-e", "/var/run/docker.sock"],
    "write_workspace": ["sh", "-c", f"echo x > /workspace/{CANARY_NAME} && ls /workspace"],
    "decode": ["python", "-c",
               f"import pyModeS as pms; print(dict(pms.decode('{DEMO_FRAME}'))['callsign'])"],
    "tshark": ["tshark", "--version"],
}


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker daemon is not running")


# ---------------------------------------------------------------------------
# Layer 1: config guard
# ---------------------------------------------------------------------------


def isolation_violations(service: dict[str, Any]) -> list[str]:
    """Every way a compose service definition weakens the sandbox."""
    problems = []
    if service.get("network_mode") != "none":
        problems.append("network_mode must be 'none'")
    for key in ("networks", "ports", "expose", "extra_hosts", "dns", "links"):
        if service.get(key):
            problems.append(f"'{key}' must not be set")
    for key in ("volumes", "volumes_from", "devices"):
        if service.get(key):
            problems.append(f"'{key}' must not be set (nothing from the host may be mounted)")
    if service.get("read_only") is not True:
        problems.append("read_only must be true")
    if service.get("privileged"):
        problems.append("privileged must not be set")
    if service.get("cap_add"):
        problems.append("cap_add must not be set")
    if service.get("cap_drop") != ["ALL"]:
        problems.append("cap_drop must be [ALL]")
    if str(service.get("user", "root")).split(":")[0] in ("root", "0"):
        problems.append("user must be a non-root user")
    if "no-new-privileges:true" not in (service.get("security_opt") or []):
        problems.append("security_opt must include no-new-privileges:true")
    for key in ("pid", "ipc", "userns_mode"):
        if service.get(key) == "host":
            problems.append(f"{key} must not be 'host'")
    return problems


def load_services(path: Path) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(path.read_text())["services"]


def test_compose_file_has_no_isolation_violations() -> None:
    services = load_services(COMPOSE_FILE)
    assert "default" in services
    for name, service in services.items():
        assert isolation_violations(service) == [], name


def test_compose_file_declares_no_networks_or_volumes() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text())
    assert not compose.get("networks") and not compose.get("volumes")


@pytest.mark.parametrize("change, expected", [
    ({"network_mode": "bridge"}, "network_mode"),
    ({"network_mode": None}, "network_mode"),
    ({"ports": ["8080:8080"]}, "ports"),
    ({"volumes": ["./:/host"]}, "volumes"),
    ({"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}, "volumes"),
    ({"read_only": False}, "read_only"),
    ({"privileged": True}, "privileged"),
    ({"cap_drop": None}, "cap_drop"),
    ({"cap_add": ["NET_ADMIN"]}, "cap_add"),
    ({"user": "root"}, "user"),
    ({"user": None}, "user"),
    ({"security_opt": None}, "security_opt"),
])
def test_guard_detects_each_weakening(change: dict[str, Any], expected: str) -> None:
    """The guard itself is checked: every single weakening must be reported."""
    service = dict(load_services(COMPOSE_FILE)["default"])
    for key, value in change.items():
        if value is None:
            service.pop(key, None)
        else:
            service[key] = value
    assert any(expected in problem for problem in isolation_violations(service))


# ---------------------------------------------------------------------------
# Layer 2a: behaviour through Inspect AI
# ---------------------------------------------------------------------------


@solver
def run_probes() -> Solver:
    """Run every probe in the sample's sandbox and return the results as JSON."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        results = {}
        for name, cmd in PROBES.items():
            result = await sandbox().exec(cmd, timeout=PROBE_TIMEOUT_S)
            results[name] = {
                "success": result.success,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        state.output = ModelOutput.from_content("mockllm/model", json.dumps(results))
        return state

    return solve


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    task = Task(
        dataset=[Sample(input="isolation probes", target="")],
        solver=run_probes(),
        sandbox=("docker", str(COMPOSE_FILE)),
    )
    [log] = inspect_eval(task, model="mockllm/model",
                         log_dir=str(tmp_path_factory.mktemp("logs")))
    assert log.status == "success", log.error
    assert log.samples
    return json.loads(log.samples[0].output.completion)


@needs_docker
def test_inspect_starts_the_sandbox_and_tools_work(probes: dict[str, dict[str, Any]]) -> None:
    """Positive control: the sandbox is usable, not just locked down."""
    assert probes["decode"]["success"] and probes["decode"]["stdout"] == "ZZZ101"
    assert probes["tshark"]["success"] and "TShark" in probes["tshark"]["stdout"]
    assert probes["uid"]["stdout"] == "1000"


# --- Acceptance criterion 2: outbound connections are blocked ----------------


@needs_docker
@pytest.mark.parametrize("probe", ["net_tcp", "net_dns", "net_http"])
def test_outbound_connection_is_blocked(probes: dict[str, dict[str, Any]], probe: str) -> None:
    assert probes[probe]["success"] is False, probes[probe]
    # It failed because of the network, not because the probe itself is broken.
    assert EXPECTED_NETWORK_ERROR[probe] in probes[probe]["stderr"], probes[probe]["stderr"]


@needs_docker
def test_no_network_device_is_up_and_there_are_no_routes(probes: dict[str, dict[str, Any]]) -> None:
    states = dict(line.split(":") for line in probes["interfaces"]["stdout"].splitlines())
    assert "lo" in states
    assert "eth0" not in states
    assert all(state == "down" for name, state in states.items() if name != "lo"), states
    # /proc/net/route is a header line followed by one line per IPv4 route.
    assert probes["routes"]["success"]
    assert len(probes["routes"]["stdout"].splitlines()) == 1, probes["routes"]["stdout"]


# --- Acceptance criterion 1: host writes fail ---------------------------------


@needs_docker
@pytest.mark.parametrize("probe", ["write_root", "write_etc", "write_usr", "write_home"])
def test_write_outside_the_workspace_fails(probes: dict[str, dict[str, Any]], probe: str) -> None:
    assert probes[probe]["success"] is False, probes[probe]
    assert "Read-only file system" in probes[probe]["stderr"]


@needs_docker
def test_docker_socket_is_not_reachable(probes: dict[str, dict[str, Any]]) -> None:
    assert probes["docker_socket"]["success"] is False


@needs_docker
def test_workspace_write_stays_inside_the_container(probes: dict[str, dict[str, Any]]) -> None:
    """The one writable path is in-memory: the file exists there and nowhere on the host."""
    assert probes["write_workspace"]["success"]
    assert CANARY_NAME in probes["write_workspace"]["stdout"]
    for host_dir in (REPO_ROOT, SANDBOX_DIR, REPO_ROOT / "tests", Path.cwd(), Path.home()):
        assert not (host_dir / CANARY_NAME).exists(), host_dir


# ---------------------------------------------------------------------------
# Layer 2b: the running container, inspected from the host
# ---------------------------------------------------------------------------


def compose(project: str, file: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "--project-name", project, "--file", str(file), *args],
        capture_output=True, text=True, timeout=900)


def start(project: str, file: Path) -> str:
    """Start a compose project and return the container id of its default service."""
    up = compose(project, file, "up", "--detach", "--build")
    assert up.returncode == 0, up.stderr[-2000:]
    return compose(project, file, "ps", "--quiet", "default").stdout.strip()


def exec_in(container: str, cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", "exec", container, *cmd],
                          capture_output=True, text=True, timeout=60)


@pytest.fixture(scope="module")
def container_config() -> Iterator[dict[str, Any]]:
    project = f"sandbox-iso-{uuid.uuid4().hex[:8]}"
    try:
        container = start(project, COMPOSE_FILE)
        out = subprocess.run(["docker", "inspect", container],
                             capture_output=True, text=True, timeout=30, check=True)
        yield json.loads(out.stdout)[0]
    finally:
        compose(project, COMPOSE_FILE, "down", "--timeout", "1")


@needs_docker
def test_running_container_has_no_host_mounts(container_config: dict[str, Any]) -> None:
    host = container_config["HostConfig"]
    assert container_config["Mounts"] == []
    assert not host.get("Binds") and not host.get("VolumesFrom") and not host.get("Devices")
    assert host["ReadonlyRootfs"] is True


@needs_docker
def test_running_container_has_no_network(container_config: dict[str, Any]) -> None:
    assert container_config["HostConfig"]["NetworkMode"] == "none"
    assert list(container_config["NetworkSettings"]["Networks"]) == ["none"]
    assert not container_config["NetworkSettings"].get("Ports")


@needs_docker
def test_running_container_is_unprivileged(container_config: dict[str, Any]) -> None:
    host = container_config["HostConfig"]
    assert host["Privileged"] is False
    assert host["CapDrop"] == ["ALL"] and not host.get("CapAdd")
    assert "no-new-privileges:true" in host["SecurityOpt"]
    assert container_config["Config"]["User"] == "sandbox"


# ---------------------------------------------------------------------------
# Layer 3: controls
# ---------------------------------------------------------------------------


def host_is_online() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def weakened(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path]]:
    """The same image with networking on, a writable root and a host directory mounted."""
    host_dir = tmp_path_factory.mktemp("hostdir")
    host_dir.chmod(0o777)
    service = dict(load_services(COMPOSE_FILE)["default"])
    del service["network_mode"]
    service["read_only"] = False
    service["build"] = {"context": str(SANDBOX_DIR), "dockerfile": "Dockerfile"}
    service["volumes"] = [f"{host_dir}:/hostdir"]
    file = tmp_path_factory.mktemp("weakened") / "compose.yaml"
    file.write_text(yaml.safe_dump({"services": {"default": service}}))
    assert isolation_violations(service)  # the guard would have caught this file

    project = f"sandbox-weak-{uuid.uuid4().hex[:8]}"
    try:
        yield start(project, file), host_dir
    finally:
        compose(project, file, "down", "--timeout", "1")


@needs_docker
def test_control_network_probe_succeeds_when_networking_is_on(weakened: tuple[str, Path]) -> None:
    """The probe that is blocked in the sandbox works once networking is allowed."""
    if not host_is_online():
        pytest.skip("host has no network, so the control cannot run")
    container, _ = weakened
    result = exec_in(container, ["python", "-c", TCP_PROBE])
    assert result.returncode == 0, result.stderr


@needs_docker
def test_control_host_write_succeeds_when_a_directory_is_mounted(weakened: tuple[str, Path]) -> None:
    """With a host mount, a write does reach the host: the sandbox's refusal is real."""
    container, host_dir = weakened
    result = exec_in(container, ["sh", "-c", f"echo x > /hostdir/{CANARY_NAME}"])
    assert result.returncode == 0, result.stderr
    assert (host_dir / CANARY_NAME).exists()
