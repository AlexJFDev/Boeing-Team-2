"""Checks for the Docker sandbox image (WBS 1.8, issue #8).

Acceptance criteria:
- When the image is built from the Dockerfile, the tool versions it reports
  match the pinned lockfile.
- When a pinned package version is unavailable, the build fails loudly instead
  of installing a different version.

These tests build images, so they need a running Docker daemon and network
access to the package mirrors. They are skipped when Docker is not available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DIR = REPO_ROOT / "sandbox"
TEST_IMAGE = "boeing-team-2-sandbox:test"

BUILD_TIMEOUT_S = 900


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not docker_available(), reason="Docker daemon is not running")


def lock_lines(path: Path) -> list[str]:
    """Pins from a lockfile: comments, blank lines and --hash continuations dropped."""
    pins = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().rstrip("\\").strip()
        if line and not line.startswith("--"):
            pins.append(line)
    return pins


def build(context: Path, tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "build", "--tag", tag, str(context)],
        capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)


@pytest.fixture(scope="module")
def image() -> str:
    result = build(SANDBOX_DIR, TEST_IMAGE)
    assert result.returncode == 0, result.stderr[-2000:]
    return TEST_IMAGE


# --- Acceptance criterion 1: reported versions match the lockfiles ----------


def test_lockfiles_pin_exact_versions() -> None:
    apt = lock_lines(SANDBOX_DIR / "apt.lock")
    pip = lock_lines(SANDBOX_DIR / "requirements.lock")
    assert apt and all("=" in pin and "*" not in pin for pin in apt)
    assert pip and all("==" in pin for pin in pip)
    assert {pin.split("==")[0] for pin in pip} == {"pyModeS", "scapy"}
    assert [pin.split("=")[0] for pin in apt] == ["tshark"]


def test_base_image_is_pinned_by_digest() -> None:
    from_lines = [line for line in (SANDBOX_DIR / "Dockerfile").read_text().splitlines()
                  if line.startswith("FROM ")]
    assert from_lines and all("@sha256:" in line for line in from_lines)


def test_reported_versions_match_lockfiles(image: str) -> None:
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image, "sandbox-versions"],
        capture_output=True, text=True, timeout=60, check=True)
    reported = sorted(out.stdout.split())
    pinned = sorted(lock_lines(SANDBOX_DIR / "apt.lock")
                    + lock_lines(SANDBOX_DIR / "requirements.lock"))
    assert reported == pinned


def test_sandbox_pymodes_matches_reference_decoder(image: str) -> None:
    """The sandbox decoder must be the version the ADS-B ground truth was built with."""
    import json
    manifest = json.loads((REPO_ROOT / "data" / "adsb" / "MANIFEST.json").read_text())
    assert manifest["reference_decoder"] in lock_lines(SANDBOX_DIR / "requirements.lock")


def test_tools_run_as_unprivileged_user(image: str) -> None:
    script = "id -u && tshark --version | head -1 && python -c 'import pyModeS, scapy'"
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image, "sh", "-c", script],
        capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.splitlines()[0] == "1000"
    assert "TShark" in out.stdout


# --- Acceptance criterion 2 (negative case): unavailable pins fail the build --


def broken_context(tmp_path: Path, lockfile: str, old: str, new: str) -> Path:
    context = tmp_path / "sandbox"
    shutil.copytree(SANDBOX_DIR, context)
    path = context / lockfile
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    return context


def test_unavailable_apt_version_fails_the_build(tmp_path: Path) -> None:
    pin = lock_lines(SANDBOX_DIR / "apt.lock")[0]
    context = broken_context(tmp_path, "apt.lock", pin, "tshark=0.0.0-unavailable")
    result = build(context, "boeing-team-2-sandbox:should-not-exist-apt")
    assert result.returncode != 0
    assert "0.0.0-unavailable" in result.stdout + result.stderr


def test_unavailable_pip_version_fails_the_build(tmp_path: Path) -> None:
    context = broken_context(tmp_path, "requirements.lock", "scapy==2.8.0", "scapy==0.0.0")
    result = build(context, "boeing-team-2-sandbox:should-not-exist-pip")
    assert result.returncode != 0
    assert "scapy==0.0.0" in result.stdout + result.stderr


def test_wrong_hash_fails_the_build(tmp_path: Path) -> None:
    """Same version number but different contents must not install."""
    lock = (SANDBOX_DIR / "requirements.lock").read_text()
    real = next(part for part in lock.split() if part.startswith("--hash=sha256:"))
    context = broken_context(tmp_path, "requirements.lock", real, "--hash=sha256:" + "0" * 64)
    result = build(context, "boeing-team-2-sandbox:should-not-exist-hash")
    assert result.returncode != 0
