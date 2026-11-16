"""The ADS-B frame set by version (data/adsb, WBS 2.1).

data/adsb/MANIFEST.json describes the current frame set and, under "frozen", every
earlier version, each with its file name and sha256. Task families pin the version
they were built from, so a new frame set never changes what a frozen task file
refers to: A1/A2 v1.0.0 and v1.1.0 use frames v1.0.0, A1/A2 v1.2.0 use v1.1.0.

    from bench.adsb.frames import load_frames
    frames = load_frames("1.1.0")
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from bench.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = REPO_ROOT / "data" / "adsb"
FRAMES_MANIFEST = FRAMES_DIR / "MANIFEST.json"


class FrameSetError(ValueError):
    """A frame-set version is not in the manifest, or its file does not match it."""


def manifest() -> dict[str, Any]:
    return json.loads(FRAMES_MANIFEST.read_text())


def versions() -> dict[str, dict[str, Any]]:
    """Version -> manifest entry, for the current and every frozen frame set."""
    current = manifest()
    entries = [{k: v for k, v in current.items() if k != "frozen"}, *current.get("frozen", [])]
    return {str(e["version"]): e for e in entries}


def entry(version: str) -> dict[str, Any]:
    """The manifest entry for one version, after checking its file's sha256."""
    found = versions().get(version)
    if found is None:
        raise FrameSetError(f"frame set v{version} is not in {FRAMES_MANIFEST.name} "
                            f"(have {', '.join(sorted(versions()))})")
    actual = sha256_file(FRAMES_DIR / found["file"])
    if actual != found["sha256"]:
        raise FrameSetError(f"{found['file']} does not match the sha256 in {FRAMES_MANIFEST.name}")
    return found


@functools.cache
def load_frames(version: str) -> tuple[dict[str, Any], ...]:
    """The frames of one version, in file order, after checking the file's sha256."""
    path = FRAMES_DIR / entry(version)["file"]
    return tuple(json.loads(line) for line in path.read_text().splitlines())


def frames_by_id(version: str) -> dict[str, dict[str, Any]]:
    return {f["frame_id"]: f for f in load_frames(version)}
