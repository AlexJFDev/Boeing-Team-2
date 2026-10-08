"""Checks for the ADS-B frame set (WBS 2.1, issue #10).

Acceptance criteria:
- When the frame set is loaded, each frame has its hex string, ICAO address, type code and expected decoded fields.
- When a frame has a bad CRC, it is labeled corrupt with the expected answer 'reject'.

Ground truth is re-derived here with pyModeS rather than with the generator's own encoder, so the test does not just check the generator against itself.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyModeS as pms
import pytest
from inspect_ai.dataset import Sample, json_dataset
from pyModeS import position

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "adsb"
MANIFEST = json.loads((DATA_DIR / "MANIFEST.json").read_text())
FRAMES_PATH = DATA_DIR / MANIFEST["file"]
FRAMES: list[dict[str, Any]] = [json.loads(line) for line in FRAMES_PATH.read_text().splitlines()]
VALID = [f for f in FRAMES if f["label"] == "valid"]
CORRUPT = [f for f in FRAMES if f["label"] == "corrupt"]

REQUIRED_FIELDS = {"frame_id", "hex", "df", "icao", "type_code", "message_type",
                   "crc_valid", "label", "expected_answer", "expected"}


def ids(frames: list[dict[str, Any]]) -> list[str]:
    return [f["frame_id"] for f in frames]


# --- Provenance -------------------------------------------------------------


def test_manifest_hash_matches_file() -> None:
    assert hashlib.sha256(FRAMES_PATH.read_bytes()).hexdigest() == MANIFEST["sha256"]
    assert MANIFEST["frame_count"] == len(FRAMES)


def test_reference_decoder_version_matches_environment() -> None:
    assert MANIFEST["reference_decoder"] == f"pyModeS=={version('pyModeS')}"


def test_generator_is_deterministic(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_frames", REPO_ROOT / "scripts" / "adsb" / "build_frames.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    module.write(tmp_path)
    assert (tmp_path / MANIFEST["file"]).read_bytes() == FRAMES_PATH.read_bytes()


def test_frame_ids_are_unique() -> None:
    assert len(set(ids(FRAMES))) == len(FRAMES)


# --- Acceptance criterion 1: fields present and correct -----------------------


@pytest.mark.parametrize("frame", FRAMES, ids=ids(FRAMES))
def test_frame_has_required_fields(frame: dict[str, Any]) -> None:
    assert REQUIRED_FIELDS <= frame.keys()
    assert len(frame["hex"]) == 28 and int(frame["hex"], 16) >= 0
    assert len(frame["icao"]) == 6 and int(frame["icao"], 16) >= 0
    assert frame["df"] in (17, 18)
    assert 0 <= frame["type_code"] <= 31
    assert frame["expected"]


@pytest.mark.parametrize("frame", VALID, ids=ids(VALID))
def test_valid_frame_matches_reference_decoder(frame: dict[str, Any]) -> None:
    decoded = dict(pms.decode(frame["hex"]))
    assert decoded["crc_valid"] is True
    assert frame["crc_valid"] is True and frame["expected_answer"] == "decode"
    assert frame["expected"] == decoded
    assert frame["icao"] == decoded["icao"]
    assert frame["type_code"] == decoded["typecode"]


def test_set_covers_the_catalogue_cases() -> None:
    kinds = {(f["df"], f["message_type"]) for f in VALID}
    assert {(17, "identification"), (17, "airborne_position"),
            (17, "airborne_velocity"), (18, "identification")} <= kinds
    assert {f["cf"] for f in FRAMES if f["df"] == 18} >= {0, 6}


def test_even_odd_pairs_resolve_to_true_position() -> None:
    """Consecutive even/odd frames under 10 s apart decode to the encoded position."""
    tracks: dict[str, list[dict[str, Any]]] = {}
    for f in VALID:
        if f["message_type"] == "airborne_position":
            tracks.setdefault(f["track_id"], []).append(f)
    checked = 0
    for frames in tracks.values():
        frames.sort(key=lambda f: f["timestamp"])
        for a, b in zip(frames, frames[1:]):
            if a["expected"]["cpr_format"] == b["expected"]["cpr_format"]:
                continue
            if b["timestamp"] - a["timestamp"] > 10:
                continue
            even, odd = (a, b) if a["expected"]["cpr_format"] == 0 else (b, a)
            result = position.airborne_position_pair(
                even["expected"]["cpr_lat"], even["expected"]["cpr_lon"],
                odd["expected"]["cpr_lat"], odd["expected"]["cpr_lon"],
                even_is_newer=even is b)
            assert result is not None
            lat, lon = result
            truth = b["synthesis"]
            assert abs(lat - truth["true_latitude"]) < 1e-3
            assert abs((lon - truth["true_longitude"] + 180) % 360 - 180) < 1e-3
            checked += 1
    assert checked >= 30


def test_stale_pair_frame_exists_for_a2() -> None:
    """At least one track has an odd frame 45 s after its previous frame."""
    tracks: dict[str, list[float]] = {}
    for f in VALID:
        if f["message_type"] == "airborne_position":
            tracks.setdefault(f["track_id"], []).append(f["timestamp"])
    gaps = [max(b - a for a, b in zip(sorted(t), sorted(t)[1:])) for t in tracks.values()]
    assert any(g >= 45 for g in gaps)


# --- Acceptance criterion 2 (negative case): corrupt frames -----------------


def test_set_contains_corrupt_frames() -> None:
    assert len(CORRUPT) >= 5
    assert any(f["df"] == 18 for f in CORRUPT)


@pytest.mark.parametrize("frame", CORRUPT, ids=ids(CORRUPT))
def test_corrupt_frame_is_labeled_reject(frame: dict[str, Any]) -> None:
    assert dict(pms.decode(frame["hex"]))["crc_valid"] is False
    assert frame["crc_valid"] is False
    assert frame["expected_answer"] == "reject"
    assert frame["expected"]["answer"] == "reject"
    assert frame["corruption"]["source_frame_id"] in ids(VALID)


@pytest.mark.parametrize("frame", FRAMES, ids=ids(FRAMES))
def test_label_agrees_with_crc(frame: dict[str, Any]) -> None:
    """No frame with a failing CRC is labeled valid, and no valid CRC is labeled corrupt."""
    crc_ok = dict(pms.decode(frame["hex"]))["crc_valid"]
    assert (frame["label"] == "valid") == crc_ok


# --- Data boundary (CLAUDE.md rule 1) ---------------------------------------


@pytest.mark.parametrize("frame", VALID, ids=ids(VALID))
def test_only_synthetic_identifiers(frame: dict[str, Any]) -> None:
    assert 0xF00000 <= int(frame["icao"], 16) <= 0xF07FFF
    callsign = frame["expected"].get("callsign")
    if callsign is not None:
        assert callsign.startswith("ZZZ")


# --- Inspect AI compatibility ----------------------------------------------


def test_loads_as_inspect_dataset() -> None:
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            id=record["frame_id"],
            input=record["hex"],
            target=json.dumps(record["expected"], sort_keys=True),
            metadata={k: v for k, v in record.items() if k not in ("hex", "expected")},
        )

    dataset = json_dataset(str(FRAMES_PATH), sample_fields=record_to_sample)
    assert len(dataset) == len(FRAMES)
    first = dataset[0]
    assert first.id == FRAMES[0]["frame_id"] and first.input == FRAMES[0]["hex"]
