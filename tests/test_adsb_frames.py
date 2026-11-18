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
            truth = b["reference_position"]  # pyModeS local decode against the receiving sensor
            assert abs(lat - truth["latitude"]) < 1e-3
            assert abs((lon - truth["longitude"] + 180) % 360 - 180) < 1e-3
            checked += 1
    assert checked >= 30


def test_stale_pair_frame_exists_for_a2() -> None:
    """At least one track has an odd frame about 40 s after its previous frame (catalogue: 45 s)."""
    tracks: dict[str, list[float]] = {}
    for f in VALID:
        if f["message_type"] == "airborne_position":
            tracks.setdefault(f["track_id"], []).append(f["timestamp"])
    gaps = [max(b - a for a, b in zip(sorted(t), sorted(t)[1:])) for t in tracks.values()]
    assert any(g >= 40 for g in gaps)


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
def test_identifiers_are_synthetic_or_from_opensky(frame: dict[str, Any]) -> None:
    """Real identities only as received from an approved open source (OpenSky)."""
    if frame["source"] == "opensky":
        return  # checked against the OpenSky selection below
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


# --- Frames v1.1.0: real OpenSky receptions (issue #10, review finding F5) ---------


SELECTION = json.loads((DATA_DIR / MANIFEST["opensky_selection"]["file"]).read_text())


def true_position(frame: dict[str, Any]) -> tuple[float, float]:
    """pyModeS' local decode against the receiving sensor, kept with each real position."""
    ref = frame["reference_position"]
    return ref["latitude"], ref["longitude"]


def _tracks() -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for f in VALID:
        if f["track_id"] is None:
            continue
        t = tracks.setdefault(f["track_id"], {"positions": [], "velocity": None})
        if f["message_type"] == "airborne_position":
            t["positions"].append(f)
        elif f["message_type"] == "airborne_velocity":
            t["velocity"] = f
    for t in tracks.values():
        t["positions"].sort(key=lambda f: f["timestamp"])
    return tracks


TRACKS = _tracks()
ALTITUDE_STEP_FT = 25  # altitude reporting resolution (Q=1)


def test_track_frames_are_the_selected_opensky_messages() -> None:
    """Every DF17 track frame is a real reception, byte for byte, at its reception time."""
    selected = {}
    for track_id, track in SELECTION["tracks"].items():
        for msg in [track["identification"], *track["positions"], track["velocity"]]:
            selected[msg["hex"]] = (track_id, msg["t"])
    real = [f for f in VALID if f["df"] == 17]
    assert len(real) == len(selected) == 61
    for f in real:
        assert selected[f["hex"]] == (f["track_id"], f["timestamp"]), f["frame_id"]
        assert f["source"] == "opensky" and f["synthesis"] is None


def test_selection_keeps_no_sensor_data() -> None:
    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    assert SELECTION["source"] == "opensky"
    assert not keys(SELECTION) & {"sensorLatitude", "sensorLongitude", "sensorSerialNumber",
                                  "sensorType", "sensorAltitude", "RSSIPacket"}


def test_selection_file_matches_the_manifest() -> None:
    path = DATA_DIR / MANIFEST["opensky_selection"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["opensky_selection"]["sha256"]


def test_constructed_frames_are_tagged_and_use_synthetic_identities() -> None:
    for f in FRAMES:
        if f["label"] == "corrupt":
            source = next(r for r in FRAMES if r["frame_id"] == f["corruption"]["source_frame_id"])
            assert f["source"] == source["source"], f["frame_id"]
        elif f["df"] == 18:
            assert f["source"] == "spec-synthesized"
            assert 0xF00000 <= int(f["icao"], 16) <= 0xF07FFF
            assert f["expected"]["callsign"].startswith("ZZZ")


def test_real_callsigns_are_not_registrations() -> None:
    """Real identities are allowed (OpenSky); registrations are still kept out (bench/boundary.py)."""
    import re

    for f in VALID:
        if f["df"] == 17 and "callsign" in f["expected"]:
            assert re.fullmatch(r"[A-Z]{3}[0-9][0-9A-Z]{0,3}", f["expected"]["callsign"]), f["frame_id"]


@pytest.mark.parametrize("track_id", sorted(TRACKS))
def test_altitude_follows_the_vertical_rate(track_id: str) -> None:
    """Over each track's position run (under 3 s), altitude changes match the reported rate."""
    track = TRACKS[track_id]
    rate = track["velocity"]["expected"]["vertical_rate"]
    first = track["positions"][0]
    for later in track["positions"][1:4]:
        elapsed_min = (later["timestamp"] - first["timestamp"]) / 60
        change = later["expected"]["altitude"] - first["expected"]["altitude"]
        expected = rate * elapsed_min
        assert abs(change - expected) <= ALTITUDE_STEP_FT + 0.25 * abs(expected), (
            f"{later['frame_id']}: altitude changed {change} ft in {elapsed_min * 60:.1f} s "
            f"at {rate} ft/min (expected about {expected:.0f} ft)")


def test_long_interval_climb_changes_altitude() -> None:
    """The review's case: trk-04's late stale-pair frame is well above its pair.

    Real aircraft change rate, so the gain is checked for direction and size, not
    against the rate reported at the start (RYR39XF eases from 1,728 ft/min).
    """
    track = TRACKS["trk-04"]
    first, late = track["positions"][0], track["positions"][-1]
    assert track["velocity"]["expected"]["vertical_rate"] > 0
    assert late["timestamp"] - first["timestamp"] >= 40
    assert late["expected"]["altitude"] - first["expected"]["altitude"] >= 500


def test_level_tracks_stay_level() -> None:
    for track_id, track in TRACKS.items():
        if abs(track["velocity"]["expected"]["vertical_rate"]) <= 128:
            alts = {p["expected"]["altitude"] for p in track["positions"]}
            assert max(alts) - min(alts) <= ALTITUDE_STEP_FT, track_id


@pytest.mark.parametrize("track_id", sorted(TRACKS))
def test_velocity_agrees_with_the_positions(track_id: str) -> None:
    """The direction the positions move matches the velocity message's track angle.

    Speed is not compared: reception times are OpenSky server times, whose network
    jitter is large against a 2 s window.
    """
    import math

    track = TRACKS[track_id]
    (lat_a, lon_a), (lat_b, lon_b) = true_position(track["positions"][0]), true_position(track["positions"][3])
    north = math.radians(lat_b - lat_a)
    east = math.radians((lon_b - lon_a + 180) % 360 - 180) * math.cos(math.radians(lat_a))
    moved = math.degrees(math.atan2(east, north)) % 360
    assert abs((moved - track["velocity"]["expected"]["track"] + 180) % 360 - 180) <= 10


def test_v1_0_0_is_frozen_and_the_layout_carries_over() -> None:
    [old] = [e for e in MANIFEST["frozen"] if e["version"] == "1.0.0"]
    old_path = DATA_DIR / old["file"]
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old["sha256"] == \
        "f48ea461bccff000e79cdb520fa7f576c6242ae51999620abe81857a433d352d"
    before = {f["frame_id"]: f for f in map(json.loads, old_path.read_text().splitlines())}
    assert set(before) == set(ids(FRAMES))
    for f in FRAMES:
        b = before[f["frame_id"]]
        assert (f["message_type"], f["label"], f["track_id"], f["df"]) == \
            (b["message_type"], b["label"], b["track_id"], b["df"]), f["frame_id"]
        if f["df"] == 18 and f["label"] == "valid":
            assert f["hex"] == b["hex"]  # the synthetic DF18 frames are unchanged
