"""Checks for the independent CPR position verifier (WBS 2.2, issue #11).

Acceptance criteria:
- When given a valid even/odd pair under 10 s apart, the computed position
  matches pyModeS within tolerance.
- When the pair is more than 10 s apart, the verifier reports 'ambiguous'
  instead of a position.
"""

from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pyModeS import position

from bench.adsb import cpr
from bench.adsb.cpr import CPRFrame, decode_hex_pair, decode_pair, parse_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "adsb"
MANIFEST = json.loads((DATA_DIR / "MANIFEST.json").read_text())
FRAMES: list[dict[str, Any]] = [
    json.loads(line) for line in (DATA_DIR / MANIFEST["file"]).read_text().splitlines()]
POSITIONS = [f for f in FRAMES if f["message_type"] == "airborne_position" and f["label"] == "valid"]

# Agreement with pyModeS. Both implement the same arithmetic, so they should
# agree to floating-point noise; 1e-6 degrees is about 0.1 m.
TOLERANCE_DEG = 1e-6

# The demo pair: trk-01, even then odd, 0.5 s apart.
DEMO_EVEN = ("8DF00A1158BF01F0FDC0B93FD599", 1000.2)
DEMO_ODD = ("8DF00A1158BF058266265571F0C7", 1000.7)


def load_encoder() -> ModuleType:
    """The 2.1 frame encoder, used only to build test inputs."""
    spec = importlib.util.spec_from_file_location(
        "build_frames", REPO_ROOT / "scripts" / "adsb" / "build_frames.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENCODER = load_encoder()


def frame_of(record: dict[str, Any]) -> CPRFrame:
    e = record["expected"]
    return CPRFrame(e["cpr_format"], e["cpr_lat"], e["cpr_lon"], record["timestamp"])


def pymodes_pair(even: CPRFrame, odd: CPRFrame) -> tuple[float, float] | None:
    return position.airborne_position_pair(
        even.cpr_lat, even.cpr_lon, odd.cpr_lat, odd.cpr_lon,
        even_is_newer=even.timestamp >= odd.timestamp)


def assert_matches_pymodes(a: CPRFrame, b: CPRFrame) -> None:
    even, odd = (a, b) if a.cpr_format == 0 else (b, a)
    expected = pymodes_pair(even, odd)
    result = decode_pair(a, b)
    if expected is None:
        assert result.status == "ambiguous" and result.reason == "zone_mismatch"
        return
    assert result.status == "ok"
    assert result.latitude is not None and result.longitude is not None
    assert abs(result.latitude - expected[0]) < TOLERANCE_DEG
    assert abs(result.longitude - expected[1]) < TOLERANCE_DEG


def fresh_pairs() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every consecutive even/odd pair in the 2.1 set that is under 10 s apart."""
    tracks: dict[str, list[dict[str, Any]]] = {}
    for f in POSITIONS:
        tracks.setdefault(f["track_id"], []).append(f)
    pairs = []
    for frames in tracks.values():
        frames.sort(key=lambda f: f["timestamp"])
        for a, b in zip(frames, frames[1:]):
            if a["expected"]["cpr_format"] != b["expected"]["cpr_format"] \
                    and b["timestamp"] - a["timestamp"] <= 10:
                pairs.append((a, b))
    return pairs


FRESH = fresh_pairs()


# --- Demo -------------------------------------------------------------------


def test_demo_pair_decodes_correctly() -> None:
    result = decode_hex_pair(DEMO_EVEN[0], DEMO_EVEN[1], DEMO_ODD[0], DEMO_ODD[1])
    assert result.status == "ok"
    assert result.latitude == pytest.approx(38.912596621755824, abs=TOLERANCE_DEG)
    assert result.longitude == pytest.approx(-71.40106201171875, abs=TOLERANCE_DEG)
    # And within CPR resolution (about 5 m) of where the frame was encoded.
    assert result.latitude == pytest.approx(38.912579, abs=1e-3)
    assert result.longitude == pytest.approx(-71.401078, abs=1e-3)


def test_cli_prints_the_demo_result() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "bench.adsb.cpr", DEMO_EVEN[0], DEMO_ODD[0],
         "--t-a", str(DEMO_EVEN[1]), "--t-b", str(DEMO_ODD[1])],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    printed = json.loads(out.stdout)
    assert printed["status"] == "ok" and printed["verifier_version"] == cpr.VERIFIER_VERSION


# --- Acceptance criterion 1: valid pairs match pyModeS -----------------------


def test_set_has_enough_fresh_pairs() -> None:
    assert len(FRESH) >= 30


@pytest.mark.parametrize("pair", FRESH, ids=[f"{a['frame_id']}+{b['frame_id']}" for a, b in FRESH])
def test_frame_set_pairs_match_pymodes(pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
    a, b = pair
    assert_matches_pymodes(frame_of(a), frame_of(b))
    # Argument order must not matter.
    assert decode_pair(frame_of(a), frame_of(b)) == decode_pair(frame_of(b), frame_of(a))


@pytest.mark.parametrize("pair", FRESH, ids=[f"{a['frame_id']}+{b['frame_id']}" for a, b in FRESH])
def test_frame_set_pairs_match_true_position(pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
    a, b = pair
    result = decode_pair(frame_of(a), frame_of(b))
    truth = b["synthesis"]  # the newer frame defines the position
    assert result.latitude == pytest.approx(truth["true_latitude"], abs=1e-3)
    assert abs((result.longitude - truth["true_longitude"] + 180) % 360 - 180) < 1e-3


def test_random_positions_match_pymodes() -> None:
    """2,000 seeded random pairs across the globe, both newer-frame choices."""
    rng = random.Random(20261008)
    zone_mismatches = 0
    for _ in range(2000):
        lat, lon = rng.uniform(-86.9, 86.9), rng.uniform(-180, 179.999)
        lat2 = max(-87.0, min(87.0, lat + rng.uniform(-0.02, 0.02)))
        lon2 = (lon + rng.uniform(-0.02, 0.02) + 180) % 360 - 180
        t = rng.uniform(0, 1000)
        dt = rng.uniform(0, 10)
        ye, xe = ENCODER.cpr_encode(lat, lon, 0)
        yo, xo = ENCODER.cpr_encode(lat2, lon2, 1)
        if rng.random() < 0.5:
            even, odd = CPRFrame(0, ye, xe, t), CPRFrame(1, yo, xo, t + dt)
        else:
            even, odd = CPRFrame(0, ye, xe, t + dt), CPRFrame(1, yo, xo, t)
        assert_matches_pymodes(even, odd)
        zone_mismatches += decode_pair(even, odd).reason == "zone_mismatch"
    assert zone_mismatches < 2000  # sanity: most pairs resolve


def test_nl_matches_pymodes_table() -> None:
    """The closed-form NL agrees with pyModeS's precomputed table across latitudes."""
    for i in range(-9000, 9001):
        lat = i / 100
        assert cpr.nl(lat) == position.cprNL(lat), lat


# --- Acceptance criterion 2 (negative case): stale pairs ----------------------


def test_pair_more_than_10s_apart_is_ambiguous() -> None:
    late = [f for f in POSITIONS if f["track_id"] == "trk-04"]
    late.sort(key=lambda f: f["timestamp"])
    even, stale_odd = late[2], late[-1]  # 45.5 s apart
    assert even["expected"]["cpr_format"] == 0 and stale_odd["expected"]["cpr_format"] == 1
    result = decode_pair(frame_of(even), frame_of(stale_odd))
    assert result.status == "ambiguous"
    assert result.reason == "pair_too_old"
    assert result.latitude is None and result.longitude is None
    assert result.pair_age_s == pytest.approx(45.5)


@pytest.mark.parametrize("gap, status", [(0.0, "ok"), (9.999, "ok"), (10.0, "ok"),
                                         (10.001, "ambiguous"), (45.0, "ambiguous")])
def test_ten_second_boundary(gap: float, status: str) -> None:
    result = decode_hex_pair(DEMO_EVEN[0], 1000.0, DEMO_ODD[0], 1000.0 + gap)
    assert result.status == status
    if status == "ambiguous":
        assert result.reason == "pair_too_old" and result.latitude is None


def test_stale_pair_is_ambiguous_even_when_it_would_decode() -> None:
    """A stale pair from a parked aircraft would still compute a position; it must not."""
    ye, xe = ENCODER.cpr_encode(38.9, -71.4, 0)
    yo, xo = ENCODER.cpr_encode(38.9, -71.4, 1)
    assert pymodes_pair(CPRFrame(0, ye, xe, 0), CPRFrame(1, yo, xo, 60)) is not None
    result = decode_pair(CPRFrame(0, ye, xe, 0), CPRFrame(1, yo, xo, 60))
    assert result.status == "ambiguous" and result.reason == "pair_too_old"


# --- Other cases that must not produce a position ---------------------------


def test_two_even_frames_are_ambiguous() -> None:
    evens = [f for f in POSITIONS if f["track_id"] == "trk-01" and f["expected"]["cpr_format"] == 0]
    result = decode_pair(frame_of(evens[0]), frame_of(evens[1]))
    assert result.status == "ambiguous" and result.reason == "same_format"
    assert result.latitude is None


def test_pair_straddling_a_zone_boundary_is_ambiguous() -> None:
    """Frames on either side of the NL 59/58 boundary (10.4705 deg) cannot be combined."""
    found = 0
    for k in range(-50, 51):
        lat_e = 10.47047 + k * 0.0002
        lat_o = lat_e + 0.004
        ye, xe = ENCODER.cpr_encode(lat_e, 20.0, 0)
        yo, xo = ENCODER.cpr_encode(lat_o, 20.0, 1)
        even, odd = CPRFrame(0, ye, xe, 0.0), CPRFrame(1, yo, xo, 1.0)
        assert_matches_pymodes(even, odd)
        if decode_pair(even, odd).reason == "zone_mismatch":
            found += 1
            assert decode_pair(even, odd).latitude is None
    assert found > 0


# --- Frame parsing ----------------------------------------------------------


@pytest.mark.parametrize("record", POSITIONS, ids=[f["frame_id"] for f in POSITIONS])
def test_parse_frame_matches_pymodes_fields(record: dict[str, Any]) -> None:
    parsed = parse_frame(record["hex"], record["timestamp"])
    assert parsed == frame_of(record)


CORRUPT_POSITIONS = [f for f in FRAMES if f["label"] == "corrupt" and f["message_type"] == "airborne_position"]


@pytest.mark.parametrize("record", CORRUPT_POSITIONS, ids=[f["frame_id"] for f in CORRUPT_POSITIONS])
def test_corrupt_frame_is_rejected(record: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="CRC"):
        parse_frame(record["hex"], 0.0)


def test_non_position_frame_is_rejected() -> None:
    ident = next(f for f in FRAMES if f["message_type"] == "identification" and f["label"] == "valid")
    with pytest.raises(ValueError, match="type code"):
        parse_frame(ident["hex"], 0.0)


@pytest.mark.parametrize("bad", ["8DF00A11", "zz" * 14])
def test_malformed_hex_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_frame(bad, 0.0)


def test_out_of_range_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        CPRFrame(2, 0, 0, 0.0)
    with pytest.raises(ValueError):
        CPRFrame(0, 2**17, 0, 0.0)
