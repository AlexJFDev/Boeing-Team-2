"""Cross-checks between the pyModeS oracle and the independent CPR verifier (WBS 2.4, issue #13).

Acceptance criteria:
- When the tests run, the oracle and CPR verifier agree on 100% of valid frames.
- When they disagree on any frame, the test fails and lists that frame.

The two tools are built separately (the oracle wraps pyModeS; the verifier is
written from DO-260B), so agreement here is real evidence that both are right.
"Agree" means, per frame:

- valid airborne position frame: both decode it, with identical CPR format,
  latitude and longitude fields
- valid frame of another type: the oracle decodes it, the verifier declines it
  as not a position message
- corrupt frame: the oracle rejects it and the verifier raises a CRC error
- invalid input: the oracle returns invalid_input and the verifier raises

Pairs are checked too: the verifier's position, or its "ambiguous" answer, must
match what the oracle's decoder (pyModeS's pair matching) produces.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyModeS as pms
import pytest

from bench.adsb import cpr, oracle

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "adsb"
MANIFEST = json.loads((DATA_DIR / "MANIFEST.json").read_text())
FRAMES: list[dict[str, Any]] = [
    json.loads(line) for line in (DATA_DIR / MANIFEST["file"]).read_text().splitlines()]
VALID = [f for f in FRAMES if f["label"] == "valid"]
CORRUPT = [f for f in FRAMES if f["label"] == "corrupt"]
POSITIONS = [f for f in VALID if f["message_type"] == "airborne_position"]

POSITION_TOLERANCE_DEG = 1e-6  # about 0.1 m


def load_encoder() -> Any:
    """The 2.1 frame encoder, used only to generate extra test frames."""
    spec = importlib.util.spec_from_file_location(
        "build_frames", REPO_ROOT / "scripts" / "adsb" / "build_frames.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENCODER = load_encoder()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One input to compare, with a label used in failure messages."""

    label: str
    frame: object


def compare(case: Case,
            parse: Callable[[str, float], cpr.CPRFrame] = cpr.parse_frame,
            decode: Callable[[object], oracle.OracleResult] = oracle.decode) -> str | None:
    """Describe how the oracle and verifier disagree on one input, or None if they agree."""
    result = decode(case.frame)
    try:
        parsed: cpr.CPRFrame | None = parse(case.frame, 0.0)  # type: ignore[arg-type]
        refusal = None
    except (ValueError, TypeError, AttributeError) as exc:
        parsed, refusal = None, str(exc)

    where = f"{case.label} ({case.frame!r})"
    if result.status == "decoded" and result.answer["message_type"] == "airborne_position":
        if parsed is None:
            return f"{where}: oracle decoded a position, verifier refused: {refusal}"
        verifier_fields = {"cpr_format": "odd" if parsed.cpr_format else "even",
                           "cpr_lat": parsed.cpr_lat, "cpr_lon": parsed.cpr_lon}
        diffs = [f"{k} oracle={result.answer[k]!r} verifier={v!r}"
                 for k, v in verifier_fields.items() if result.answer[k] != v]
        return f"{where}: " + "; ".join(diffs) if diffs else None
    if parsed is not None:
        return f"{where}: verifier parsed a position, oracle said {result.status}" + (
            f" ({result.answer.get('message_type')})" if result.status == "decoded" else "")
    if result.status == "rejected" and "CRC" not in (refusal or ""):
        return f"{where}: oracle rejected the CRC, verifier refused for another reason: {refusal}"
    return None


def assert_all_agree(cases: Iterable[Case], **tools: Any) -> int:
    """Compare every case; fail once, listing every frame that disagrees. Returns the count."""
    cases = list(cases)
    problems = [p for p in (compare(c, **tools) for c in cases) if p]
    if problems:
        agree = len(cases) - len(problems)
        raise AssertionError(
            f"Oracle and CPR verifier disagree on {len(problems)} of {len(cases)} frames "
            f"({100 * agree / len(cases):.1f}% agreement):\n  " + "\n  ".join(problems))
    return len(cases)


def random_position_frames(n: int, seed: int) -> list[Case]:
    """Seeded, synthetic DF17/DF18 airborne position frames across the globe."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        lat, lon = rng.uniform(-89.9, 89.9), rng.uniform(-180, 179.999)
        tc = rng.choice(list(range(9, 19)))
        alt = rng.randrange(0, 45000, 25)
        df = rng.choice([17, 18])
        ca_cf = 5 if df == 17 else rng.choice([0, 6])
        icao = rng.randrange(0xF00000, 0xF08000)
        me = ENCODER.me_airborne_position(tc, alt, rng.randint(0, 1), lat, lon)
        cases.append(Case(f"random-{i:04d}", ENCODER.assemble(df, ca_cf, icao, me)))
    return cases


# ---------------------------------------------------------------------------
# Acceptance criterion 1: 100% agreement on valid frames
# ---------------------------------------------------------------------------


def test_agree_on_every_valid_frame_in_the_set() -> None:
    assert assert_all_agree(Case(f["frame_id"], f["hex"]) for f in VALID) == len(VALID) == 64


def test_agree_on_every_position_frame_in_the_set() -> None:
    """The 41 position frames are where both tools produce fields; none may differ."""
    assert assert_all_agree(Case(f["frame_id"], f["hex"]) for f in POSITIONS) == 41


def test_agree_on_5000_random_position_frames() -> None:
    assert assert_all_agree(random_position_frames(5000, seed=20261008)) == 5000


def test_agree_on_every_corrupt_frame() -> None:
    """Both refuse corrupt frames: the oracle rejects them, the verifier raises a CRC error."""
    assert assert_all_agree(Case(f["frame_id"], f["hex"]) for f in CORRUPT) == 7


@pytest.mark.parametrize("bad", ["", "8DF00A11", "8DF00A11ZZ", "zz" * 14, "A0" + "0" * 26])
def test_agree_on_invalid_input(bad: str) -> None:
    assert_all_agree([Case("invalid", bad)])


def test_agree_on_random_corruptions() -> None:
    """Flip one random bit in 1,000 valid position frames; both must refuse every one."""
    rng = random.Random(7)
    cases = []
    for i in range(1000):
        frame = rng.choice(POSITIONS)["hex"]
        cases.append(Case(f"flip-{i:04d}", ENCODER.flip_bits(frame, [rng.randint(1, 112)])))
    assert assert_all_agree(cases) == 1000


# --- Pairs: the verifier matches pyModeS pair matching, ambiguity included --


def pymodes_pair_position(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float] | None:
    """Position pyModeS's own pair matching gives for the newer frame of a pair."""
    first, second = sorted([a, b], key=lambda f: f["timestamp"])
    results = pms.decode([first["hex"], second["hex"]],
                         timestamps=[first["timestamp"], second["timestamp"]])
    newest = results[-1]
    if newest.get("latitude") is None:
        return None
    return newest["latitude"], newest["longitude"]


def track_pairs(max_gap: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = {}
    for f in POSITIONS:
        tracks.setdefault(f["track_id"], []).append(f)
    pairs = []
    for frames in tracks.values():
        frames.sort(key=lambda f: f["timestamp"])
        pairs += [(a, b) for a, b in zip(frames, frames[1:])
                  if b["timestamp"] - a["timestamp"] <= max_gap]
    return pairs


def test_pair_positions_agree_with_pymodes_pair_matching() -> None:
    problems = []
    pairs = track_pairs(max_gap=10)
    for a, b in pairs:
        ours = cpr.decode_hex_pair(a["hex"], a["timestamp"], b["hex"], b["timestamp"])
        ref = pymodes_pair_position(a, b)
        label = f"{a['frame_id']}+{b['frame_id']}"
        if ref is None or ours.status != "ok":
            problems.append(f"{label}: verifier={ours.status}/{ours.reason} pyModeS={ref}")
        elif (abs(ours.latitude - ref[0]) > POSITION_TOLERANCE_DEG  # type: ignore[operator]
              or abs(ours.longitude - ref[1]) > POSITION_TOLERANCE_DEG):  # type: ignore[operator]
            problems.append(f"{label}: verifier=({ours.latitude}, {ours.longitude}) pyModeS={ref}")
    assert not problems, f"{len(problems)} of {len(pairs)} pairs disagree:\n  " + "\n  ".join(problems)
    assert len(pairs) == 30  # 10 tracks x 3 consecutive pairs


def test_both_refuse_the_stale_pair() -> None:
    trk04 = sorted((f for f in POSITIONS if f["track_id"] == "trk-04"), key=lambda f: f["timestamp"])
    even, stale_odd = trk04[2], trk04[-1]
    ours = cpr.decode_hex_pair(even["hex"], even["timestamp"], stale_odd["hex"], stale_odd["timestamp"])
    assert ours.status == "ambiguous" and ours.reason == "pair_too_old"
    assert pymodes_pair_position(even, stale_odd) is None


def test_both_refuse_two_even_frames() -> None:
    evens = [f for f in POSITIONS if f["track_id"] == "trk-01" and f["expected"]["cpr_format"] == 0]
    ours = cpr.decode_hex_pair(evens[0]["hex"], evens[0]["timestamp"],
                               evens[1]["hex"], evens[1]["timestamp"])
    assert ours.status == "ambiguous" and ours.reason == "same_format"
    assert pymodes_pair_position(evens[0], evens[1]) is None


# ---------------------------------------------------------------------------
# Acceptance criterion 2 (negative case): a disagreement fails and names the frame
# ---------------------------------------------------------------------------


def broken_verifier(frame: str, timestamp: float) -> cpr.CPRFrame:
    """The real verifier with its latitude off by one, to simulate a bug."""
    parsed = cpr.parse_frame(frame, timestamp)
    return cpr.CPRFrame(parsed.cpr_format, (parsed.cpr_lat + 1) % cpr.NB, parsed.cpr_lon, timestamp)


def test_disagreement_fails_and_lists_every_frame() -> None:
    cases = [Case(f["frame_id"], f["hex"]) for f in POSITIONS[:3]]
    with pytest.raises(AssertionError) as failure:
        assert_all_agree(cases, parse=broken_verifier)
    message = str(failure.value)
    assert "disagree on 3 of 3 frames (0.0% agreement)" in message
    for f in POSITIONS[:3]:
        assert f["frame_id"] in message and f["hex"] in message
    assert "cpr_lat oracle=" in message


def test_one_bad_frame_among_many_is_named() -> None:
    target = POSITIONS[5]

    def verifier_wrong_on_one(frame: str, timestamp: float) -> cpr.CPRFrame:
        return broken_verifier(frame, timestamp) if frame == target["hex"] else cpr.parse_frame(frame, timestamp)

    with pytest.raises(AssertionError) as failure:
        assert_all_agree((Case(f["frame_id"], f["hex"]) for f in VALID), parse=verifier_wrong_on_one)
    message = str(failure.value)
    assert f"disagree on 1 of {len(VALID)} frames" in message
    assert target["frame_id"] in message
    others = [f["frame_id"] for f in POSITIONS if f is not target]
    assert not any(f"{fid} (" in message for fid in others)


def test_verifier_accepting_a_corrupt_frame_is_a_disagreement() -> None:
    def verifier_without_crc(frame: str, timestamp: float) -> cpr.CPRFrame:
        n = int(frame, 16)
        me = (n >> 24) & ((1 << 56) - 1)
        return cpr.CPRFrame((me >> 34) & 1, (me >> 17) & (cpr.NB - 1), me & (cpr.NB - 1), timestamp)

    corrupt_position = next(f for f in CORRUPT if f["message_type"] == "airborne_position")
    with pytest.raises(AssertionError, match=corrupt_position["frame_id"]):
        assert_all_agree([Case(corrupt_position["frame_id"], corrupt_position["hex"])],
                         parse=verifier_without_crc)


def test_oracle_missing_a_position_is_a_disagreement() -> None:
    def oracle_that_rejects_everything(frame: object) -> oracle.OracleResult:
        return oracle.OracleResult("rejected", {"answer": "reject", "reason": "crc_check_failed"})

    with pytest.raises(AssertionError, match=POSITIONS[0]["frame_id"]):
        assert_all_agree([Case(POSITIONS[0]["frame_id"], POSITIONS[0]["hex"])],
                         decode=oracle_that_rejects_everything)
