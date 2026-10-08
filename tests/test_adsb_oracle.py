"""Checks for the pyModeS oracle wrapper (WBS 2.3, issue #12).

Acceptance criteria:
- When a frame is passed to the oracle, it returns fields in the task answer
  schema.
- When truncated or non-hex input is passed, the oracle returns a handled
  'invalid input' result instead of crashing.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.dataset import MemoryDataset, Sample

from bench.adsb import oracle
from bench.adsb.oracle import decode, to_sample, validate_answer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "adsb"
MANIFEST = json.loads((DATA_DIR / "MANIFEST.json").read_text())
FRAMES: list[dict[str, Any]] = [
    json.loads(line) for line in (DATA_DIR / MANIFEST["file"]).read_text().splitlines()]
VALID = [f for f in FRAMES if f["label"] == "valid"]
CORRUPT = [f for f in FRAMES if f["label"] == "corrupt"]
DEMO_FRAME = "8DF00A112369A6B1C31820ABE595"  # trk-01 identification, ZZZ101


def ids(frames: list[dict[str, Any]]) -> list[str]:
    return [f["frame_id"] for f in frames]


# --- Demo -------------------------------------------------------------------


def test_demo_frame_decodes_correctly() -> None:
    result = decode(DEMO_FRAME)
    assert result.status == "decoded"
    assert result.answer == {"df": 17, "icao": "F00A11", "type_code": 4,
                             "message_type": "identification", "callsign": "ZZZ101",
                             "category": 3, "wake_vortex": "Medium 2"}


def test_cli_prints_the_demo_result() -> None:
    out = subprocess.run([sys.executable, "-m", "bench.adsb.oracle", DEMO_FRAME],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    printed = json.loads(out.stdout)
    assert printed["status"] == "decoded" and printed["answer"]["callsign"] == "ZZZ101"
    assert printed["decoder"] == MANIFEST["reference_decoder"]


# --- Acceptance criterion 1: answers follow the schema -----------------------


@pytest.mark.parametrize("frame", VALID, ids=ids(VALID))
def test_valid_frame_answer_follows_schema(frame: dict[str, Any]) -> None:
    result = decode(frame["hex"])
    assert result.status == "decoded"
    assert validate_answer(result.answer) == []


@pytest.mark.parametrize("frame", VALID, ids=ids(VALID))
def test_valid_frame_answer_matches_ground_truth(frame: dict[str, Any]) -> None:
    """The answer carries the same values as the 2.1 ground truth, under schema names."""
    answer = decode(frame["hex"]).answer
    expected = frame["expected"]
    assert answer["df"] == frame["df"]
    assert answer["icao"] == frame["icao"]
    assert answer["type_code"] == frame["type_code"]
    assert answer["message_type"] == frame["message_type"]
    for name, key in oracle.TYPE_FIELDS[frame["message_type"]]:
        value = expected[key]
        if name == "cpr_format":
            value = "odd" if value else "even"
        elif name == "track_deg":
            value = round(value, 2)
        assert answer[name] == value, name


@pytest.mark.parametrize("frame", CORRUPT, ids=ids(CORRUPT))
def test_corrupt_frame_is_rejected(frame: dict[str, Any]) -> None:
    result = decode(frame["hex"])
    assert result.status == "rejected"
    assert result.answer == {"answer": "reject", "reason": "crc_check_failed"}
    assert validate_answer(result.answer) == []
    assert result.raw is not None and result.raw["crc_valid"] is False  # kept for rescoring


def test_raw_decoder_output_is_kept() -> None:
    result = decode(DEMO_FRAME)
    assert result.raw is not None and result.raw["callsign"] == "ZZZ101"


def test_provenance_records_versions() -> None:
    provenance = decode(DEMO_FRAME).provenance
    assert provenance == {"oracle_version": oracle.ORACLE_VERSION,
                          "answer_schema_version": oracle.ANSWER_SCHEMA_VERSION,
                          "decoder": MANIFEST["reference_decoder"]}


def test_lowercase_and_whitespace_are_accepted() -> None:
    assert decode(f"  {DEMO_FRAME.lower()}\n").answer == decode(DEMO_FRAME).answer


@pytest.mark.parametrize("bad, problem", [
    ({"df": 17, "icao": "F00A11", "type_code": 4, "message_type": "identification"}, "fields"),
    ({"df": 11, "icao": "F00A11", "type_code": 19, "message_type": "airborne_velocity",
      "groundspeed_kt": 1, "track_deg": 1.0, "vertical_rate_fpm": 0,
      "vertical_rate_source": "BARO"}, "df"),
    ({"df": 17, "icao": "XYZ", "type_code": 99, "message_type": "other"}, "icao"),
    ({"icao": "F00A11"}, "missing"),
    ({"answer": "reject", "reason": "crc_check_failed", "callsign": "ZZZ101"}, "unexpected"),
])
def test_validator_catches_bad_answers(bad: dict[str, Any], problem: str) -> None:
    assert any(problem in p for p in validate_answer(bad))


# --- Acceptance criterion 2 (negative case): invalid input is handled --------


@pytest.mark.parametrize("length", [0, 1, 2, 13, 14, 20, 26, 27, 29, 56])
def test_truncated_or_overlong_input_is_invalid(length: int) -> None:
    text = (DEMO_FRAME * 3)[:length]
    result = decode(text)
    assert result.status == "invalid_input"
    assert result.answer["answer"] == "invalid_input" and result.reason


@pytest.mark.parametrize("bad", [
    "8DF00A112369A6B1C31820ABE59G",  # one non-hex character
    "8DF00A11 2369A6B1C31820ABE595",  # space inside the frame
    "0x8DF00A112369A6B1C31820ABE5",  # prefix
    "hello world",
    "Ω" * 28,
])
def test_non_hex_input_is_invalid(bad: str) -> None:
    result = decode(bad)
    assert result.status == "invalid_input"
    assert validate_answer(result.answer) == []


@pytest.mark.parametrize("bad", [None, 0x8DF00A11, b"8DF00A112369A6B1C31820ABE595", ["8D"], 3.14])
def test_non_string_input_is_invalid(bad: object) -> None:
    result = decode(bad)
    assert result.status == "invalid_input" and "hex string" in (result.reason or "")


def test_other_downlink_formats_are_invalid() -> None:
    """A well-formed long frame that is not DF17/DF18 (here DF20) is not ADS-B."""
    df20 = "A0" + DEMO_FRAME[2:]
    result = decode(df20)
    assert result.status == "invalid_input" and "downlink format 20" in (result.reason or "")


def test_oracle_never_raises_on_random_input() -> None:
    rng = random.Random(20261008)
    alphabet = "0123456789ABCDEFabcdef \t\nxyzXYZ!#-"
    statuses = set()
    for _ in range(3000):
        n = rng.choice([rng.randint(0, 40), 28])
        text = "".join(rng.choice(alphabet) for _ in range(n))
        if rng.random() < 0.3:  # well-formed hex, mostly DF17 with a random payload
            text = "8D" + "".join(rng.choice("0123456789ABCDEF") for _ in range(26))
        statuses.add(decode(text).status)
    assert statuses == {"rejected", "invalid_input"}  # random payloads essentially never pass CRC


# --- Inspect AI: the answer is the Sample target ----------------------------


def test_target_is_canonical_json_of_the_answer() -> None:
    result = decode(DEMO_FRAME)
    target = result.to_target()
    assert json.loads(target) == result.answer
    assert target == json.dumps(result.answer, sort_keys=True, separators=(",", ":"))


def test_to_sample_builds_an_inspect_sample() -> None:
    sample = to_sample(DEMO_FRAME, sample_id="adsb-0001", metadata={"family": "A1"})
    assert isinstance(sample, Sample)
    assert sample.id == "adsb-0001" and sample.input == DEMO_FRAME
    assert json.loads(sample.target) == decode(DEMO_FRAME).answer
    assert sample.metadata and sample.metadata["family"] == "A1"
    assert sample.metadata["oracle"]["status"] == "decoded"
    assert sample.metadata["oracle"]["raw"]["callsign"] == "ZZZ101"


def test_whole_frame_set_loads_as_inspect_dataset() -> None:
    dataset = MemoryDataset([to_sample(f["hex"], f["frame_id"]) for f in FRAMES])
    assert len(dataset) == len(FRAMES)
    statuses = [s.metadata["oracle"]["status"] for s in dataset if s.metadata]
    assert statuses.count("decoded") == len(VALID)
    assert statuses.count("rejected") == len(CORRUPT)


def test_invalid_input_still_builds_a_sample() -> None:
    sample = to_sample("8DF00A11")
    assert json.loads(sample.target) == {"answer": "invalid_input",
                                         "reason": "expected 28 hex characters (112 bits), got 8"}
