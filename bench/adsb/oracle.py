"""pyModeS ground-truth oracle for ADS-B tasks (WBS 2.3).

Wraps the pinned reference decoder (pyModeS) behind one call that never raises
and always answers in the task answer schema below. Tasks use it to build
Inspect AI samples: the answer becomes ``Sample.target`` as canonical JSON, and
the structured answer plus the raw decoder output go in ``Sample.metadata``.

Every input gets one of three statuses:

- ``decoded``: a well-formed DF17/DF18 frame whose CRC checks
- ``rejected``: a well-formed DF17/DF18 frame whose CRC fails; the answer is
  ``reject``, matching the corrupt frames in data/adsb
- ``invalid_input``: anything the oracle cannot treat as an extended squitter
  (not a string, empty, non-hex, truncated or overlong, another downlink
  format, or a decoder error), with a reason instead of a crash

Answer schema (version ANSWER_SCHEMA_VERSION). Every decoded answer has:

    df, icao, type_code, message_type

plus the fields for its message type:

    identification     callsign, category, wake_vortex
    airborne_position  altitude_ft, cpr_format ("even"/"odd"), cpr_lat, cpr_lon
    airborne_velocity  groundspeed_kt, track_deg, vertical_rate_fpm,
                       vertical_rate_source

Other ADS-B message types carry the four common fields only. Field names are
ours, not pyModeS's, so a pyModeS upgrade that renames a key cannot silently
change task answers; it fails the tests instead.

Command-line demo:

    python -m bench.adsb.oracle 8DF00A112369A6B1C31820ABE595
"""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from typing import Any, Literal

import pyModeS as pms
from inspect_ai.dataset import Sample

ORACLE_VERSION = "1.0.0"
ANSWER_SCHEMA_VERSION = "1.0.0"
DECODER = f"pyModeS=={version('pyModeS')}"

FRAME_HEX_LENGTH = 28  # 112-bit extended squitter
HEX_DIGITS = frozenset(string.hexdigits)

Status = Literal["decoded", "rejected", "invalid_input"]

MESSAGE_TYPES: dict[range | frozenset[int], str] = {
    range(1, 5): "identification",
    range(5, 9): "surface_position",
    range(9, 19): "airborne_position",
    frozenset({19}): "airborne_velocity",
    range(20, 23): "airborne_position",
    frozenset({28}): "aircraft_status",
    frozenset({29}): "target_state",
    frozenset({31}): "operational_status",
}

COMMON_FIELDS = ("df", "icao", "type_code", "message_type")

# Fields each message type adds to the answer, as (answer field, pyModeS key).
TYPE_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "identification": (("callsign", "callsign"), ("category", "category"),
                       ("wake_vortex", "wake_vortex")),
    "airborne_position": (("altitude_ft", "altitude"), ("cpr_format", "cpr_format"),
                          ("cpr_lat", "cpr_lat"), ("cpr_lon", "cpr_lon")),
    "airborne_velocity": (("groundspeed_kt", "groundspeed"), ("track_deg", "track"),
                          ("vertical_rate_fpm", "vertical_rate"),
                          ("vertical_rate_source", "vr_source")),
}


@dataclass(frozen=True)
class OracleResult:
    status: Status
    answer: dict[str, Any]
    reason: str | None = None
    raw: dict[str, Any] | None = None  # untouched pyModeS output, for rescoring
    provenance: dict[str, str] = field(default_factory=lambda: {
        "oracle_version": ORACLE_VERSION,
        "answer_schema_version": ANSWER_SCHEMA_VERSION,
        "decoder": DECODER,
    })

    def to_target(self) -> str:
        """The answer as canonical JSON, the form Inspect's Sample.target takes."""
        return json.dumps(self.answer, sort_keys=True, separators=(",", ":"))


def _invalid(reason: str) -> OracleResult:
    return OracleResult("invalid_input", {"answer": "invalid_input", "reason": reason}, reason)


def message_type(type_code: int) -> str:
    for codes, name in MESSAGE_TYPES.items():
        if type_code in codes:
            return name
    return "other"


def _answer_from(raw: dict[str, Any]) -> dict[str, Any]:
    tc = raw["typecode"]
    mtype = message_type(tc)
    answer: dict[str, Any] = {"df": raw["df"], "icao": raw["icao"], "type_code": tc,
                              "message_type": mtype}
    for name, key in TYPE_FIELDS.get(mtype, ()):
        value = raw.get(key)
        if name == "cpr_format" and value is not None:
            value = "odd" if value else "even"
        elif name == "track_deg" and value is not None:
            value = round(value, 2)
        answer[name] = value
    return answer


def decode(frame: object) -> OracleResult:
    """Decode one frame into the answer schema. Never raises."""
    if not isinstance(frame, str):
        return _invalid(f"expected a hex string, got {type(frame).__name__}")
    text = frame.strip().upper()
    if not text:
        return _invalid("empty input")
    if not set(text) <= HEX_DIGITS:
        return _invalid("input contains non-hex characters")
    if len(text) != FRAME_HEX_LENGTH:
        return _invalid(f"expected {FRAME_HEX_LENGTH} hex characters (112 bits), got {len(text)}")
    df = int(text[:2], 16) >> 3
    if df not in (17, 18):
        return _invalid(f"downlink format {df} is not an ADS-B extended squitter (DF17/DF18)")
    try:
        raw = dict(pms.decode(text))
    except Exception as exc:  # the oracle must answer, not crash
        return _invalid(f"decoder error: {type(exc).__name__}: {exc}")
    if not raw.get("crc_valid"):
        return OracleResult("rejected", {"answer": "reject", "reason": "crc_check_failed"},
                            "crc_check_failed", raw)
    return OracleResult("decoded", _answer_from(raw), None, raw)


def validate_answer(answer: dict[str, Any]) -> list[str]:
    """Check an answer against the schema. Returns a list of problems, empty when valid."""
    if answer.get("answer") in ("reject", "invalid_input"):
        extra = set(answer) - {"answer", "reason"}
        return [f"unexpected fields {sorted(extra)}"] if extra else []
    problems = [f"missing {name}" for name in COMMON_FIELDS if name not in answer]
    if problems:
        return problems
    expected = set(COMMON_FIELDS) | {n for n, _ in TYPE_FIELDS.get(answer["message_type"], ())}
    if set(answer) != expected:
        problems.append(f"fields {sorted(answer)} != schema {sorted(expected)}")
    if answer["df"] not in (17, 18):
        problems.append(f"df {answer['df']} is not 17 or 18")
    if not (isinstance(answer["icao"], str) and len(answer["icao"]) == 6
            and set(answer["icao"]) <= HEX_DIGITS):
        problems.append(f"icao {answer['icao']!r} is not 6 hex characters")
    if not (isinstance(answer["type_code"], int) and 0 <= answer["type_code"] <= 31):
        problems.append(f"type_code {answer['type_code']!r} out of range")
    if answer["message_type"] != message_type(answer["type_code"]):
        problems.append("message_type does not match type_code")
    return problems


def to_sample(frame: str, sample_id: str | None = None,
              metadata: dict[str, Any] | None = None) -> Sample:
    """An Inspect AI Sample whose target is the oracle's answer for this frame.

    The input is the raw frame; prompt wording belongs to the task (WBS 2.5),
    which can replace it. Oracle status, structured answer, raw decoder output
    and versions go in metadata so results can be rescored later.
    """
    result = decode(frame)
    return Sample(
        id=sample_id,
        input=frame,
        target=result.to_target(),
        metadata={**(metadata or {}), "oracle": asdict(result)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode an ADS-B frame with the pyModeS oracle.")
    parser.add_argument("frame", help="112-bit frame, 28 hex characters")
    args = parser.parse_args()
    result = decode(args.frame)
    out = {"status": result.status, "answer": result.answer, **result.provenance}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
