"""A1 task family: ADS-B identification and altitude decode (WBS 2.5, issue #48).

Each A1 task shows the model one extended squitter from the synthetic frame set
(data/adsb) and asks for the decoded identification or altitude fields, or a
rejection if the parity check fails. Ground truth is the pyModeS oracle
(bench/adsb/oracle.py), restricted to the A1 fields below.

Which frames are used:

- every identification frame, valid and corrupt, DF17 and DF18 (the DF18 frames
  are ADS-R rebroadcasts and a non-transponder device, catalogue A1's
  discriminator)
- the first airborne position frame of each synthetic track, for altitude (the
  other position frames repeat the same altitude; latitude and longitude are A2)
- every corrupt airborne position frame (the answer is a rejection)

Velocity frames are left out: A1 is identification and altitude.

A1 answer fields, a subset of the oracle's answer schema:

    every decoded frame  df, icao, type_code, message_type
    identification       + callsign, category
    airborne_position    + altitude_ft

``wake_vortex`` is left out because its wording is pyModeS's own label, which a
model cannot derive from the specification; ``category`` carries the same
information. The CPR fields are left out because they belong to A2.

Validation (``validate_samples``) re-decodes every sample's frame with pyModeS
and fails if the target differs, so a hand edit to the task file, a frame that
drifts from the frame set, or a pyModeS upgrade that changes a decode stops the
run instead of silently changing the answer key.

Build the task file (output is deterministic; the tests check the committed file
matches it):

    python scripts/adsb/build_a1_tasks.py

Check the committed task file against pyModeS and print one sample:

    python -m bench.adsb.a1
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample

from bench.adsb import oracle
from bench.provenance import corpus_hash_from_manifest

A1_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES_MANIFEST = REPO_ROOT / "data" / "adsb" / "MANIFEST.json"
TASK_FILE = REPO_ROOT / "data" / "tasks" / f"a1_adsb_decode_v{A1_VERSION}.jsonl"

PREAMBLE = "adsb_v1.0.0"
SOURCE = "spec-synthesized"

A1_FIELDS: dict[str, tuple[str, ...]] = {
    "identification": ("callsign", "category"),
    "airborne_position": ("altitude_ft",),
}

TASK_TEXT = """\
Decode the ADS-B message below.

Message: {frame}

Answer with a single JSON object and nothing else: keys in alphabetical order, no \
spaces or line breaks, strings in double quotes, numbers as plain integers.

If the message passes the parity check, report these keys:
- "df": downlink format (integer)
- "icao": 24-bit ICAO address as 6 uppercase hexadecimal characters
- "type_code": ADS-B type code (integer)
- "message_type": one of "identification", "surface_position", "airborne_position", \
"airborne_velocity", "aircraft_status", "target_state", "operational_status", "other"

and, depending on the message type, also:
- identification: "callsign" (uppercase, without padding characters) and "category" \
(the 3-bit emitter category field, integer 0-7)
- airborne_position: "altitude_ft" (barometric altitude in feet, integer). Do not \
report latitude or longitude.
- any other type: no further keys

If the parity check fails, answer exactly {{"answer":"reject","reason":"crc_check_failed"}}"""


class A1ValidationError(ValueError):
    """An A1 sample's target does not match the pyModeS reference decode."""


def reference_answer(frame: str) -> dict[str, Any]:
    """The A1 answer for a frame, from the pyModeS oracle."""
    result = oracle.decode(frame)
    if result.status == "rejected":
        return dict(result.answer)
    if result.status != "decoded":
        raise A1ValidationError(f"frame {frame} is not usable for A1: {result.reason}")
    answer = result.answer
    keep = oracle.COMMON_FIELDS + A1_FIELDS.get(answer["message_type"], ())
    return {name: answer[name] for name in keep}


def canonical(answer: dict[str, Any]) -> str:
    """Canonical JSON, the form the target takes (matches OracleResult.to_target)."""
    return json.dumps(answer, sort_keys=True, separators=(",", ":"))


def load_frames() -> list[dict[str, Any]]:
    """The frame set, after checking it against the sha256 in its manifest."""
    corpus_hash_from_manifest(FRAMES_MANIFEST)
    manifest = json.loads(FRAMES_MANIFEST.read_text())
    path = FRAMES_MANIFEST.parent / manifest["file"]
    return [json.loads(line) for line in path.read_text().splitlines()]


@functools.cache
def _frame_hex_by_id() -> dict[str, str]:
    return {f["frame_id"]: f["hex"] for f in load_frames()}


def _case(frame: dict[str, Any]) -> str:
    if frame["label"] == "corrupt":
        return "crc_failure"
    if frame["message_type"] == "airborne_position":
        return "altitude"
    return "df18_identification" if frame["df"] == 18 else "df17_identification"


def select_frames(frames: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The frames A1 uses, in frame-set order (see the module docstring)."""
    chosen, tracks_with_altitude = [], set()
    for frame in frames:
        mtype = frame["message_type"]
        if mtype == "identification":
            chosen.append(frame)
        elif mtype == "airborne_position" and frame["label"] == "corrupt":
            chosen.append(frame)
        elif mtype == "airborne_position" and frame["track_id"] not in tracks_with_altitude:
            tracks_with_altitude.add(frame["track_id"])
            chosen.append(frame)
    return chosen


def build_records() -> list[dict[str, Any]]:
    """Task records in the loader's format (bench/loader.py), one per frame."""
    manifest = json.loads(FRAMES_MANIFEST.read_text())
    records = []
    for frame in select_frames(load_frames()):
        answer = reference_answer(frame["hex"])
        records.append({
            "id": f"a1-{frame['frame_id']}",
            "family": "A",
            "preamble": PREAMBLE,
            "source": SOURCE,
            "input": TASK_TEXT.format(frame=frame["hex"]),
            "target": canonical(answer),
            "metadata": {
                "subfamily": "A1",
                "a1_version": A1_VERSION,
                "case": _case(frame),
                "frame": frame["hex"],
                "frame_id": frame["frame_id"],
                "frames_version": manifest["version"],
                "frames_sha256": manifest["sha256"],
                "expected_status": "rejected" if frame["label"] == "corrupt" else "decoded",
                "oracle_version": oracle.ORACLE_VERSION,
                "answer_schema_version": oracle.ANSWER_SCHEMA_VERSION,
                "decoder": oracle.DECODER,
            },
        })
    return records


def write_task_file(path: Path = TASK_FILE) -> Path:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in build_records()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def sample_problems(sample: Sample) -> list[str]:
    """Why a sample fails validation; empty when it passes."""
    meta = sample.metadata or {}
    frame = meta.get("frame")
    if not isinstance(frame, str) or not frame:
        return ["metadata has no frame"]
    problems = []
    frame_id = meta.get("frame_id")
    if frame_id is not None and _frame_hex_by_id().get(frame_id) != frame:
        problems.append(f"frame {frame} is not frame {frame_id!r} in the frame set")
    if frame not in str(sample.input):
        problems.append(f"prompt does not contain the frame {frame}")
    if not isinstance(sample.target, str):
        return [*problems, "target must be a single JSON string"]
    try:
        expected = reference_answer(frame)
    except A1ValidationError as exc:
        return [*problems, str(exc)]
    if sample.target != canonical(expected):
        problems.append(
            f"target {sample.target} does not match the pyModeS reference decode "
            f"{canonical(expected)} ({oracle.DECODER})"
        )
    status = "rejected" if expected.get("answer") == "reject" else "decoded"
    if meta.get("expected_status") not in (None, status):
        problems.append(f"expected_status {meta['expected_status']!r} but pyModeS gives {status!r}")
    return problems


def validate_samples(samples: Iterable[Sample]) -> None:
    """Raise A1ValidationError listing every sample that does not match pyModeS."""
    failures = [f"{s.id}: {'; '.join(p)}" for s in samples if (p := sample_problems(s))]
    if failures:
        raise A1ValidationError(
            f"A1 task set failed validation ({len(failures)} sample(s)):\n" + "\n".join(failures)
        )


def main() -> None:
    from bench.loader import load_tasks

    parser = argparse.ArgumentParser(description="Validate the A1 task file and show a sample.")
    parser.add_argument("tasks", nargs="?", default=str(TASK_FILE), help="A1 task file")
    parser.add_argument("--show", default=None, help="sample ID to print (default: the first)")
    args = parser.parse_args()
    dataset = load_tasks(args.tasks)
    try:
        validate_samples(dataset)
    except A1ValidationError as exc:
        raise SystemExit(f"error: {exc}") from None
    sample = next((s for s in dataset if s.id == args.show), None) if args.show else dataset[0]
    if sample is None:
        raise SystemExit(f"error: no sample {args.show!r}")
    print(f"{len(dataset)} samples match the pyModeS reference decode ({oracle.DECODER}).\n")
    meta = sample.metadata or {}
    print(f"===== {sample.id} ({meta.get('case')})")
    print(f"----- frame:  {meta.get('frame')}")
    print(f"----- target: {sample.target}")
    print("----- prompt:")
    print(sample.input)


if __name__ == "__main__":
    main()
