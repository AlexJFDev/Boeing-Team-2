"""A2 task family: CPR position decode (WBS 2.6, issue #49).

Each A2 task gives the model one or two airborne position messages from the
synthetic frame set (data/adsb) and asks for the aircraft's latitude and
longitude, or for the stated no-decode outcome when the position cannot be
determined. The reference is the independent CPR verifier, bench/adsb/cpr.py.

Cases (``CASES``):

    global_pair      an even and an odd frame under 10 s apart; target is the
                     verifier's global decode, the position at the newer frame
    local_reference  one frame plus a receiver position within 180 NM; target is
                     the verifier's local decode
    stale_pair       an even and an odd frame more than 10 s apart; the verifier
                     answers pair_too_old, so the target is no-decode
    same_format      two even or two odd frames; the verifier answers same_format,
                     so the target is no-decode
    single_frame     one frame and no reference position; by premise 4 of the
                     preamble the position cannot be determined

Targets are canonical JSON: ``{"latitude":..,"longitude":..}`` in decimal degrees
rounded to POSITION_DECIMALS places, or NO_DECODE_TARGET. The unrounded verifier
position goes in metadata (``expected_position``) for the 2.8 scorer's tolerance.

Validation (``validate_samples``) runs before every eval. It re-derives each
target from the verifier and checks the frames, times and receiver position in
the prompt are the ones the target was computed from. A mismatch raises
A2ValidationError and the run does not start.

    python scripts/adsb/build_a2_tasks.py     # rebuild the task file
    python -m bench.adsb.a2                   # validate it and show samples
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample

from bench.adsb import cpr
from bench.provenance import corpus_hash_from_manifest

A2_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES_MANIFEST = REPO_ROOT / "data" / "adsb" / "MANIFEST.json"
TASK_FILE = REPO_ROOT / "data" / "tasks" / f"a2_cpr_position_v{A2_VERSION}.jsonl"

PREAMBLE = "adsb_v1.0.0"
SOURCE = "spec-synthesized"

POSITION_DECIMALS = 5  # about 1 m; CPR airborne resolution is about 5 m
NO_DECODE = {"answer": "cannot_be_determined"}
NO_DECODE_TARGET = json.dumps(NO_DECODE, separators=(",", ":"))
RECEIVER_RANGE_NM = 180  # DO-260B A.1.7.5: local decoding needs the reference within 180 NM

CASES = ("global_pair", "local_reference", "stale_pair", "same_format", "single_frame")

PAIR_QUESTION = """\
Two airborne position messages were received from the same aircraft:

{frames}

What was the aircraft's latitude and longitude at the time of the more recent message?"""

SINGLE_QUESTION = """\
One airborne position message was received from this aircraft:

{frames}

What was the aircraft's latitude and longitude when it sent this message?"""

ANSWER_FORMAT = """\
Answer with a single JSON object and nothing else: keys in alphabetical order, no \
spaces or line breaks.
- If the position can be determined: {{"latitude":<number>,"longitude":<number>}} in \
decimal degrees, north and east positive, longitude between -180 and 180, each rounded \
to {decimals} decimal places and written without trailing zeros.
- If it cannot be determined, answer exactly {no_decode}"""


@dataclass(frozen=True)
class Spec:
    """One task before it is built: which frames, and the receiver position if any."""

    case: str
    frame_ids: tuple[str, ...]
    receiver: tuple[float, float] | None = None
    note: str = ""


# Receiver positions are about 40-90 NM from the aircraft, rounded to 0.01 degree.
# Several sit across a boundary from the aircraft, where a decoder that skips the
# zone adjustment or the longitude wrap goes wrong.
SPECS: tuple[Spec, ...] = (
    # Global pairs: one per track. Odd-numbered tracks use frames 1-2 (odd frame
    # newer), even-numbered tracks frames 2-3 (even frame newer).
    Spec("global_pair", ("adsb-0002", "adsb-0003"), note="odd frame newer"),
    Spec("global_pair", ("adsb-0009", "adsb-0010"), note="even frame newer; southern hemisphere"),
    Spec("global_pair", ("adsb-0014", "adsb-0015"), note="odd frame newer"),
    Spec("global_pair", ("adsb-0021", "adsb-0022"), note="even frame newer"),
    Spec("global_pair", ("adsb-0027", "adsb-0028"), note="odd frame newer; near the equator"),
    Spec("global_pair", ("adsb-0034", "adsb-0035"),
         note="even frame newer; the aircraft crosses the 180th meridian between the frames"),
    Spec("global_pair", ("adsb-0039", "adsb-0040"), note="odd frame newer; high latitude, NL 19"),
    Spec("global_pair", ("adsb-0046", "adsb-0047"), note="even frame newer; southern and western hemispheres"),
    Spec("global_pair", ("adsb-0051", "adsb-0052"), note="odd frame newer; next to the equator and prime meridian"),
    Spec("global_pair", ("adsb-0058", "adsb-0059"), note="even frame newer"),
    # Local decodes: one frame per track with a receiver position, five even and
    # five odd. Only trk-06 reuses its global pair's newer frame (for the wrap).
    Spec("local_reference", ("adsb-0005",), (39.53, -72.45)),
    Spec("local_reference", ("adsb-0008",), (-34.55, 155.62)),
    Spec("local_reference", ("adsb-0017",), (22.42, -157.80)),
    Spec("local_reference", ("adsb-0024",), (52.10, 2.71),
         note="the late frame from the stale pair; decodable locally"),
    Spec("local_reference", ("adsb-0029",), (-0.20, -28.91),
         note="receiver south of the equator, aircraft north of it"),
    Spec("local_reference", ("adsb-0035",), (-12.10, 179.40),
         note="aircraft just west of -180, receiver east of +180"),
    Spec("local_reference", ("adsb-0042",), (70.52, -37.91), note="high latitude, NL 19"),
    Spec("local_reference", ("adsb-0045",), (-44.95, -64.67),
         note="receiver in a different longitude-zone band (NL 42) from the aircraft (NL 41)"),
    Spec("local_reference", ("adsb-0054",), (-0.40, -0.59),
         note="receiver across both the equator and the prime meridian"),
    Spec("local_reference", ("adsb-0057",), (59.40, 16.90),
         note="receiver in a different longitude-zone band (NL 30) from the aircraft (NL 29)"),
    # Stale pairs: the catalogue's "valid pair 45 seconds apart".
    Spec("stale_pair", ("adsb-0022", "adsb-0024"), note="even and odd, 45.5 s apart"),
    Spec("stale_pair", ("adsb-0020", "adsb-0024"), note="even and odd, 46.5 s apart"),
    # Same format: the catalogue's "two even frames", and two odd.
    Spec("same_format", ("adsb-0020", "adsb-0022"), note="two even frames, 1 s apart"),
    Spec("same_format", ("adsb-0009", "adsb-0011"), note="two odd frames, 1 s apart"),
    Spec("same_format", ("adsb-0045", "adsb-0047"), note="two even frames, 1 s apart"),
    Spec("same_format", ("adsb-0058", "adsb-0060"), note="two odd frames, 1 s apart"),
    # One frame, no reference.
    Spec("single_frame", ("adsb-0002",), note="even frame alone"),
    Spec("single_frame", ("adsb-0040",), note="odd frame alone"),
)


class A2ValidationError(ValueError):
    """An A2 sample's target does not match the CPR verifier's reference decode."""


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@functools.cache
def frames_by_id() -> dict[str, dict[str, Any]]:
    """The frame set by frame ID, after checking its sha256 against the manifest."""
    corpus_hash_from_manifest(FRAMES_MANIFEST)
    manifest = json.loads(FRAMES_MANIFEST.read_text())
    path = FRAMES_MANIFEST.parent / manifest["file"]
    frames = (json.loads(line) for line in path.read_text().splitlines())
    return {f["frame_id"]: f for f in frames}


def format_time(t: float) -> str:
    return f"{t:.1f}"


def format_coord(value: float) -> str:
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# Reference answers
# ---------------------------------------------------------------------------


def _rounded(value: float) -> float:
    rounded = round(value, POSITION_DECIMALS)
    return 0.0 if rounded == 0 else rounded  # no "-0.0"


def canonical(answer: dict[str, Any]) -> str:
    return json.dumps(answer, sort_keys=True, separators=(",", ":"))


def reference_decode(frames: list[tuple[str, float]],
                     receiver: tuple[float, float] | None) -> cpr.CPRResult | None:
    """The verifier's result for a task's inputs; None when no decode applies.

    Two frames: global pair decode. One frame and a receiver: local decode. One
    frame and no receiver: None, which premise 4 of the preamble makes a
    no-decode. Raises A2ValidationError for any other combination or for a frame
    the verifier cannot use.
    """
    try:
        if len(frames) == 2 and receiver is None:
            (hex_a, t_a), (hex_b, t_b) = frames
            return cpr.decode_hex_pair(hex_a, t_a, hex_b, t_b)
        if len(frames) == 1 and receiver is not None:
            return cpr.decode_hex_with_reference(frames[0][0], *receiver)
        if len(frames) == 1 and receiver is None:
            cpr.parse_frame(*frames[0])  # still has to be a usable position frame
            return None
    except ValueError as exc:
        raise A2ValidationError(f"frame not usable for A2: {exc}") from None
    raise A2ValidationError(
        f"{len(frames)} frame(s) with{'' if receiver else 'out'} a receiver is not an A2 task shape")


def reference_answer(frames: list[tuple[str, float]],
                     receiver: tuple[float, float] | None) -> dict[str, Any]:
    """The A2 answer for a task's inputs, from the CPR verifier."""
    result = reference_decode(frames, receiver)
    if result is None or result.status != "ok":
        return dict(NO_DECODE)
    assert result.latitude is not None and result.longitude is not None
    return {"latitude": _rounded(result.latitude), "longitude": _rounded(result.longitude)}


def _case_of(frames: list[tuple[str, float]], receiver: tuple[float, float] | None,
             result: cpr.CPRResult | None) -> str:
    if receiver is not None:
        return "local_reference"
    if len(frames) == 1:
        return "single_frame"
    assert result is not None
    if result.status == "ok":
        return "global_pair"
    return {"pair_too_old": "stale_pair", "same_format": "same_format"}.get(
        str(result.reason), str(result.reason))


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def task_text(frames: list[tuple[str, float]]) -> str:
    lines = "\n".join(f"t = {format_time(t)} s: {h}" for h, t in frames)
    question = (PAIR_QUESTION if len(frames) == 2 else SINGLE_QUESTION).format(frames=lines)
    answer = ANSWER_FORMAT.format(decimals=POSITION_DECIMALS, no_decode=NO_DECODE_TARGET)
    return f"{question}\n\n{answer}"


def receiver_premises(receiver: tuple[float, float]) -> list[str]:
    lat, lon = receiver
    return [
        f"The receiving station is at latitude {format_coord(lat)}, longitude "
        f"{format_coord(lon)} (decimal degrees, north and east positive).",
        f"The aircraft is within {RECEIVER_RANGE_NM} NM of the receiving station.",
    ]


SINGLE_FRAME_PREMISE = "This is the only position message available for this aircraft."


def build_record(index: int, spec: Spec) -> dict[str, Any]:
    by_id = frames_by_id()
    records = sorted((by_id[i] for i in spec.frame_ids), key=lambda f: f["timestamp"])
    frames = [(f["hex"], f["timestamp"]) for f in records]
    result = reference_decode(frames, spec.receiver)
    case = _case_of(frames, spec.receiver, result)
    if case != spec.case:
        raise A2ValidationError(f"spec {index} is declared {spec.case} but decodes as {case}")
    answer = reference_answer(frames, spec.receiver)
    premises = (receiver_premises(spec.receiver) if spec.receiver
                else [SINGLE_FRAME_PREMISE] if case == "single_frame" else None)
    manifest = json.loads(FRAMES_MANIFEST.read_text())
    position = (None if result is None or result.status != "ok"
                else {"latitude": result.latitude, "longitude": result.longitude})
    record: dict[str, Any] = {
        "id": f"a2-{index:03d}-{case.replace('_', '-')}",
        "family": "A",
        "preamble": PREAMBLE,
        "source": SOURCE,
        "input": task_text(frames),
        "target": canonical(answer),
        "metadata": {
            "subfamily": "A2",
            "a2_version": A2_VERSION,
            "case": case,
            "frames": [{"frame_id": f["frame_id"], "hex": f["hex"], "timestamp": f["timestamp"]}
                       for f in records],
            "receiver": (None if spec.receiver is None
                         else {"latitude": spec.receiver[0], "longitude": spec.receiver[1]}),
            "track_id": records[0]["track_id"],
            "expected_position": position,
            "verifier_status": None if result is None else result.status,
            "verifier_reason": None if result is None else result.reason,
            "pair_age_s": None if result is None else result.pair_age_s,
            "note": spec.note,
            "frames_version": manifest["version"],
            "frames_sha256": manifest["sha256"],
            "verifier_version": cpr.VERIFIER_VERSION,
        },
    }
    if premises:
        record["premises"] = premises
    return record


def build_records() -> list[dict[str, Any]]:
    return [build_record(i, spec) for i, spec in enumerate(SPECS, start=1)]


def write_task_file(path: Path = TASK_FILE) -> Path:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in build_records()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _frames_from(meta: dict[str, Any]) -> list[tuple[str, float]] | str:
    raw = meta.get("frames")
    if not isinstance(raw, list) or not raw:
        return "metadata has no frames"
    by_id = frames_by_id()
    frames = []
    for item in raw:
        if not isinstance(item, dict):
            return "metadata frames are malformed"
        known = by_id.get(str(item.get("frame_id")))
        if known is None or known["hex"] != item.get("hex") or known["timestamp"] != item.get("timestamp"):
            return f"frame {item.get('frame_id')!r} does not match the frame set"
        frames.append((known["hex"], known["timestamp"]))
    if frames != sorted(frames, key=lambda f: f[1]):
        return "frames are not in reception order"
    return frames


def sample_problems(sample: Sample) -> list[str]:
    """Why a sample fails validation; empty when it passes."""
    meta = sample.metadata or {}
    frames = _frames_from(meta)
    if isinstance(frames, str):
        return [frames]
    raw_receiver = meta.get("receiver")
    receiver = (None if raw_receiver is None
                else (float(raw_receiver["latitude"]), float(raw_receiver["longitude"])))
    prompt = str(sample.input)
    problems = [f"prompt does not contain t = {format_time(t)} s: {h}"
                for h, t in frames if f"t = {format_time(t)} s: {h}" not in prompt]
    if receiver is not None:
        stated = f"latitude {format_coord(receiver[0])}, longitude {format_coord(receiver[1])}"
        if stated not in prompt:
            problems.append(f"prompt does not state the receiver position ({stated})")
    elif "receiving station" in prompt:
        problems.append("prompt states a receiver but metadata has none")
    if not isinstance(sample.target, str):
        return [*problems, "target must be a single JSON string"]
    try:
        expected = reference_answer(frames, receiver)
    except A2ValidationError as exc:
        return [*problems, str(exc)]
    if sample.target != canonical(expected):
        problems.append(
            f"target {sample.target} does not match the CPR verifier reference "
            f"{canonical(expected)} (bench.adsb.cpr {cpr.VERIFIER_VERSION})")
    return problems


def validate_samples(samples: Iterable[Sample]) -> None:
    """Raise A2ValidationError listing every sample that does not match the verifier."""
    failures = [f"{s.id}: {'; '.join(p)}" for s in samples if (p := sample_problems(s))]
    if failures:
        raise A2ValidationError(
            f"A2 task set failed validation ({len(failures)} sample(s)):\n" + "\n".join(failures))


def distance_nm(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in nautical miles (haversine, mean Earth radius)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 3440.065 * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    from bench.loader import load_tasks

    parser = argparse.ArgumentParser(description="Validate the A2 task file and show samples.")
    parser.add_argument("tasks", nargs="?", default=str(TASK_FILE), help="A2 task file")
    parser.add_argument("--show", action="append", default=None,
                        help="sample ID to print; repeat for more (default: one per case)")
    parser.add_argument("--prompt", action="store_true", help="also print each full prompt")
    args = parser.parse_args()
    dataset = load_tasks(args.tasks)
    try:
        validate_samples(dataset)
    except A2ValidationError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(f"{len(dataset)} samples match the CPR verifier (bench.adsb.cpr {cpr.VERIFIER_VERSION}).")
    if args.show:
        chosen = [s for s in dataset if s.id in args.show]
        missing = set(args.show) - {str(s.id) for s in chosen}
        if missing:
            raise SystemExit(f"error: no sample {sorted(missing)}")
    else:
        first_of_case: dict[str, Sample] = {}
        for s in dataset:
            first_of_case.setdefault((s.metadata or {}).get("case", ""), s)
        chosen = list(first_of_case.values())
    for sample in chosen:
        meta = sample.metadata or {}
        print(f"\n===== {sample.id} ({meta.get('case')})")
        for f in meta["frames"]:
            print(f"frame:     t = {format_time(f['timestamp'])} s  {f['hex']}  ({f['frame_id']})")
        if meta.get("receiver"):
            r = meta["receiver"]
            print(f"receiver:  {r['latitude']}, {r['longitude']}")
        status = meta.get("verifier_status") or "n/a (one frame, no reference)"
        reason = f" / {meta['verifier_reason']}" if meta.get("verifier_reason") else ""
        print(f"verifier:  {status}{reason}")
        if meta.get("expected_position"):
            p = meta["expected_position"]
            print(f"reference: {p['latitude']!r}, {p['longitude']!r}")
        print(f"target:    {sample.target}")
        if args.prompt:
            print("----- prompt:")
            print(sample.input)


if __name__ == "__main__":
    main()
