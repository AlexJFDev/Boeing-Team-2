"""AFDX BAG and SkewMax anomaly scorer: precision and recall (WBS 4.3, issue #69).

Scores a model's findings against the bag_violation and skew_violation events
in the 4.2 truth files. A/B copies of a frame are one logical frame. Benign
look-alikes are never ground truth. Precision and recall are never combined.

Answer format, matching rule and demo: see the Scoring section of
bench/afdx/README.md.

Bump ANOMALY_SCORER_VERSION when scoring behavior changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState

from bench.provenance import STORE_KEY, sha256_file

ANOMALY_SCORER_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "afdx" / "manifest.json"

IN_SCOPE_KINDS = ("bag_violation", "skew_violation")
DEFAULT_TOLERANCE_US = 1000
GROUPS = ("anomaly", "benign", "clean", "mixed", "mixed_benign")

PARSE_OK = "ok"
PARSE_OK_EXTRACTED = "ok_extracted"
PARSE_UNPARSEABLE = "unparseable"

MATCHING_RULE = (
    "Logical frame = labeled (vl_id, sn, t_us) with A/B copies collapsed. A finding "
    "matches a logical frame of the same (kind, vl_id, sn) when |round(t_s*1e6) - t_us| "
    "<= tolerance_us (nearest wins); network is informational only. Findings hitting the "
    "same logical frame count once; unmatched findings are deduplicated on "
    "(kind, vl_id, sn, t_us) and each is one FP; each malformed in-scope finding "
    "(missing kind, non-integer vl_id/sn, missing/null/non-numeric t_s) is one FP and "
    "never matches. Event TP = event with any logical frame hit; frame TP = logical "
    "frame hit. Out-of-scope kinds are counted and ignored."
)

COUNT_KEYS = ("event_tp", "event_fp", "event_fn", "frame_tp", "frame_fp", "frame_fn")


class TruthError(ValueError):
    """A truth file's anomalies and violations do not join cleanly."""


# --- Ground truth -------------------------------------------------------------


def events_from_truth(truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The in-scope events of one truth file, with A/B copies collapsed.

    Returns ``[{"event_id", "kind", "vl_id", "frames": [{"sn", "t_us", "networks"}]}]``.
    """
    by_id = {v["frame_id"]: v for v in truth["violations"]}
    events: list[dict[str, Any]] = []
    for anomaly in truth["anomalies"]:
        kind = anomaly["kind"]
        if kind not in IN_SCOPE_KINDS:
            continue
        frames: dict[tuple[int, int], set[str]] = {}
        for frame_id in anomaly["frame_ids"]:
            v = by_id.get(frame_id)
            if v is None:
                raise TruthError(f"{kind} frame {frame_id!r} has no violations entry")
            if v["kind"] != kind or v["vl_id"] != anomaly["vl_id"]:
                raise TruthError(
                    f"violation {frame_id!r} is {v['kind']} on VL {v['vl_id']}, "
                    f"but its event is {kind} on VL {anomaly['vl_id']}"
                )
            frames.setdefault((int(v["sn"]), int(v["t_us"])), set()).add(v["network"])
        events.append({
            "event_id": f"E{len(events)}",
            "kind": kind,
            "vl_id": int(anomaly["vl_id"]),
            "frames": [
                {"sn": sn, "t_us": t_us, "networks": sorted(nets)}
                for (sn, t_us), nets in sorted(frames.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            ],
        })
    return events


def canonical_events(events: list[dict[str, Any]]) -> str:
    """The target string: canonical JSON of the event list."""
    return json.dumps({"events": events}, sort_keys=True, separators=(",", ":"))


def manifest_sha256(manifest_path: Path | str = DEFAULT_MANIFEST) -> str:
    return sha256_file(manifest_path)


def sample_fields(
    scenario: Mapping[str, Any],
    truth: Mapping[str, Any],
    manifest_hash: str,
    corpus_schema_version: str | None = None,
) -> dict[str, Any]:
    """Loader-record fields for one AFDX scenario (hook for the C-task authors).

    Returns ``{"family": "C", "target": <canonical events JSON>, "metadata": {...}}``,
    ready to merge into a bench.loader task record. The author adds ``id``,
    ``preamble``, ``source``, ``input`` (and ``premises`` such as the resync rule).
    The metadata keys do not collide with the ones the loader adds.
    """
    events = events_from_truth(truth)
    return {
        "family": "C",
        "target": canonical_events(events),
        "metadata": {
            "scenario_id": scenario["id"],
            "group": scenario["category"],
            "anomaly_kinds": sorted({a["kind"] for a in truth["anomalies"]}),
            "events": events,
            "corpus_manifest_sha256": manifest_hash,
            "corpus_schema_version": corpus_schema_version,
            "anomaly_scorer_version": ANOMALY_SCORER_VERSION,
        },
    }


@dataclass(frozen=True)
class Scenario:
    row: dict[str, Any]
    truth: dict[str, Any]
    fields: dict[str, Any]


def load_corpus(manifest_path: Path | str = DEFAULT_MANIFEST) -> tuple[list[Scenario], str]:
    """Every scenario listed in the manifest, with its truth and sample fields."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    digest = manifest_sha256(manifest_path)
    version = manifest.get("generator", {}).get("schema_version")
    scenarios = []
    for row in manifest["scenarios"]:
        truth = json.loads((manifest_path.parent / "truth" / f"{row['id']}.json").read_text())
        scenarios.append(Scenario(row, truth, sample_fields(row, truth, digest, version)))
    return scenarios, digest


# --- Parsing the model answer ------------------------------------------------

# Decimal keeps the timestamp exactly as written, so rounding to microseconds is exact.
_DECODER = json.JSONDecoder(parse_float=Decimal)


def parse_answer(raw: str | None) -> tuple[str, list[Any] | None]:
    """Return (parse_status, findings list or None)."""
    text = (raw or "").strip()
    if not text:
        return PARSE_UNPARSEABLE, None
    try:
        obj, end = _DECODER.raw_decode(text)
        if end == len(text) and isinstance(obj, dict) and "findings" in obj:
            findings = obj["findings"]
            return (PARSE_OK, findings) if isinstance(findings, list) else (PARSE_UNPARSEABLE, None)
    except json.JSONDecodeError:
        pass
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = _DECODER.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and "findings" in obj:
                findings = obj["findings"]
                if isinstance(findings, list):
                    return PARSE_OK_EXTRACTED, findings
                return PARSE_UNPARSEABLE, None
            i = end
            continue
        i += 1
    return PARSE_UNPARSEABLE, None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _to_us(t_s: Any) -> int | None:
    """Seconds (JSON number) to whole microseconds, or None if not a finite number."""
    if isinstance(t_s, bool) or not isinstance(t_s, (int, Decimal)):
        return None  # floats only arise from NaN/Infinity constants
    if isinstance(t_s, Decimal) and not t_s.is_finite():
        return None
    return int((Decimal(t_s) * 1_000_000).to_integral_value())  # round half even


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(
        value, default=lambda o: float(o) if isinstance(o, Decimal) else str(o)
    ))


def classify_finding(item: Any) -> tuple[str, dict[str, Any] | str]:
    """("in_scope", finding) | ("out_of_scope", kind) | ("malformed", reason)."""
    if not isinstance(item, dict):
        return "malformed", "finding is not an object"
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        return "malformed", "missing kind"
    kind = kind.strip()
    if kind not in IN_SCOPE_KINDS:
        return "out_of_scope", kind
    if not _is_int(item.get("vl_id")):
        return "malformed", "missing or non-integer vl_id"
    if not _is_int(item.get("sn")):
        return "malformed", "missing or non-integer sn"
    if item.get("t_s") is None:
        return "malformed", "missing timestamp t_s"
    t_us = _to_us(item["t_s"])
    if t_us is None:
        return "malformed", "non-numeric t_s"
    network = item.get("network")
    return "in_scope", {
        "kind": kind,
        "vl_id": item["vl_id"],
        "sn": item["sn"],
        "t_us": t_us,
        "network": network if isinstance(network, str) else None,
    }


# --- Matching ----------------------------------------------------------------


def match_findings(
    events: list[dict[str, Any]],
    findings: list[Any] | None,
    tolerance_us: int = DEFAULT_TOLERANCE_US,
) -> dict[str, Any]:
    """Score one sample. ``findings`` None means the answer was unparseable."""
    index: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for e_i, event in enumerate(events):
        for f_i, frame in enumerate(event["frames"]):
            index.setdefault((event["kind"], event["vl_id"], frame["sn"]), []).append((e_i, f_i))

    hit: set[tuple[int, int]] = set()
    unmatched: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    malformed: dict[str, dict[str, Any]] = {}
    out_of_scope = 0
    network_mismatch = 0

    for item in findings or []:
        status, value = classify_finding(item)
        if status == "out_of_scope":
            out_of_scope += 1
            continue
        if status == "malformed":
            key = json.dumps(item, sort_keys=True, default=str)
            malformed.setdefault(key, {"finding": _jsonable(item), "reason": value})
            continue
        assert isinstance(value, dict)
        best: tuple[int, int, int] | None = None  # (distance, e_i, f_i)
        for e_i, f_i in index.get((value["kind"], value["vl_id"], value["sn"]), []):
            distance = abs(events[e_i]["frames"][f_i]["t_us"] - value["t_us"])
            if distance <= tolerance_us and (best is None or distance < best[0]):
                best = (distance, e_i, f_i)
        if best is None:
            unmatched.setdefault((value["kind"], value["vl_id"], value["sn"], value["t_us"]), value)
            continue
        _, e_i, f_i = best
        hit.add((e_i, f_i))
        if value["network"] is not None and value["network"] not in events[e_i]["frames"][f_i]["networks"]:
            network_mismatch += 1

    n_frames = sum(len(e["frames"]) for e in events)
    events_hit = {e_i for e_i, _ in hit}
    fp = len(unmatched) + len(malformed)
    counts = {
        "event_tp": len(events_hit),
        "event_fp": fp,
        "event_fn": len(events) - len(events_hit),
        "frame_tp": len(hit),
        "frame_fp": fp,
        "frame_fn": n_frames - len(hit),
    }
    return {
        "counts": counts,
        "unmatched_findings": list(unmatched.values()),
        "malformed_findings": list(malformed.values()),
        "unmatched_events": [
            {k: e[k] for k in ("event_id", "kind", "vl_id")}
            for e_i, e in enumerate(events) if e_i not in events_hit
        ],
        "unmatched_frames": [
            {"event_id": e["event_id"], "kind": e["kind"], "vl_id": e["vl_id"], **fr}
            for e_i, e in enumerate(events)
            for f_i, fr in enumerate(e["frames"]) if (e_i, f_i) not in hit
        ],
        "out_of_scope_count": out_of_scope,
        "network_mismatch_count": network_mismatch,
    }


def score_answer(
    raw: str | None,
    events: list[dict[str, Any]],
    tolerance_us: int = DEFAULT_TOLERANCE_US,
) -> dict[str, Any]:
    """Parse and match one answer (plain Python; the Inspect scorer wraps this)."""
    status, findings = parse_answer(raw)
    result = match_findings(events, findings, tolerance_us)
    result["parse_status"] = status
    return result


# --- Aggregation (micro-average) ---------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def aggregate(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum counts across samples; precision/recall are None ("n/a") on a zero denominator."""
    totals = {k: 0.0 for k in COUNT_KEYS}
    unparseable = 0.0
    n = 0
    for value in values:
        n += 1
        for k in COUNT_KEYS:
            totals[k] += float(value[k])
        unparseable += float(value.get("unparseable", 0))
    out: dict[str, Any] = {"samples": n, "unparseable_count": unparseable, **totals}
    for level in ("event", "frame"):
        tp, fp, fn = totals[f"{level}_tp"], totals[f"{level}_fp"], totals[f"{level}_fn"]
        out[f"{level}_precision"] = _ratio(tp, tp + fp)
        out[f"{level}_recall"] = _ratio(tp, tp + fn)
    return out


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _metric_value(scores: list[SampleScore], key: str) -> float:
    value = aggregate(s.score.value for s in scores if isinstance(s.score.value, Mapping))[key]
    return math.nan if value is None else float(value)


@metric
def event_precision() -> Metric:
    """Micro-averaged event-level precision (NaN when no findings at all)."""
    def compute(scores: list[SampleScore]) -> float:
        return _metric_value(scores, "event_precision")
    return compute


@metric
def event_recall() -> Metric:
    """Micro-averaged event-level recall (NaN when there are no labeled events)."""
    def compute(scores: list[SampleScore]) -> float:
        return _metric_value(scores, "event_recall")
    return compute


@metric
def frame_precision() -> Metric:
    """Micro-averaged frame-level precision (secondary)."""
    def compute(scores: list[SampleScore]) -> float:
        return _metric_value(scores, "frame_precision")
    return compute


@metric
def frame_recall() -> Metric:
    """Micro-averaged frame-level recall (secondary)."""
    def compute(scores: list[SampleScore]) -> float:
        return _metric_value(scores, "frame_recall")
    return compute


@metric
def unparseable_count() -> Metric:
    """Number of samples whose answer could not be parsed."""
    def compute(scores: list[SampleScore]) -> float:
        return _metric_value(scores, "unparseable_count")
    return compute


# --- Inspect scorer ----------------------------------------------------------


def _store_corpus_hash(state: TaskState) -> str | None:
    store = getattr(state, "store", None)
    getter = getattr(store, "get", None)
    if getter is None:
        return None
    prov = getter(STORE_KEY, None)
    return prov.get("corpus_hash") if isinstance(prov, dict) else None


@scorer(metrics=[event_precision(), event_recall(), frame_precision(), frame_recall(),
                 unparseable_count()])
def afdx_anomaly_scorer(tolerance_us: int = DEFAULT_TOLERANCE_US) -> Scorer:
    """Precision/recall of BAG and SkewMax findings against the 4.2 labels.

    The target is the canonical events JSON from ``sample_fields``. Score.value
    holds the per-sample counts; the metrics micro-average them. Precision and
    recall are never combined into one number.
    """
    if not _is_int(tolerance_us) or tolerance_us < 0:
        raise ValueError("tolerance_us must be a non-negative integer")

    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion if state.output else ""
        events = json.loads(target.text)["events"]
        result = score_answer(raw, events, tolerance_us)
        counts = result["counts"]
        meta = state.metadata if isinstance(getattr(state, "metadata", None), dict) else {}
        unparseable = result["parse_status"] == PARSE_UNPARSEABLE
        return Score(
            value={**counts, "unparseable": int(unparseable)},
            answer=raw,
            explanation=(
                f"parse={result['parse_status']} events TP={counts['event_tp']} "
                f"FP={counts['event_fp']} FN={counts['event_fn']}; frames "
                f"TP={counts['frame_tp']} FP={counts['frame_fp']} FN={counts['frame_fn']}; "
                f"out_of_scope={result['out_of_scope_count']}; tolerance={tolerance_us} us"
            ),
            metadata={
                "scorer_version": ANOMALY_SCORER_VERSION,
                "score_type": "capability_only",
                "tolerance_us": tolerance_us,
                "tolerance_s": tolerance_us / 1_000_000,
                "matching_rule": MATCHING_RULE,
                **counts,
                "parse_status": result["parse_status"],
                "unmatched_findings": result["unmatched_findings"],
                "malformed_findings": result["malformed_findings"],
                "unmatched_events": result["unmatched_events"],
                "unmatched_frames": result["unmatched_frames"],
                "out_of_scope_count": result["out_of_scope_count"],
                "network_mismatch_count": result["network_mismatch_count"],
                "scenario_id": meta.get("scenario_id"),
                "group": meta.get("group"),
                "corpus_manifest_sha256": meta.get("corpus_manifest_sha256"),
                "corpus_hash": _store_corpus_hash(state),
            },
        )

    return score


# --- Demo ----------------------------------------------------------------------


def _finding(kind: str, vl_id: int, sn: int, network: str, t_us: int) -> dict[str, Any]:
    return {"kind": kind, "vl_id": vl_id, "sn": sn, "network": network, "t_s": t_us / 1_000_000}


def _perfect(sc: Scenario, pos: int) -> list[dict[str, Any]]:
    return [
        _finding(e["kind"], e["vl_id"], fr["sn"], fr["networks"][0], fr["t_us"])
        for e in sc.fields["metadata"]["events"] for fr in e["frames"]
    ]


def _partial(sc: Scenario, pos: int) -> list[dict[str, Any]]:
    # Scenarios at even corpus positions: first logical frame of each event only.
    # Odd positions: nothing reported, so their events are missed.
    if pos % 2:
        return []
    return [
        _finding(e["kind"], e["vl_id"], e["frames"][0]["sn"], e["frames"][0]["networks"][0],
                 e["frames"][0]["t_us"])
        for e in sc.fields["metadata"]["events"]
    ]


def _false_alarm(sc: Scenario, pos: int) -> list[dict[str, Any]]:
    # Perfect answers, plus every benign look-alike flagged as the violation it resembles.
    findings = _perfect(sc, pos)
    frames = {f["id"]: f for f in sc.truth["frames"]}
    look_alike = {"late_frame_gap": "bag_violation", "skew_offset_within_limit": "skew_violation"}
    for oddity in sc.truth.get("benign_oddities", []):
        if oddity["kind"] in look_alike:
            f = frames[oddity["frame_ids"][0]]
            findings.append(_finding(look_alike[oddity["kind"]], f["vl_id"], f["sn"],
                                     f["network"], round(f["t_s"] * 1_000_000)))
    return findings


DEMO_RUNS = {
    "perfect": _perfect,
    "partial (every other scenario, first frame of each event only)": _partial,
    "perfect + benign look-alikes flagged": _false_alarm,
}


async def _score_one(scorer_fn: Scorer, sc: Scenario, raw: str) -> Score:
    from inspect_ai.model import ModelName, ModelOutput

    state = TaskState(
        model=ModelName("demo/canned"), sample_id=sc.row["id"], epoch=1, input="", messages=[],
        output=ModelOutput.from_content(model="demo/canned", content=raw),
        metadata=sc.fields["metadata"],
    )
    result = await scorer_fn(state, Target(sc.fields["target"]))
    assert result is not None
    return result


def _line(label: str, agg: Mapping[str, Any]) -> str:
    return (
        f"  {label:<13} n={agg['samples']:<4} "
        f"event P={fmt(agg['event_precision'])} R={fmt(agg['event_recall'])} "
        f"(TP {agg['event_tp']:.0f} FP {agg['event_fp']:.0f} FN {agg['event_fn']:.0f})   "
        f"frame P={fmt(agg['frame_precision'])} R={fmt(agg['frame_recall'])} "
        f"(TP {agg['frame_tp']:.0f} FP {agg['frame_fp']:.0f} FN {agg['frame_fn']:.0f})"
    )


def demo(manifest_path: Path | str = DEFAULT_MANIFEST, tolerance_us: int = DEFAULT_TOLERANCE_US) -> None:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(
            f"error: {manifest_path} not found; run python -m bench.afdx.generate_afdx_traces first"
        )
    scenarios, digest = load_corpus(manifest_path)
    n_events = sum(len(s.fields["metadata"]["events"]) for s in scenarios)
    print(f"AFDX anomaly scorer {ANOMALY_SCORER_VERSION}  tolerance {tolerance_us} us")
    print(f"corpus {manifest_path}  sha256 {digest[:16]}...")
    print(f"{len(scenarios)} scenarios, {n_events} BAG/SkewMax events (headline: event level)\n")
    scorer_fn = afdx_anomaly_scorer(tolerance_us)

    async def run(build: Any) -> list[tuple[Scenario, Score]]:
        return [
            (sc, await _score_one(scorer_fn, sc, json.dumps({"findings": build(sc, pos)})))
            for pos, sc in enumerate(scenarios)
        ]

    for name, build in DEMO_RUNS.items():
        scored = asyncio.run(run(build))
        print(f"== {name}")
        print(_line("all", aggregate(s.value for _, s in scored)))  # type: ignore[misc]
        for group in GROUPS:
            values = [s.value for sc, s in scored if sc.row["category"] == group]
            print(_line(group, aggregate(values)))  # type: ignore[arg-type]
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--demo", action="store_true", help="score canned answers against the corpus")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--tolerance-us", type=int, default=DEFAULT_TOLERANCE_US)
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return
    demo(args.manifest, args.tolerance_us)


if __name__ == "__main__":
    main()
