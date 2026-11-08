"""Unit tests for the AFDX BAG/SkewMax anomaly scorer (WBS 4.3, issue #69).

All truth and prediction fixtures are hand-written; nothing here reads the
generated corpus in data/afdx/.

Acceptance criteria (backlog wording):
  AC1  When a model's triage output is scored, true/false positives and false
       negatives are counted against the labels.
  AC2  When the output lists an anomaly with no timestamp, it is counted as a
       false positive. (negative case)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelName, ModelOutput, get_model
from inspect_ai.scorer import Metric, MetricProtocol, SampleScore, Score, Target
from inspect_ai.solver import TaskState, generate

from bench.afdx import anomaly_scorer as sc
from bench.loader import load_tasks
from bench.provenance import sha256_file
from bench.scorers import rescore_raw_output

# --- Fixtures ------------------------------------------------------------------

BAG_T = 2_496_000            # BAG event, VL 203, SN 39; copies at different pcap positions
SKEW_T0 = 69_000             # skew event, VL 100 (BAG 2 ms), SN 68..73 on B, 2 ms apart


def _v(kind, fid, vl, sn, t_us):
    return {"kind": kind, "frame_id": fid, "network": fid[0], "vl_id": vl, "sn": sn,
            "t_s": t_us / 1e6, "t_us": t_us}


SKEW_IDS = ["B43", "B44", "B46", "B47", "B48", "B49"]

TRUTH = {
    "anomalies": [
        {"kind": "bag_violation", "vl_id": 203, "frame_ids": ["A120", "B115"]},
        {"kind": "injected_payload", "vl_id": 202, "frame_ids": ["A67"]},
        {"kind": "skew_violation", "vl_id": 100, "frame_ids": SKEW_IDS},
    ],
    "violations": [
        _v("injected_payload", "A67", 202, 136, 832_000),
        _v("bag_violation", "A120", 203, 39, BAG_T),
        _v("bag_violation", "B115", 203, 39, BAG_T),
        *[_v("skew_violation", fid, 100, 68 + i, SKEW_T0 + 2000 * i) for i, fid in enumerate(SKEW_IDS)],
    ],
    "benign_oddities": [{"kind": "late_frame_gap", "vl_id": 201, "frame_ids": ["A14", "B14"]}],
}

BAG_ONLY = {
    "anomalies": [{"kind": "bag_violation", "vl_id": 203, "frame_ids": ["A120", "B115"]}],
    "violations": [_v("bag_violation", "A120", 203, 39, BAG_T), _v("bag_violation", "B115", 203, 39, BAG_T)],
    "benign_oddities": [],
}

# Benign-only scenario: a late_frame_gap look-alike on VL 201, no anomalies.
BENIGN_ONLY = {
    "anomalies": [],
    "violations": [],
    "benign_oddities": [{"kind": "late_frame_gap", "vl_id": 201, "frame_ids": ["A14", "B14"]}],
}

CLEAN = {"anomalies": [], "violations": [], "benign_oddities": []}

EVENTS = sc.events_from_truth(TRUTH)
BAG_EVENTS = sc.events_from_truth(BAG_ONLY)


def bag(**over):
    f = {"kind": "bag_violation", "vl_id": 203, "sn": 39, "network": "A", "t_s": BAG_T / 1e6}
    f.update(over)
    return f


def skew(i, **over):
    f = {"kind": "skew_violation", "vl_id": 100, "sn": 68 + i, "network": "B",
         "t_s": (SKEW_T0 + 2000 * i) / 1e6}
    f.update(over)
    return f


def answer(*findings) -> str:
    return json.dumps({"findings": list(findings)})


def run(raw, events=EVENTS, tol=sc.DEFAULT_TOLERANCE_US):
    return sc.score_answer(raw, events, tol)


def counts(raw, events=EVENTS, tol=sc.DEFAULT_TOLERANCE_US):
    return run(raw, events, tol)["counts"]


def c(etp, efp, efn, ftp, ffp, ffn):
    return {"event_tp": etp, "event_fp": efp, "event_fn": efn,
            "frame_tp": ftp, "frame_fp": ffp, "frame_fn": ffn}


# --- Ground truth join -----------------------------------------------------------


def test_events_join_collapse_ab_copies_and_scope():
    assert [e["kind"] for e in EVENTS] == ["bag_violation", "skew_violation"]
    bag_event, skew_event = EVENTS
    # A120 and B115 sit at different pcap positions but are one logical frame.
    assert bag_event["frames"] == [{"sn": 39, "t_us": BAG_T, "networks": ["A", "B"]}]
    assert [f["sn"] for f in skew_event["frames"]] == list(range(68, 74))
    assert all(f["networks"] == ["B"] for f in skew_event["frames"])
    # benign_oddities and out-of-scope anomalies are not ground truth
    assert sc.events_from_truth(BENIGN_ONLY) == []


def test_events_join_rejects_missing_violation():
    broken = {**BAG_ONLY, "violations": BAG_ONLY["violations"][:1]}
    with pytest.raises(sc.TruthError):
        sc.events_from_truth(broken)


def test_canonical_target_round_trips():
    assert json.loads(sc.canonical_events(EVENTS)) == {"events": EVENTS}


# --- AC1: TP / FP / FN counted against the labels ----------------------------------


def test_ac1_triage_output_counts_tp_fp_fn_against_labels():
    """AC1: When a model's triage output is scored, true/false positives and false
    negatives are counted against the labels."""
    raw = answer(bag(), skew(0), bag(vl_id=999))  # BAG hit, 1 of 6 skew frames, 1 false alarm
    assert counts(raw) == c(2, 1, 0, 2, 1, 5)
    result = run(raw)
    assert result["unmatched_findings"] == [
        {"kind": "bag_violation", "vl_id": 999, "sn": 39, "t_us": BAG_T, "network": "A"}]
    assert len(result["unmatched_frames"]) == 5


# --- AC2 (negative case): no timestamp is a false positive -------------------------


@pytest.mark.parametrize("missing", [{"drop": True}, {"t_s": None}])
def test_ac2_anomaly_without_timestamp_is_false_positive(missing):
    """AC2: When the output lists an anomaly with no timestamp, it is counted as a
    false positive. Correct kind, vl_id and sn, but no t_s: FP 1, and the event is
    still a false negative."""
    f = bag()
    if missing.get("drop"):
        del f["t_s"]
    else:
        f["t_s"] = None
    result = run(answer(f), BAG_EVENTS)
    assert result["counts"] == c(0, 1, 1, 0, 1, 1)
    assert result["malformed_findings"][0]["reason"] == "missing timestamp t_s"
    assert result["unmatched_events"] == [{"event_id": "E0", "kind": "bag_violation", "vl_id": 203}]


# --- Basic shapes ------------------------------------------------------------------


def test_perfect_match():
    raw = answer(bag(), *[skew(i) for i in range(6)])
    assert counts(raw) == c(2, 0, 0, 7, 0, 0)
    agg = sc.aggregate([counts(raw)])
    assert agg["event_precision"] == agg["event_recall"] == 1.0


def test_all_missed():
    assert counts(answer()) == c(0, 0, 2, 0, 0, 7)
    agg = sc.aggregate([counts(answer())])
    assert agg["event_precision"] is None and agg["event_recall"] == 0.0


def test_all_false_positives():
    raw = answer(bag(sn=40), skew(0, vl_id=101))
    assert counts(raw) == c(0, 2, 2, 0, 2, 7)
    assert sc.aggregate([counts(raw)])["event_precision"] == 0.0


def test_partial_overlap_exact_values():
    raw = answer(skew(1), skew(4), bag(sn=50))
    assert counts(raw) == c(1, 1, 1, 2, 1, 5)
    agg = sc.aggregate([counts(raw)])
    assert agg["event_precision"] == 0.5 and agg["event_recall"] == 0.5
    assert agg["frame_precision"] == pytest.approx(2 / 3)
    assert agg["frame_recall"] == pytest.approx(2 / 7)


# --- Tolerance (integer microseconds, inclusive) -------------------------------------


@pytest.mark.parametrize("offset_us, hit", [
    (1000, True), (-1000, True),      # exactly at the bound
    (999, True), (-999, True),        # one microsecond inside
    (1001, False), (-1001, False),    # one microsecond outside
])
def test_tolerance_edge(offset_us, hit):
    raw = answer(bag(t_s=(BAG_T + offset_us) / 1e6))
    assert counts(raw, BAG_EVENTS) == (c(1, 0, 0, 1, 0, 0) if hit else c(0, 1, 1, 0, 1, 1))


def test_tolerance_rounds_to_whole_microsecond():
    assert counts(answer(bag(t_s=2.4970004)), BAG_EVENTS)["event_tp"] == 1   # 1000.4 us -> 1000
    assert counts(answer(bag(t_s=2.4970006)), BAG_EVENTS)["event_tp"] == 0   # 1000.6 us -> 1001


def test_tolerance_is_configurable():
    raw = answer(bag(t_s=(BAG_T + 5000) / 1e6))
    assert counts(raw, BAG_EVENTS, tol=5000)["event_tp"] == 1
    assert counts(raw, BAG_EVENTS, tol=4999)["event_tp"] == 0
    assert counts(answer(bag()), BAG_EVENTS, tol=0)["event_tp"] == 1


def test_skew_a_copy_time_at_bag_2ms_is_exactly_1ms_off():
    # Generator delays B by 0.5 x BAG; at BAG 2 ms the A copy is 1000 us before the label.
    a_copy = skew(0, network="A", t_s=(SKEW_T0 - 1000) / 1e6)
    r = run(answer(a_copy))
    assert r["counts"]["frame_tp"] == 1 and r["network_mismatch_count"] == 1
    assert counts(answer(a_copy), tol=999)["frame_tp"] == 0


# --- Duplicates and several findings per event --------------------------------------


def test_duplicates_count_once_regardless_of_key_order_and_spacing():
    dup = '{ "t_s" : 2.496 , "sn":39, "network":"A","vl_id" :203,"kind":"bag_violation"}'
    raw = '{"findings": [' + json.dumps(bag()) + ", " + dup + "]}"
    assert counts(raw, BAG_EVENTS) == c(1, 0, 0, 1, 0, 0)


def test_unmatched_duplicates_are_one_false_positive():
    assert counts(answer(bag(sn=40), bag(sn=40, network="B")), BAG_EVENTS)["event_fp"] == 1


def test_bag_a_and_b_copies_reported_separately_count_once():
    r = run(answer(bag(network="A"), bag(network="B")), BAG_EVENTS)
    assert r["counts"] == c(1, 0, 0, 1, 0, 0)
    assert r["network_mismatch_count"] == 0


def test_several_findings_on_one_event_count_once():
    raw = answer(*[skew(i) for i in range(6)], skew(2), skew(2, t_s=(SKEW_T0 + 4500) / 1e6))
    r = run(raw)
    assert r["counts"] == c(1, 0, 1, 6, 0, 1)


# --- Wrong fields ----------------------------------------------------------------------


@pytest.mark.parametrize("over", [{"vl_id": 204}, {"sn": 40}, {"kind": "skew_violation"}],
                         ids=["wrong_vl", "wrong_sn", "wrong_kind_bag_as_skew"])
def test_wrong_vl_sn_or_kind_is_fp_and_fn(over):
    assert counts(answer(bag(**over)), BAG_EVENTS) == c(0, 1, 1, 0, 1, 1)


def test_wrong_network_still_matches_and_is_recorded():
    r = run(answer(skew(0, network="A")))
    assert r["counts"]["frame_tp"] == 1
    assert r["network_mismatch_count"] == 1


@pytest.mark.parametrize("over, reason", [
    ({"t_s": "2.496"}, "non-numeric t_s"),
    ({"t_s": True}, "non-numeric t_s"),
    ({"vl_id": True}, "missing or non-integer vl_id"),
    ({"vl_id": "203"}, "missing or non-integer vl_id"),
    ({"sn": 39.0}, "missing or non-integer sn"),
    ({"sn": False}, "missing or non-integer sn"),
    ({"sn": None}, "missing or non-integer sn"),
    ({"kind": None}, "missing kind"),
])
def test_malformed_findings_are_fp_and_never_match(over, reason):
    r = run(answer(bag(**over)), BAG_EVENTS)
    assert r["counts"] == c(0, 1, 1, 0, 1, 1)
    assert r["malformed_findings"][0]["reason"] == reason


def test_malformed_non_object_and_nan_timestamp():
    r = run('{"findings": ["bag on 203", {"kind": "bag_violation", "vl_id": 203, "sn": 39, "t_s": NaN}]}',
            BAG_EVENTS)
    assert r["counts"] == c(0, 2, 1, 0, 2, 1)
    assert [m["reason"] for m in r["malformed_findings"]] == ["finding is not an object", "non-numeric t_s"]


# --- Parsing -----------------------------------------------------------------------------


def test_parse_status_ok_and_extracted():
    assert run(answer(bag()), BAG_EVENTS)["parse_status"] == sc.PARSE_OK
    prose = "Triage summary:\n```json\n" + answer(bag()) + "\n```\nThat is all."
    r = run(prose, BAG_EVENTS)
    assert r["parse_status"] == sc.PARSE_OK_EXTRACTED and r["counts"]["event_tp"] == 1
    # first object with a findings key wins; earlier objects without one are skipped
    two = '{"note": 1} then ' + answer(bag()) + " and " + answer()
    assert run(two, BAG_EVENTS)["counts"]["event_tp"] == 1


@pytest.mark.parametrize("raw", [
    "", "   ", None, "VL 203 has a BAG violation at 2.496 s",
    '{"anomalies": []}', '{"findings": {"kind": "bag_violation"}}', '{"findings": [',
    "[" + json.dumps(bag()) + "]",
])
def test_unparseable_answer_is_own_category(raw):
    r = run(raw)
    assert r["parse_status"] == sc.PARSE_UNPARSEABLE
    assert r["counts"] == c(0, 0, 2, 0, 0, 7)


# --- Out of scope and empty cases ------------------------------------------------------------


def test_out_of_scope_kinds_do_not_affect_precision_or_recall():
    oos = [{"kind": "injected_payload", "vl_id": 202, "sn": 136, "t_s": 0.832},
           {"kind": "sn_gap", "vl_id": 100}, {"kind": "late_frame_gap", "vl_id": 201, "sn": 5, "t_s": 0.03}]
    r = run(answer(bag(), *oos), BAG_EVENTS)
    assert r["counts"] == c(1, 0, 0, 1, 0, 0)
    assert r["out_of_scope_count"] == 3
    assert run(answer(*oos), sc.events_from_truth(CLEAN))["counts"] == c(0, 0, 0, 0, 0, 0)


def test_empty_cases_give_na_without_crash():
    clean_events = sc.events_from_truth(CLEAN)
    assert counts(answer(), clean_events) == c(0, 0, 0, 0, 0, 0)
    agg = sc.aggregate([counts(answer(), clean_events)])
    assert all(agg[k] is None for k in
               ("event_precision", "event_recall", "frame_precision", "frame_recall"))
    assert sc.aggregate([])["event_precision"] is None
    assert sc.fmt(None) == "n/a"


def test_benign_only_scenario_with_false_alarm():
    # Flagging the late_frame_gap look-alike as a BAG violation is a false positive.
    alarm = {"kind": "bag_violation", "vl_id": 201, "sn": 50, "network": "A", "t_s": 0.029}
    events = sc.events_from_truth(BENIGN_ONLY)
    assert counts(answer(alarm), events) == c(0, 1, 0, 0, 1, 0)
    agg = sc.aggregate([counts(answer(alarm), events)])
    assert agg["event_precision"] == 0.0 and agg["event_recall"] is None


def test_micro_average_sums_counts_across_samples():
    samples = [
        counts(answer(bag()), BAG_EVENTS),                                   # 1 TP
        counts(answer(), BAG_EVENTS),                                        # 1 FN
        counts(answer(bag(vl_id=7)), sc.events_from_truth(CLEAN)),           # 1 FP on a clean trace
    ]
    agg = sc.aggregate(samples)
    assert (agg["event_tp"], agg["event_fp"], agg["event_fn"]) == (1, 1, 1)
    assert agg["event_precision"] == 0.5 and agg["event_recall"] == 0.5


# --- Inspect scorer, metrics, rescoring --------------------------------------------------------


def _state(raw: str, metadata: dict | None = None) -> TaskState:
    return TaskState(model=ModelName("mockllm/model"), sample_id="s1", epoch=1, input="triage",
                     messages=[], output=ModelOutput.from_content(model="mockllm/model", content=raw),
                     metadata=metadata or {})


@pytest.mark.asyncio
async def test_inspect_score_value_answer_and_metadata():
    raw = "Findings:\n" + answer(bag(), {"kind": "sn_gap", "vl_id": 1}, bag(sn=1))
    meta = {"scenario_id": "M01_mixed", "group": "mixed", "corpus_manifest_sha256": "ab" * 32}
    result = await sc.afdx_anomaly_scorer()(_state(raw, meta), Target(sc.canonical_events(BAG_EVENTS)))
    assert result is not None
    assert result.value == {**c(1, 1, 0, 1, 1, 0), "unparseable": 0}
    assert result.answer == raw
    md = result.metadata or {}
    for key in ("scorer_version", "tolerance_us", "matching_rule", "event_tp", "event_fp",
                "event_fn", "frame_tp", "frame_fp", "frame_fn", "unmatched_findings",
                "unmatched_events", "malformed_findings", "parse_status", "out_of_scope_count",
                "network_mismatch_count", "corpus_manifest_sha256"):
        assert key in md, key
    assert md["scorer_version"] == sc.ANOMALY_SCORER_VERSION
    assert md["tolerance_us"] == 1000 and md["parse_status"] == sc.PARSE_OK_EXTRACTED
    assert md["out_of_scope_count"] == 1 and md["scenario_id"] == "M01_mixed"
    json.dumps(md)  # metadata must be serializable into the eval log


@pytest.mark.asyncio
async def test_rescore_from_raw_output_reproduces_score():
    raw = answer(bag())
    target = sc.canonical_events(BAG_EVENTS)
    first = await sc.afdx_anomaly_scorer()(_state(raw), Target(target))
    again = await rescore_raw_output(sc.afdx_anomaly_scorer(), raw, target, {})
    assert first is not None and again.value == first.value


def test_scorer_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        sc.afdx_anomaly_scorer(tolerance_us=-1)


def _ss(value):
    return SampleScore(score=Score(value=value), sample_id="x")


def _metric_value(metric: Metric, scores: list[SampleScore]) -> float:
    # Metric is MetricProtocol | MetricDeprecated (list[Score]). These metrics take
    # list[SampleScore], which is what Inspect 0.3.273 passes them, and return a float.
    return cast(float, cast(MetricProtocol, metric)(scores))


def test_metrics_micro_average_and_nan_on_zero_denominator():
    scores = [_ss({**c(1, 0, 0, 1, 0, 5), "unparseable": 0}),
              _ss({**c(0, 1, 1, 0, 1, 1), "unparseable": 0}),
              _ss({**c(0, 0, 1, 0, 0, 1), "unparseable": 1})]
    assert _metric_value(sc.event_precision(), scores) == 0.5
    assert _metric_value(sc.event_recall(), scores) == pytest.approx(1 / 3)
    assert _metric_value(sc.frame_precision(), scores) == 0.5
    assert _metric_value(sc.frame_recall(), scores) == pytest.approx(1 / 8)
    assert _metric_value(sc.unparseable_count(), scores) == 1
    empty = [_ss({**c(0, 0, 0, 0, 0, 0), "unparseable": 0})]
    assert math.isnan(_metric_value(sc.event_precision(), empty))
    assert math.isnan(_metric_value(sc.event_recall(), empty))


def test_end_to_end_inspect_eval_reports_precision_and_recall(tmp_path):
    target = sc.canonical_events(BAG_EVENTS)
    task = Task(
        dataset=[Sample(input="t1", target=target), Sample(input="t2", target=target),
                 Sample(input="t3", target=sc.canonical_events([]))],
        solver=[generate()],
        scorer=sc.afdx_anomaly_scorer(),
    )
    outputs = [ModelOutput.from_content("mockllm/model", answer(bag())),
               ModelOutput.from_content("mockllm/model", "no idea"),
               ModelOutput.from_content("mockllm/model", answer(bag(vl_id=1)))]
    model = get_model("mockllm/model", custom_outputs=outputs)
    log = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success" and log.results is not None
    metrics = log.results.scores[0].metrics
    # samples: 1 TP; unparseable -> 1 FN; clean with 1 FP
    assert metrics["event_precision"].value == 0.5
    assert metrics["event_recall"].value == 0.5
    assert metrics["unparseable_count"].value == 1
    assert "f1" not in " ".join(metrics).lower()


# --- Corpus plumbing (hand-written mini corpus) ---------------------------------------------------


def test_sample_fields_fit_the_task_loader(tmp_path):
    (tmp_path / "pre").mkdir()
    (tmp_path / "pre" / "afdx_test.txt").write_text("AFDX premises.\n")
    fields = sc.sample_fields({"id": "S001_bag", "category": "anomaly"}, BAG_ONLY, "cd" * 32, "1.0")
    record = {"id": "c-test-1", "preamble": "afdx_test", "source": "spec-synthesized",
              "input": "List the BAG and SkewMax violations.", **fields}
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(json.dumps(record) + "\n")
    sample = list(load_tasks(task_file, tmp_path / "pre", verify=False))[0]
    assert sample.target == sc.canonical_events(BAG_EVENTS)
    md = sample.metadata or {}
    assert md["family"] == "C" and md["group"] == "anomaly" and md["events"] == BAG_EVENTS
    assert md["corpus_manifest_sha256"] == "cd" * 32
    assert md["anomaly_scorer_version"] == sc.ANOMALY_SCORER_VERSION


def test_load_corpus_resolves_truth_through_manifest(tmp_path):
    root = tmp_path / "corpus"
    (root / "truth").mkdir(parents=True)
    (root / "truth" / "S001_bag.json").write_text(json.dumps(BAG_ONLY))
    (root / "truth" / "C001_clean.json").write_text(json.dumps(CLEAN))
    manifest = {"generator": {"schema_version": "1.0"},
                "scenarios": [{"id": "S001_bag", "category": "anomaly"},
                              {"id": "C001_clean", "category": "clean"}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    scenarios, digest = sc.load_corpus(root / "manifest.json")
    assert digest == sha256_file(root / "manifest.json")
    assert [len(s.fields["metadata"]["events"]) for s in scenarios] == [1, 0]
    assert scenarios[0].fields["metadata"]["corpus_schema_version"] == "1.0"


def test_module_does_not_read_receiver_decisions():
    src = Path(sc.__file__).read_text()
    assert '"decisions"' not in src and "['decisions']" not in src
