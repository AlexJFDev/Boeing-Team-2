"""Cross-checks the labeled traces against the independent oracle (afdx_oracle).

The generator labels anomalies by construction; the oracle re-derives them from the
pcap bytes alone. Any disagreement is a bug in one of them. Run the generator first:
    python generate_afdx_traces.py
"""
import json
from pathlib import Path

import pytest

from bench.afdx import afdx_oracle as oracle
from bench.afdx import afdx_synth as afdx
from bench.afdx.generate_afdx_traces import T0

ROOT = Path(__file__).resolve().parents[1] / "data" / "afdx"
pytestmark = pytest.mark.skipif(not (ROOT / "manifest.json").exists(),
                                reason="run generate_afdx_traces.py first")

MANIFEST = json.loads((ROOT / "manifest.json").read_text()) if (ROOT / "manifest.json").exists() else {"scenarios": [], "summary": {}}
IDS = [s["id"] for s in MANIFEST["scenarios"]]


def load(sid):
    truth = json.loads((ROOT / "truth" / f"{sid}.json").read_text())
    frames = oracle.load_frames(ROOT / truth["pcap"]["A"], ROOT / truth["pcap"]["B"])
    vls = {v["vl_id"]: v for v in truth["config"]["virtual_links"]}
    return truth, frames, vls


def labeled(truth, label):
    return {f["id"] for f in truth["frames"] if label in f["labels"]}


# ---------------------------------------------------------------- corpus-level
def test_corpus_size_and_coverage():
    s = MANIFEST["summary"]
    assert s["scenarios"] >= 150
    assert s["frames_total"] >= 30000
    assert s["by_category"]["anomaly"] >= 50 and s["by_category"]["benign"] >= 30
    assert s["by_category"]["mixed"] >= 30 and s["by_category"]["clean"] >= 5
    assert all(n >= 8 for n in s["by_kind"].values())


@pytest.mark.parametrize("sid", IDS)
def test_config_in_every_scenario_passes_schema_validation(sid):
    truth, _, _ = load(sid)
    assert afdx.validate_config(truth["config"]) == []


# ---------------------------------------------------------------- label vs oracle
@pytest.mark.parametrize("sid", IDS)
def test_oracle_detectors_match_labels_exactly(sid):
    truth, frames, vls = load(sid)
    found = oracle.analyze(frames, vls)
    assert found["bag_violation"] == labeled(truth, "bag_violation")
    assert found["skew_violation"] == labeled(truth, "skew_violation")
    assert found["spoofed_source"] == labeled(truth, "spoofed_source")
    assert found["unknown_vl"] == labeled(truth, "unknown_vl")
    assert found["lmax_exceeded"] == labeled(truth, "lmax_exceeded")
    # fault = bad UDP checksum; injection = A/B mismatch with BOTH checksums valid
    assert found["bad_udp_checksum"] == labeled(truth, "bus_fault_corruption")
    injected = {a for a, b in found["ab_mismatch_pairs"]
                if a not in found["bad_udp_checksum"] and b not in found["bad_udp_checksum"]}
    assert injected == labeled(truth, "injected_payload")  # the altered copy is always the A copy


@pytest.mark.parametrize("sid", IDS)
def test_benign_oddities_are_never_flagged_by_detectors(sid):
    truth, frames, vls = load(sid)
    found = oracle.analyze(frames, vls)
    flagged = set().union(*(found[k] for k in oracle.DETECTORS))
    benign = {f["id"] for f in truth["frames"] if f["benign"]}
    assert not (flagged & benign), "a benign look-alike tripped a detector"


# ---------------------------------------------------------------- sequence / redundancy
@pytest.mark.parametrize("sid", [s["id"] for s in MANIFEST["scenarios"] if s["kind"] in ("sn_gap", "sn_replay")])
def test_sn_anomaly_frames_are_dropped_by_integrity_check_under_both_rules(sid):
    truth, _, _ = load(sid)
    cause = [f for f in truth["frames"] if f["labels"]]
    assert cause
    for f in cause:
        for rule in oracle.RESYNC_RULES:
            assert f["decisions"][rule]["ic"] == "drop"


@pytest.mark.parametrize("sid", [s["id"] for s in MANIFEST["scenarios"] if s["kind"] in ("clean", "sn_wrap", "single_frame_loss", "late_frame_gap", "skew_offset_within_limit")])
def test_conformant_traffic_has_no_integrity_drops(sid):
    truth, _, _ = load(sid)
    for f in truth["frames"]:
        for rule in oracle.RESYNC_RULES:
            assert f["decisions"][rule]["ic"] == "accept"


@pytest.mark.parametrize("sid", [s["id"] for s in MANIFEST["scenarios"] if s["kind"] == "clean"])
def test_clean_ab_pairs_deliver_once_and_discard_the_other(sid):
    truth, _, _ = load(sid)
    rm = [f["decisions"]["drop_with_resync"]["rm"] for f in truth["frames"] if f["vl_id"] == 100]
    assert rm.count("deliver") == rm.count("discard_redundant") == len(rm) // 2


def test_resync_rule_changes_outcomes_on_sn_gap():
    """Catalogue C1: same frames, two rules, different accept/drop sets."""
    differing = 0
    for s in MANIFEST["scenarios"]:
        if s["kind"] != "sn_gap":
            continue
        truth, _, _ = load(s["id"])
        a = [f["decisions"]["drop_with_resync"]["ic"] for f in truth["frames"]]
        b = [f["decisions"]["drop_without_resync"]["ic"] for f in truth["frames"]]
        differing += a != b
    assert differing == sum(1 for s in MANIFEST["scenarios"] if s["kind"] == "sn_gap")


def test_network_dropout_makes_resync_rule_matter_for_b_only():
    for s in MANIFEST["scenarios"]:
        if s["kind"] != "network_dropout":
            continue
        truth, _, _ = load(s["id"])
        resume = [f for f in truth["frames"] if "network_b_resume_after_dropout" in f["benign"]][0]
        assert resume["decisions"]["drop_with_resync"]["ic"] == "drop"
        assert resume["decisions"]["drop_without_resync"]["ic"] == "drop"
        # A keeps flowing, so every SN is still delivered once under either rule
        a_frames = [f for f in truth["frames"] if f["network"] == "A" and f["vl_id"] == 100]
        assert all(f["decisions"]["drop_with_resync"]["ic"] == "accept" for f in a_frames)


# ---------------------------------------------------------------- oracle self-tests
def test_nxt_wraps_255_to_1_and_skips_zero():
    assert oracle.nxt(255) == 1 and oracle.nxt(254, 2) == 1 and oracle.nxt(255, 2) == 2
    assert oracle.in_window(254, 255) and oracle.in_window(254, 1) and not oracle.in_window(254, 2)
    assert oracle.in_window(255, 1) and oracle.in_window(255, 2) and not oracle.in_window(255, 3)


@pytest.mark.parametrize("sid", IDS)
def test_pcap_roundtrip_preserves_vl_sn_and_time_against_truth(sid):
    truth, frames, _ = load(sid)
    by_id = {f["id"]: f for f in frames}
    for f in truth["frames"]:
        assert by_id[f["id"]]["vl_id"] == f["vl_id"] and by_id[f["id"]]["sn"] == f["sn"]
        assert abs((by_id[f["id"]]["t"] - oracle_t0(frames)) - f["t_s"]) < 5e-6


def oracle_t0(frames):
    return T0


# ---------------------------------------------------------------- backlog acceptance criteria
TIMING_KINDS = ("bag_violation", "skew_violation")


@pytest.mark.parametrize("sid", IDS)
def test_every_violation_has_an_exact_timestamp_matching_the_pcap(sid):
    """AC1: each injected violation is recorded with the timestamp of its frame in the pcap."""
    truth, frames, _ = load(sid)
    by_id = {f["id"]: f for f in frames}
    rows = {f["id"]: f for f in truth["frames"]}
    labeled_total = sum(len(f["labels"]) for f in truth["frames"])
    assert len(truth["violations"]) == labeled_total          # nothing dropped, nothing invented
    for v in truth["violations"]:
        assert v["t_us"] == round(v["t_s"] * 1e6)
        assert v["t_s"] == rows[v["frame_id"]]["t_s"]          # agrees with the per-frame row
        assert v["kind"] in rows[v["frame_id"]]["labels"]
        assert abs((by_id[v["frame_id"]]["t"] - T0) - v["t_s"]) <= oracle.TOL   # agrees with the pcap bytes
        # the entry must also name the right network, VL and SN, per the frame row and the pcap bytes
        r, p = rows[v["frame_id"]], by_id[v["frame_id"]]
        assert (v["network"], v["vl_id"], v["sn"]) == (r["network"], r["vl_id"], r["sn"])
        assert (v["network"], v["vl_id"], v["sn"]) == (p["net"], p["vl_id"], p["sn"])


@pytest.mark.parametrize("sid", IDS)
def test_oracle_timing_violations_match_labeled_timestamps(sid):
    """AC1: what the oracle finds in the pcaps, with its timestamp, equals the labeled (frame, time)."""
    truth, frames, vls = load(sid)
    found = oracle.analyze(frames, vls)
    by_id = {f["id"]: f for f in frames}
    for kind in TIMING_KINDS:
        got = {(i, round((by_id[i]["t"] - T0) * 1e6)) for i in found[kind]}
        want = {(v["frame_id"], v["t_us"]) for v in truth["violations"] if v["kind"] == kind}
        assert {i for i, _ in got} == {i for i, _ in want}
        for (i, t_got), (_, t_want) in zip(sorted(got), sorted(want)):
            assert abs(t_got - t_want) <= 1               # within one pcap microsecond


@pytest.mark.parametrize("sid", [s["id"] for s in MANIFEST["scenarios"] if s["category"] in ("clean", "benign")])
def test_no_anomalies_requested_means_zero_violations(sid):
    """AC2: a trace generated without anomalies has no labels and the oracle finds none."""
    truth, frames, vls = load(sid)
    assert truth["anomalies_requested"] == []
    assert truth["violations"] == [] and truth["anomalies"] == []
    found = oracle.analyze(frames, vls)
    assert all(not found[k] for k in oracle.DETECTORS)
    assert found["ab_mismatch_pairs"] == []


def test_requested_anomalies_are_all_recorded():
    for s in MANIFEST["scenarios"]:
        if s["category"] not in ("anomaly", "mixed"):
            continue
        truth, _, _ = load(s["id"])
        assert truth["anomalies_requested"]
        assert {v["kind"] for v in truth["violations"]} == set(truth["anomalies_requested"])