"""Checks for the A2 CPR position decode task set (WBS 2.6, issue #49).

Acceptance criteria:
- When a task provides a valid even/odd pair, the target equals the reference
  position from bench/adsb/cpr.py.
- When the even/odd pair is too old to decode, the target is the stated no-decode
  outcome, not a position.

Targets are also cross-checked against pyModeS and against the position each
frame was encoded from, so the verifier is not only checked against itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from pyModeS import position

from bench.adsb import a2, cpr
from bench.loader import load_preamble, load_tasks
from tasks.a2_cpr_position import a2_cpr_position

DATASET = load_tasks(a2.TASK_FILE)
SAMPLES = list(DATASET)
FRAMES = a2.frames_by_id()


def meta(sample: Sample) -> dict[str, Any]:
    assert sample.metadata is not None
    return sample.metadata


def prompt(sample: Sample) -> str:
    assert isinstance(sample.input, str)
    return sample.input


def target(sample: Sample) -> dict[str, Any]:
    assert isinstance(sample.target, str)
    return json.loads(sample.target)


def by_case(case: str) -> list[Sample]:
    return [s for s in SAMPLES if meta(s)["case"] == case]


def with_changes(sample: Sample, **changes: Any) -> Sample:
    return sample.model_copy(update=changes, deep=True)


def frame(sample: Sample, i: int) -> cpr.CPRFrame:
    f = meta(sample)["frames"][i]
    return cpr.parse_frame(f["hex"], f["timestamp"])


def lon_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


POSITION_CASES = ("global_pair", "local_reference")
NO_DECODE_CASES = ("stale_pair", "same_format", "single_frame")


# The committed task file


def test_committed_file_matches_the_builder(tmp_path: Path) -> None:
    rebuilt = a2.write_task_file(tmp_path / a2.TASK_FILE.name)
    assert rebuilt.read_bytes() == a2.TASK_FILE.read_bytes(), (
        "the A2 task file differs from scripts/adsb/build_a2_tasks.py output; "
        "rebuild it (and bump A2_VERSION if a run has used this version)")


def test_task_file_name_carries_the_version() -> None:
    assert a2.TASK_FILE.name == f"a2_cpr_position_v{a2.A2_VERSION}.jsonl"


def test_coverage() -> None:
    counts = {case: len(by_case(case)) for case in a2.CASES}
    assert counts == {"global_pair": 10, "local_reference": 10, "stale_pair": 2,
                      "same_format": 4, "single_frame": 2}
    for case in POSITION_CASES:
        assert len({meta(s)["track_id"] for s in by_case(case)}) == 10, case


def test_both_newer_frame_formats_are_covered() -> None:
    newer = [frame(s, 1).cpr_format for s in by_case("global_pair")]
    assert newer.count(0) == 5 and newer.count(1) == 5
    local = [frame(s, 0).cpr_format for s in by_case("local_reference")]
    assert local.count(0) == 5 and local.count(1) == 5


def test_same_format_covers_two_evens_and_two_odds() -> None:
    formats = sorted({frame(s, 0).cpr_format for s in by_case("same_format")})
    assert formats == [0, 1]
    for s in by_case("same_format"):
        assert frame(s, 0).cpr_format == frame(s, 1).cpr_format


def test_catalogue_triple_from_one_track() -> None:
    """Catalogue A2: a valid pair, two even frames and a pair ~45 s apart, from the same aircraft."""
    trk04 = {meta(s)["case"] for s in SAMPLES if meta(s)["track_id"] == "trk-04"}
    assert {"global_pair", "same_format", "stale_pair"} <= trk04


# Every sample has a prompt that states its inputs


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_prompt_states_the_inputs(sample: Sample) -> None:
    text = prompt(sample)
    assert text.startswith(load_preamble(a2.PREAMBLE))
    times = [f["timestamp"] for f in meta(sample)["frames"]]
    assert times == sorted(times)
    for f in meta(sample)["frames"]:
        assert f"t = {f['timestamp']:.1f} s: {f['hex']}" in text
        assert FRAMES[f["frame_id"]]["hex"] == f["hex"]
    receiver = meta(sample)["receiver"]
    if receiver:
        assert f"latitude {receiver['latitude']:.2f}, longitude {receiver['longitude']:.2f}" in text
        assert "within 180 NM" in text
    else:
        assert "receiving station" not in text
    assert a2.NO_DECODE_TARGET in text


def test_prompt_does_not_leak_the_answer_or_the_case() -> None:
    for sample in SAMPLES:
        task_text = meta(sample)["task_text"].lower()
        for word in ("even", "odd", "stale", "too old", "same_format", "adsb-"):
            assert word not in task_text, (sample.id, word)
        if meta(sample)["expected_position"]:
            for value in target(sample).values():
                assert str(value) not in prompt(sample), sample.id


# Criterion 1: valid pairs (and local decodes) equal the verifier's position


@pytest.mark.parametrize("sample", by_case("global_pair"), ids=lambda s: str(s.id))
def test_global_target_equals_the_verifier_position(sample: Sample) -> None:
    a, b = meta(sample)["frames"]
    result = cpr.decode_hex_pair(a["hex"], a["timestamp"], b["hex"], b["timestamp"])
    assert result.status == "ok" and result.latitude is not None and result.longitude is not None
    assert target(sample) == {"latitude": round(result.latitude, 5),
                              "longitude": round(result.longitude, 5)}
    assert meta(sample)["expected_position"] == {"latitude": result.latitude,
                                                 "longitude": result.longitude}


@pytest.mark.parametrize("sample", by_case("local_reference"), ids=lambda s: str(s.id))
def test_local_target_equals_the_verifier_position(sample: Sample) -> None:
    f, r = meta(sample)["frames"][0], meta(sample)["receiver"]
    result = cpr.decode_hex_with_reference(f["hex"], r["latitude"], r["longitude"])
    assert result.latitude is not None and result.longitude is not None
    assert target(sample) == {"latitude": round(result.latitude, 5),
                              "longitude": round(result.longitude, 5)}


@pytest.mark.parametrize("sample", by_case("global_pair") + by_case("local_reference"),
                         ids=lambda s: str(s.id))
def test_position_target_agrees_with_pymodes_and_the_encoded_position(sample: Sample) -> None:
    got = target(sample)
    newer = frame(sample, -1)
    if meta(sample)["case"] == "global_pair":
        even, odd = sorted((frame(sample, 0), frame(sample, 1)), key=lambda f: f.cpr_format)
        expected = position.airborne_position_pair(
            even.cpr_lat, even.cpr_lon, odd.cpr_lat, odd.cpr_lon,
            even_is_newer=even.timestamp >= odd.timestamp)
        assert expected is not None
    else:
        r = meta(sample)["receiver"]
        expected = position.airborne_position_with_ref(
            newer.cpr_format, newer.cpr_lat, newer.cpr_lon, r["latitude"], r["longitude"])
    assert got["latitude"] == pytest.approx(expected[0], abs=1e-5)
    assert lon_diff(got["longitude"], expected[1]) < 1e-5
    truth = FRAMES[meta(sample)["frames"][-1]["frame_id"]]["synthesis"]
    assert got["latitude"] == pytest.approx(truth["true_latitude"], abs=1e-3)
    assert lon_diff(got["longitude"], truth["true_longitude"]) < 1e-3
    assert -180 <= got["longitude"] < 180


@pytest.mark.parametrize("sample", by_case("global_pair") + by_case("local_reference"),
                         ids=lambda s: str(s.id))
def test_position_is_not_on_a_rounding_boundary(sample: Sample) -> None:
    """Rounding to 5 places can't flip on floating-point noise (margin over 1e-8 degrees)."""
    for value in meta(sample)["expected_position"].values():
        assert abs((value * 1e5) % 1 - 0.5) > 1e-3, sample.id


@pytest.mark.parametrize("sample", by_case("local_reference"), ids=lambda s: str(s.id))
def test_receiver_is_within_range_but_not_on_top_of_the_aircraft(sample: Sample) -> None:
    r = meta(sample)["receiver"]
    truth = FRAMES[meta(sample)["frames"][0]["frame_id"]]["synthesis"]
    nm = a2.distance_nm((r["latitude"], r["longitude"]),
                        (truth["true_latitude"], truth["true_longitude"]))
    assert 30 < nm < a2.RECEIVER_RANGE_NM


def test_local_set_includes_the_hard_references() -> None:
    notes = " ".join(meta(s)["note"] for s in by_case("local_reference"))
    for phrase in ("equator", "prime meridian", "180", "longitude-zone band", "stale pair"):
        assert phrase in notes, phrase
    for s in by_case("local_reference"):
        if "longitude-zone band" in meta(s)["note"]:
            r = meta(s)["receiver"]
            truth = FRAMES[meta(s)["frames"][0]["frame_id"]]["synthesis"]
            assert cpr.nl(r["latitude"]) != cpr.nl(truth["true_latitude"]), s.id


# Criterion 2: stale pairs get the stated no-decode outcome, not a position


@pytest.mark.parametrize("sample", by_case("stale_pair"), ids=lambda s: str(s.id))
def test_stale_pair_target_is_the_no_decode_outcome(sample: Sample) -> None:
    assert sample.target == a2.NO_DECODE_TARGET
    assert a2.NO_DECODE_TARGET in prompt(sample)  # the outcome is stated in the task
    a, b = meta(sample)["frames"]
    result = cpr.decode_hex_pair(a["hex"], a["timestamp"], b["hex"], b["timestamp"])
    assert result.status == "ambiguous" and result.reason == "pair_too_old"
    assert result.pair_age_s is not None and result.pair_age_s > cpr.PAIR_WINDOW_S
    assert frame(sample, 0).cpr_format != frame(sample, 1).cpr_format
    assert "10 seconds" in prompt(sample)  # the window is stated in the preamble


@pytest.mark.parametrize("sample", by_case("stale_pair"), ids=lambda s: str(s.id))
def test_stale_pair_has_a_plausible_wrong_answer(sample: Sample) -> None:
    """Ignoring the 10 s window still yields a confident position: the overconfidence probe."""
    a, b = meta(sample)["frames"]
    naive = cpr.decode_hex_pair(a["hex"], a["timestamp"], b["hex"], b["timestamp"], window_s=1e9)
    assert naive.status == "ok" and naive.latitude is not None
    assert -90 <= naive.latitude <= 90


@pytest.mark.parametrize("sample", by_case("same_format") + by_case("single_frame"),
                         ids=lambda s: str(s.id))
def test_other_no_decode_cases(sample: Sample) -> None:
    assert sample.target == a2.NO_DECODE_TARGET
    assert meta(sample)["expected_position"] is None
    if meta(sample)["case"] == "same_format":
        assert meta(sample)["verifier_reason"] == "same_format"
    else:
        assert len(meta(sample)["frames"]) == 1 and meta(sample)["receiver"] is None
        assert a2.SINGLE_FRAME_PREMISE in prompt(sample)


# Validation: a target that disagrees with the verifier fails


def test_committed_set_passes_validation() -> None:
    a2.validate_samples(SAMPLES)


def test_stale_pair_with_a_position_target_fails() -> None:
    sample = by_case("stale_pair")[0]
    a, b = meta(sample)["frames"]
    naive = cpr.decode_hex_pair(a["hex"], a["timestamp"], b["hex"], b["timestamp"], window_s=1e9)
    assert naive.latitude is not None and naive.longitude is not None
    bad = with_changes(sample, target=a2.canonical(
        {"latitude": round(naive.latitude, 5), "longitude": round(naive.longitude, 5)}))
    with pytest.raises(a2.A2ValidationError, match=rf"{sample.id}: target .* does not match the CPR verifier"):
        a2.validate_samples([bad])


@pytest.mark.parametrize("case", POSITION_CASES)
def test_position_off_by_one_unit_fails(case: str) -> None:
    sample = by_case(case)[0]
    moved = {**target(sample), "longitude": round(target(sample)["longitude"] + 1e-5, 5)}
    with pytest.raises(a2.A2ValidationError, match=str(sample.id)):
        a2.validate_samples([with_changes(sample, target=a2.canonical(moved))])


def test_valid_pair_with_a_no_decode_target_fails() -> None:
    sample = by_case("global_pair")[0]
    with pytest.raises(a2.A2ValidationError, match=str(sample.id)):
        a2.validate_samples([with_changes(sample, target=a2.NO_DECODE_TARGET)])


def test_receiver_changed_in_metadata_only_fails() -> None:
    sample = by_case("local_reference")[0]
    m = {**meta(sample), "receiver": {"latitude": 40.0, "longitude": -72.45}}
    with pytest.raises(a2.A2ValidationError, match="does not state the receiver position"):
        a2.validate_samples([with_changes(sample, metadata=m)])


def test_frame_dropped_from_the_prompt_fails() -> None:
    sample = by_case("global_pair")[0]
    hex_b = meta(sample)["frames"][1]["hex"]
    with pytest.raises(a2.A2ValidationError, match="prompt does not contain"):
        a2.validate_samples([with_changes(sample, input=prompt(sample).replace(hex_b, "<frame>"))])


def test_frame_not_in_the_frame_set_fails() -> None:
    sample = by_case("global_pair")[0]
    frames = [dict(f) for f in meta(sample)["frames"]]
    frames[1]["timestamp"] = 1030.7
    m = {**meta(sample), "frames": frames}
    with pytest.raises(a2.A2ValidationError, match="does not match the frame set"):
        a2.validate_samples([with_changes(sample, metadata=m)])


def test_every_failing_sample_is_reported() -> None:
    bad = [with_changes(s, target=a2.NO_DECODE_TARGET) for s in by_case("local_reference")[:3]]
    with pytest.raises(a2.A2ValidationError) as err:
        a2.validate_samples(bad)
    assert "3 sample(s)" in str(err.value)
    assert all(str(s.id) in str(err.value) for s in bad)


def test_task_refuses_to_build_from_a_bad_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = a2.TASK_FILE.read_text().splitlines()
    record = json.loads(next(line for line in lines if "stale-pair" in line))
    record["target"] = '{"latitude":52.6,"longitude":3.9}'
    bad_file = tmp_path / "a2_bad.jsonl"
    bad_file.write_text("\n".join(json.dumps(record) if "stale-pair" in line and record["id"] in line
                                  else line for line in lines) + "\n")
    monkeypatch.setattr("tasks.a2_cpr_position.TASK_FILE", bad_file)
    with pytest.raises(a2.A2ValidationError, match=record["id"]):
        a2_cpr_position()


def test_reference_answer_refuses_unusable_inputs() -> None:
    corrupt = next(f for f in FRAMES.values()
                   if f["label"] == "corrupt" and f["message_type"] == "airborne_position")
    with pytest.raises(a2.A2ValidationError, match="not usable"):
        a2.reference_answer([(corrupt["hex"], corrupt["timestamp"])], (40.0, -70.0))
    with pytest.raises(a2.A2ValidationError, match="not an A2 task shape"):
        a2.reference_answer([("8DF00A1158BF01F0FDC0B93FD599", 1.0)] * 2, (40.0, -70.0))


# End to end with inspect eval


def test_inspect_eval_runs_end_to_end(tmp_path: Path) -> None:
    [log] = inspect_eval(a2_cpr_position(), model="mockllm/model", log_dir=str(tmp_path),
                         display="none")
    assert log.status == "success", log.error
    assert log.samples is not None and len(log.samples) == len(SAMPLES)
    assert log.eval.task_version == a2.A2_VERSION
    for sample in log.samples:
        for f in sample.metadata["frames"]:
            assert f["hex"] in sample.messages[0].text
        assert sample.target
        assert sample.store["provenance"]["corpus_hash"]
