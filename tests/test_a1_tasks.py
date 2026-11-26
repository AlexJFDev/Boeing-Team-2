"""Checks for the A1 ADS-B decode task set (WBS 2.5, issue #48).

Acceptance criteria:
- When the task set is run with `inspect eval`, it completes end to end and each
  sample has a prompt, a frame and a ground-truth target.
- When a sample's target does not match the pyModeS reference decode, the task set
  fails validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyModeS as pms
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample

from bench.adsb import a1, oracle
from bench.loader import load_preamble, load_tasks
from tasks.a1_adsb_decode import a1_adsb_decode

DATASET = load_tasks(a1.TASK_FILE)
SAMPLES = list(DATASET)
FRAMES = {f["frame_id"]: f for f in a1.load_frames()}


def with_changes(sample: Sample, **changes) -> Sample:
    return sample.model_copy(update=changes, deep=True)


def meta(sample: Sample) -> dict:
    assert sample.metadata is not None
    return sample.metadata


def prompt(sample: Sample) -> str:
    assert isinstance(sample.input, str)
    return sample.input


def target_of(sample: Sample) -> str:
    assert isinstance(sample.target, str)
    return sample.target


def by_case(case: str) -> list[Sample]:
    return [s for s in SAMPLES if meta(s)["case"] == case]


# The committed task file


def test_committed_file_matches_the_builder(tmp_path: Path) -> None:
    rebuilt = a1.write_task_file(tmp_path / a1.TASK_FILE.name)
    assert rebuilt.read_bytes() == a1.TASK_FILE.read_bytes(), (
        "data/tasks A1 file differs from scripts/adsb/build_a1_tasks.py output; "
        "rebuild it (and bump A1_VERSION if a run has used this version)"
    )


def test_task_file_name_carries_the_version() -> None:
    assert a1.TASK_FILE.name == f"a1_adsb_decode_v{a1.A1_VERSION}.jsonl"


def test_coverage() -> None:
    counts = {case: len(by_case(case)) for case in
              ("df17_identification", "df18_identification", "altitude", "crc_failure")}
    assert counts == {"df17_identification": 10, "df18_identification": 3,
                      "altitude": 10, "crc_failure": 6}
    rejected_types = {FRAMES[meta(s)["frame_id"]]["message_type"] for s in by_case("crc_failure")}
    assert rejected_types == {"identification", "airborne_position"}
    assert any(FRAMES[meta(s)["frame_id"]]["df"] == 18 for s in by_case("crc_failure"))


def test_no_velocity_frames() -> None:
    assert all(FRAMES[meta(s)["frame_id"]]["message_type"] != "airborne_velocity"
               for s in SAMPLES)


def test_one_altitude_item_per_track() -> None:
    tracks = [FRAMES[meta(s)["frame_id"]]["track_id"] for s in by_case("altitude")]
    assert len(tracks) == len(set(tracks)) == 10


# Each sample has a prompt, a frame and a ground-truth target


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_sample_has_prompt_frame_and_target(sample: Sample) -> None:
    frame = meta(sample)["frame"]
    assert isinstance(prompt(sample), str)
    assert prompt(sample).startswith(load_preamble(a1.PREAMBLE))
    assert f"Message: {frame}" in prompt(sample)
    assert len(frame) == 28 and frame == FRAMES[meta(sample)["frame_id"]]["hex"]
    assert isinstance(sample.target, str) and json.loads(target_of(sample))
    assert meta(sample)["subfamily"] == "A1"
    assert meta(sample)["family"] == "A"


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_matches_pymodes(sample: Sample) -> None:
    """Re-derive the target from pyModeS directly, not through the oracle wrapper."""
    frame = meta(sample)["frame"]
    target = json.loads(target_of(sample))
    if pms.decode(frame)["crc_valid"] is not True:
        assert target == {"answer": "reject", "reason": "crc_check_failed"}
        return
    raw = pms.decode(frame)
    assert target["df"] == raw["df"]
    assert target["icao"] == raw["icao"]
    assert target["type_code"] == raw["typecode"]
    # Origin: pyModeS does not report CF, so compare with what the generator encoded.
    encoded = FRAMES[meta(sample)["frame_id"]]
    origin = {"transmitter"} | ({"cf"} if raw["df"] == 18 else set())
    if raw["df"] == 18:
        assert target["cf"] == encoded["cf"]
        assert target["transmitter"] == {0: "non_transponder", 6: "ads_r"}[encoded["cf"]]
    else:
        assert target["transmitter"] == "transponder"
    if target["message_type"] == "identification":
        assert set(target) == {"df", "icao", "type_code", "message_type", "callsign", "category"} | origin
        assert (target["callsign"], target["category"]) == (raw["callsign"], raw["category"])
    else:
        assert target["message_type"] == "airborne_position"
        assert set(target) == {"df", "icao", "type_code", "message_type", "altitude_ft"} | origin
        assert target["altitude_ft"] == raw["altitude"]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_agrees_with_the_frame_set_label(sample: Sample) -> None:
    frame = FRAMES[meta(sample)["frame_id"]]
    rejected = json.loads(target_of(sample)).get("answer") == "reject"
    assert rejected == (frame["expected_answer"] == "reject")


def test_prompt_does_not_leak_the_answer() -> None:
    for sample in SAMPLES:
        task_text = meta(sample)["task_text"]
        target = json.loads(target_of(sample))
        if "callsign" in target:
            assert target["callsign"] not in task_text, sample.id
        assert meta(sample)["frame_id"] not in task_text, sample.id
        assert "corrupt" not in task_text.lower(), sample.id


# Validation: a target that disagrees with pyModeS fails


def test_committed_set_passes_validation() -> None:
    a1.validate_samples(SAMPLES)


@pytest.mark.parametrize("bad_target", [
    '{"callsign":"ZZZ102","category":3,"df":17,"icao":"F00A11","message_type":"identification","type_code":4}',
    '{"callsign":"ZZZ101","category":3,"df":18,"icao":"F00A11","message_type":"identification","type_code":4}',
    '{"callsign":"ZZZ101","category":3,"df":17,"icao":"F00A11","message_type":"identification","type_code":4,"wake_vortex":"Medium 2"}',
    '{"answer":"reject","reason":"crc_check_failed"}',
    '{"callsign": "ZZZ101", "category": 3, "df": 17, "icao": "F00A11", "message_type": "identification", "type_code": 4}',
])
def test_wrong_identification_target_fails(bad_target: str) -> None:
    bad = with_changes(SAMPLES[0], target=bad_target)
    with pytest.raises(a1.A1ValidationError, match=rf"{bad.id}: target .* does not match the pyModeS"):
        a1.validate_samples([*SAMPLES[1:], bad])


def test_wrong_altitude_target_fails() -> None:
    sample = by_case("altitude")[0]
    target = json.loads(target_of(sample))
    target["altitude_ft"] += 25
    bad = with_changes(sample, target=a1.canonical(target))
    with pytest.raises(a1.A1ValidationError, match=str(sample.id)):
        a1.validate_samples([bad])


def test_corrupt_frame_with_a_decoded_target_fails() -> None:
    sample = by_case("crc_failure")[0]
    naive = FRAMES[meta(sample)["frame_id"]]["corruption"]["naive_decode"]
    naive_target = a1.canonical({
        "callsign": naive["callsign"], "category": naive["category"], "df": naive["df"],
        "icao": naive["icao"], "message_type": "identification", "type_code": naive["typecode"],
    })
    with pytest.raises(a1.A1ValidationError, match=str(sample.id)):
        a1.validate_samples([with_changes(sample, target=naive_target)])


def test_every_failing_sample_is_reported() -> None:
    bad = [with_changes(s, target='{"answer":"reject","reason":"crc_check_failed"}')
           for s in by_case("df17_identification")[:3]]
    with pytest.raises(a1.A1ValidationError) as err:
        a1.validate_samples(bad)
    assert "3 sample(s)" in str(err.value)
    assert all(str(s.id) in str(err.value) for s in bad)


def test_frame_swapped_for_another_frame_set_entry_fails() -> None:
    sample, other = SAMPLES[0], SAMPLES[2]
    changed = {**meta(sample), "frame": meta(other)["frame"]}
    swapped = with_changes(sample, metadata=changed,
                           input=prompt(sample).replace(meta(sample)["frame"], meta(other)["frame"]))
    with pytest.raises(a1.A1ValidationError, match="is not frame 'adsb-0001'"):
        a1.validate_samples([swapped])


def test_prompt_without_its_frame_fails() -> None:
    sample = SAMPLES[0]
    bad = with_changes(sample, input=prompt(sample).replace(meta(sample)["frame"], "<frame>"))
    with pytest.raises(a1.A1ValidationError, match="prompt does not contain the frame"):
        a1.validate_samples([bad])


def test_sample_without_a_frame_fails() -> None:
    sample = SAMPLES[0]
    changed = {k: v for k, v in meta(sample).items() if k != "frame"}
    with pytest.raises(a1.A1ValidationError, match="metadata has no frame"):
        a1.validate_samples([with_changes(sample, metadata=changed)])


def test_task_refuses_to_build_from_a_bad_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import functools
    from bench.loader import load_tasks
    lines = a1.TASK_FILE.read_text().splitlines()
    record = json.loads(lines[0])
    target = json.loads(record["target"])
    target["callsign"] = target["callsign"][:-1] + ("9" if target["callsign"][-1] != "9" else "8")
    record["target"] = a1.canonical(target)
    bad_file = tmp_path / "a1_bad.jsonl"
    bad_file.write_text("\n".join([json.dumps(record), *lines[1:]]) + "\n")
    monkeypatch.setattr("tasks.a1_adsb_decode.TASK_FILE", bad_file)
    monkeypatch.setattr("tasks.a1_adsb_decode.load_tasks", functools.partial(load_tasks, verify=False))
    with pytest.raises(a1.A1ValidationError, match="a1-adsb-0001"):
        a1_adsb_decode()


def test_reference_answer_refuses_frames_a1_cannot_use() -> None:
    with pytest.raises(a1.A1ValidationError, match="not usable for A1"):
        a1.reference_answer("5DF00A11FFFFFF")


def test_targets_are_a_subset_of_the_oracle_answer() -> None:
    """Apart from the origin fields, which pyModeS does not report (see a1.origin)."""
    for sample in SAMPLES:
        target = json.loads(target_of(sample))
        decoded = {k: v for k, v in target.items() if k not in ("cf", "transmitter")}
        assert decoded.items() <= oracle.decode(meta(sample)["frame"]).answer.items(), sample.id


# Origin discrimination (data review finding F6)


def test_df18_items_require_distinguishable_origins() -> None:
    """CF=0 and CF=6 frames have different expected classifications."""
    df18 = {json.loads(target_of(s))["cf"]: json.loads(target_of(s))["transmitter"]
            for s in by_case("df18_identification")}
    assert df18 == {0: "non_transponder", 6: "ads_r"}
    df17 = {json.loads(target_of(s)).get("transmitter") for s in SAMPLES
            if meta(s)["expected_status"] == "decoded" and json.loads(target_of(s))["df"] == 17}
    assert df17 == {"transponder"}


def test_prompt_defines_every_transmitter_label() -> None:
    for label in a1.TRANSMITTERS:
        assert f'"{label}"' in a1.TASK_TEXT
    assert set(a1.TRANSMITTER_BY_CF.values()) | {"transponder"} == set(a1.TRANSMITTERS)


@pytest.mark.parametrize("change", [{"transmitter": "transponder"}, {"cf": 0}, {"cf": None}])
def test_wrong_origin_target_fails(change: dict) -> None:
    sample = next(s for s in by_case("df18_identification") if json.loads(target_of(s))["cf"] == 6)
    target = {**json.loads(target_of(sample)), **change}
    target = {k: v for k, v in target.items() if v is not None}
    bad = with_changes(sample, target=a1.canonical(target))
    with pytest.raises(a1.A1ValidationError, match=str(sample.id)):
        a1.validate_samples([bad])


def test_frame_set_cf_disagreement_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CF read from the bits is cross-checked against the generator's encoded CF."""
    sample = by_case("df18_identification")[0]
    frames = {k: dict(v) for k, v in a1._frames_by_id().items()}
    frames[meta(sample)["frame_id"]]["cf"] = 3
    monkeypatch.setattr(a1, "_frames_by_id", lambda: frames)
    with pytest.raises(a1.A1ValidationError, match="disagrees with the encoded CF"):
        a1.validate_samples([sample])


@pytest.mark.asyncio
async def test_correct_callsign_with_wrong_origin_is_not_full_credit() -> None:
    """F6 acceptance check, against both capability scorers."""
    from unittest.mock import MagicMock

    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from bench.scorers import adsb_field_capability_scorer, deterministic_exact_match

    sample = next(s for s in by_case("df18_identification") if json.loads(target_of(s))["cf"] == 6)
    wrong = {**json.loads(target_of(sample)), "transmitter": "transponder"}
    state = MagicMock(spec=TaskState)
    state.sample_id, state.metadata, state.store = sample.id, {}, {}
    state.output = MagicMock()
    for response, full in ((target_of(sample), True), (a1.canonical(wrong), False)):
        state.output.completion = response
        field = await adsb_field_capability_scorer()(state, Target(target_of(sample)))
        exact = await deterministic_exact_match()(state, Target(target_of(sample)))
        assert (field.value == 1.0, exact.value == 1.0) == (full, full), response


# End to end with inspect eval


def test_inspect_eval_runs_end_to_end(tmp_path: Path) -> None:
    [log] = inspect_eval(a1_adsb_decode(), model="mockllm/model", log_dir=str(tmp_path),
                         display="none")
    assert log.status == "success", log.error
    assert log.samples is not None and len(log.samples) == len(SAMPLES)
    assert log.eval.task_version == a1.A1_TASK_VERSION
    for sample in log.samples:
        assert sample.metadata["frame"] in sample.messages[0].text
        assert sample.target
        assert sample.store["provenance"]["corpus_hash"]
