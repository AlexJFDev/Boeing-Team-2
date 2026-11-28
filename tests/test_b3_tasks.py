"""Checks for the B3 ARINC 429 bus conformance task set (WBS 3.5, issue #60).

Demo (issue #60): load a B3 task whose preamble declares bus rate and gap rules and
show the ground-truth verdict follow those rules. Then answer with a rate that
contradicts the preamble and show it score 0.

Acceptance criteria checked here:
- When a B3 task is loaded, its prompt states the bus speed, word timing, gap rule and
  every checked label's transmission rate before the stream, and its target is the
  conformance checker's verdict under exactly those stated rules. A correct answer
  scores 1 end to end.
- When the answer is built on a rate that contradicts the stated one, or on the other
  bus speed, it scores 0 on every task (negative case).
- When a sample's target, premises, prompt rows or words disagree with the checker,
  metadata or generator, the task set fails validation and the eval does not start
  (negative case).
"""

from __future__ import annotations

import functools
import json
import statistics
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from bench.arinc429 import b3, decoder
from bench.arinc429 import conformance as cf
from bench.arinc429.b1 import PREAMBLE, canonical
from bench.corpus import manifest_entries
from bench.loader import TASK_PREMISES_HEADING, load_preamble, load_tasks
from tasks.b3_arinc429_bus_conformance import b3_arinc429_bus_conformance

DATASET = load_tasks(b3.TASK_FILE)
SAMPLES = list(DATASET)


def with_changes(sample: Sample, **changes: Any) -> Sample:
    return sample.model_copy(update=changes, deep=True)


def meta(sample: Sample) -> dict[str, Any]:
    assert sample.metadata is not None
    return sample.metadata


def prompt(sample: Sample) -> str:
    assert isinstance(sample.input, str)
    return sample.input


def target(sample: Sample) -> list[list[Any]]:
    assert isinstance(sample.target, str)
    return json.loads(sample.target)["violations"]


def by_case(case: str) -> list[Sample]:
    return [s for s in SAMPLES if meta(s)["case"] == case]


def stated_rules(sample: Sample) -> cf.BusRules:
    bus = meta(sample)["bus"]
    return b3.rules_for(bus["speed"], bus["stated_intervals_us"])


# --- The committed file --------------------------------------------------------


def test_committed_file_matches_the_builder(tmp_path: Path) -> None:
    rebuilt = b3.write_task_file(tmp_path / b3.TASK_FILE.name)
    assert rebuilt.read_bytes() == b3.TASK_FILE.read_bytes(), (
        "data/tasks B3 file differs from scripts/arinc429/build_b3_tasks.py output; "
        "rebuild it (and bump B3_VERSION if a run has used this version)"
    )


def test_file_is_listed_in_the_manifest() -> None:
    assert b3.TASK_FILE.name in manifest_entries(b3.TASK_FILE.parent / "MANIFEST.json")


def test_task_file_name_carries_the_version() -> None:
    assert b3.TASK_FILE.name == f"b3_arinc429_bus_conformance_v{b3.B3_VERSION}.jsonl"


# --- The stated rules ------------------------------------------------------------


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_prompt_states_the_rules_before_the_stream(sample: Sample) -> None:
    text = prompt(sample)
    assert text.startswith(load_preamble(PREAMBLE))
    rules_at = text.index(TASK_PREMISES_HEADING)
    stream_at = text.index(b3.ROW_HEADER)
    assert rules_at < stream_at
    bus = meta(sample)["bus"]
    kbit = "100" if bus["speed"] == "high" else "12.5"
    assert f"{bus['speed']} speed, {kbit} kbit/s" in text
    assert "at least 4 bit times" in text
    for label, (lo, hi) in bus["rate_windows_us"].items():
        assert f"Label {label} is transmitted every" in text
        assert f"must be from {lo} to {hi} us inclusive" in text


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_prompt_reads_back_to_the_metadata(sample: Sample) -> None:
    speed, rates, stream = b3.parse_prompt(prompt(sample))
    bus = meta(sample)["bus"]
    assert speed == bus["speed"] and rates == bus["stated_intervals_us"]
    assert [w["start_us"] for w in stream] == [w["start_us"] for w in meta(sample)["stream"]]


def test_windows_are_ten_percent_either_side_of_the_stated_interval() -> None:
    for sample in SAMPLES:
        bus = meta(sample)["bus"]
        for label, interval in bus["stated_intervals_us"].items():
            assert bus["rate_windows_us"][label] == [interval * 9 // 10, interval * 11 // 10]


def test_the_same_label_is_stated_at_different_rates_in_different_tasks() -> None:
    rates: dict[str, set[int]] = {}
    for sample in SAMPLES:
        for label, interval in meta(sample)["bus"]["stated_intervals_us"].items():
            rates.setdefault(label, set()).add(interval)
    assert sum(len(v) > 1 for v in rates.values()) >= 3, rates


# --- Coverage ------------------------------------------------------------------


def test_coverage() -> None:
    cases = [meta(s)["case"] for s in SAMPLES]
    assert len(SAMPLES) == 16
    assert cases.count("conforming") == 3
    for case in ("rate_slow", "rate_fast", "gap", "word_timing", "mixed"):
        assert cases.count(case) >= 2, case
    speeds = [meta(s)["bus"]["speed"] for s in SAMPLES]
    assert speeds.count("high") == 11 and speeds.count("low") == 5


def test_every_rule_is_exercised_at_each_speed() -> None:
    for speed in ("high", "low"):
        rules = {r for s in SAMPLES if meta(s)["bus"]["speed"] == speed for _, r in target(s)}
        assert rules == set(cf.RULES), speed


def test_controls_conform_and_faults_break_their_rule() -> None:
    expected = {"rate_slow": {"rate_slow"}, "rate_fast": {"rate_fast"}, "gap_short": {"gap"},
                "overlap": {"gap"}, "word_long": {"word_timing"}, "word_short": {"word_timing"}}
    for sample in SAMPLES:
        rules = {r for _, r in target(sample)}
        want = set().union(*(expected[f] for f in meta(sample)["faults"]))
        assert rules == want, sample.id
    assert all(target(s) == [] for s in by_case("conforming"))


def test_extra_transmission_flags_two_words_and_a_drop_flags_one() -> None:
    for sample in SAMPLES:
        faults = meta(sample)["faults"]
        rules = [r for _, r in target(sample)]
        assert rules.count("rate_fast") == (2 if "rate_fast" in faults else 0), sample.id
        assert rules.count("rate_slow") == (1 if "rate_slow" in faults else 0), sample.id


def test_overlap_item_really_overlaps() -> None:
    [sample] = [s for s in SAMPLES if "overlap" in meta(s)["faults"]]
    checks = cf.check_stream(meta(sample)["stream"], stated_rules(sample))
    assert any(c.gap_us is not None and c.gap_us < 0 for c in checks)


def test_distractor_labels_are_irregular_and_never_flagged() -> None:
    with_distractor = [s for s in SAMPLES if meta(s)["distractor_label"]]
    assert len(with_distractor) == 4
    for sample in with_distractor:
        label = meta(sample)["distractor_label"]
        assert label not in meta(sample)["bus"]["stated_intervals_us"]
        starts = [w["start_us"] for w in meta(sample)["stream"] if w["label"] == label]
        intervals = [b - a for a, b in zip(starts, starts[1:])]
        assert len(intervals) >= 2
        mean = statistics.mean(intervals)
        assert any(abs(i - mean) > mean / 10 for i in intervals), sample.id
        flagged = {n for n, _ in target(sample)}
        numbers = [n for n, w in enumerate(meta(sample)["stream"], start=1) if w["label"] == label]
        assert not flagged & set(numbers), sample.id


def test_words_are_good_parity_on_their_shown_label_and_never_repeat() -> None:
    seen = []
    for sample in SAMPLES:
        for w in meta(sample)["stream"]:
            word = int(w["word_hex"], 16)
            assert decoder.decode(word).status == "decoded"
            assert decoder.read_label(word) == w["label"]
            seen.append(w["word_hex"])
    assert len(seen) == len(set(seen))


# --- Each sample ------------------------------------------------------------------


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_is_the_checker_verdict_under_the_stated_rules(sample: Sample) -> None:
    _, rates, stream = b3.parse_prompt(prompt(sample))
    assert target(sample) == cf.violations(stream, stated_rules(sample))


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_answers_under_contradicting_rules_are_wrong(sample: Sample) -> None:
    assert meta(sample)["contradicting_rate_answer"] != sample.target
    assert meta(sample)["wrong_speed_answer"] != sample.target


def test_task_text_does_not_leak_the_answer() -> None:
    for sample in SAMPLES:
        assert sample.target not in prompt(sample) or sample.target == '{"violations":[]}'


# --- Validation (negative cases) --------------------------------------------------


def test_committed_set_passes_validation() -> None:
    b3.validate_samples(SAMPLES)


def test_contradicting_rate_target_fails_validation() -> None:
    sample = by_case("rate_slow")[0]
    bad = with_changes(sample, target=meta(sample)["contradicting_rate_answer"])
    with pytest.raises(b3.B3ValidationError, match="does not match the conformance checker"):
        b3.validate_samples([bad])


def test_rate_edited_in_the_prompt_fails_validation() -> None:
    sample = by_case("rate_slow")[0]
    label, interval = next(iter(meta(sample)["bus"]["stated_intervals_us"].items()))
    edited = prompt(sample).replace(f"({interval} us)", f"({interval * 2} us)", 1)
    with pytest.raises(b3.B3ValidationError, match="prompt states"):
        b3.validate_samples([with_changes(sample, input=edited)])


def test_premises_that_disagree_with_the_bus_metadata_fail() -> None:
    sample = SAMPLES[0]
    changed = {**meta(sample), "premises": b3.render_premises("low", meta(sample)["bus"]["stated_intervals_us"])}
    with pytest.raises(b3.B3ValidationError, match="premises do not state"):
        b3.validate_samples([with_changes(sample, metadata=changed)])


def test_time_edited_in_the_prompt_fails_validation() -> None:
    sample = SAMPLES[0]
    w = meta(sample)["stream"][1]
    edited = prompt(sample).replace(f"{w['start_us']:>9}", f"{w['start_us'] - 1:>9}", 1)
    with pytest.raises(b3.B3ValidationError, match="prompt rows do not match"):
        b3.validate_samples([with_changes(sample, input=edited)])


def test_word_swapped_for_another_generated_word_fails() -> None:
    sample, other = SAMPLES[0], SAMPLES[1]
    stream = [dict(w) for w in meta(sample)["stream"]]
    stream[0]["word_hex"] = meta(other)["stream"][0]["word_hex"]
    with pytest.raises(b3.B3ValidationError, match="is not word"):
        b3.validate_samples([with_changes(sample, metadata={**meta(sample), "stream": stream})])


def test_every_failing_sample_is_reported() -> None:
    bad = [with_changes(s, target=meta(s)["wrong_speed_answer"]) for s in SAMPLES[:3]]
    with pytest.raises(b3.B3ValidationError) as err:
        b3.validate_samples(bad)
    assert "3 sample(s)" in str(err.value)
    assert all(str(s.id) in str(err.value) for s in bad)


def test_task_refuses_to_build_from_a_bad_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [json.loads(line) for line in b3.TASK_FILE.read_text().splitlines()]
    records[2]["target"] = records[2]["metadata"]["contradicting_rate_answer"]
    bad_file = tmp_path / "b3_bad.jsonl"
    bad_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr("tasks.b3_arinc429_bus_conformance.TASK_FILE", bad_file)
    monkeypatch.setattr("tasks.b3_arinc429_bus_conformance.load_tasks",
                        functools.partial(load_tasks, verify=False))
    with pytest.raises(b3.B3ValidationError, match=records[2]["id"]):
        b3_arinc429_bus_conformance()


# --- End to end with inspect eval ---------------------------------------------------


def scripted_model(answer_for):
    """mockllm answering each prompt with ``answer_for(speed, rates, stream)``."""
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        output = ModelOutput.from_content("mockllm/model", answer_for(*b3.parse_prompt(messages[-1].text)))
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)  # no tokenizer download
        return output
    return get_model("mockllm/model", custom_outputs=respond)


def run_scores(answer_for, tmp_path: Path) -> dict[str, float]:
    [log] = inspect_eval(b3_arinc429_bus_conformance(), model=scripted_model(answer_for),
                         log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    assert log.samples is not None and len(log.samples) == len(SAMPLES)
    assert log.eval.task_version == b3.B3_TASK_VERSION
    return {str(s.id): float(next(iter(s.scores.values())).value) for s in log.samples}


def test_answers_under_the_stated_rules_pass_end_to_end(tmp_path: Path) -> None:
    scores = run_scores(lambda speed, rates, stream: canonical(b3.answer(stream, b3.rules_for(speed, rates))),
                        tmp_path)
    assert set(scores.values()) == {1.0}


@pytest.mark.parametrize("factor", [2, 3])
def test_a_rate_contradicting_the_premises_scores_zero_end_to_end(factor: int, tmp_path: Path) -> None:
    scores = run_scores(
        lambda speed, rates, stream: canonical(b3.answer(stream, b3.rules_for(speed, rates, factor))),
        tmp_path)
    assert set(scores.values()) == {0.0}


def test_the_wrong_bus_speed_scores_zero_end_to_end(tmp_path: Path) -> None:
    def other(speed, rates, stream):
        return canonical(b3.answer(stream, b3.rules_for("low" if speed == "high" else "high", rates)))
    scores = run_scores(other, tmp_path)
    assert set(scores.values()) == {0.0}


def test_saying_everything_conforms_passes_only_the_controls(tmp_path: Path) -> None:
    scores = run_scores(lambda *_: '{"violations":[]}', tmp_path)
    controls = {str(s.id) for s in by_case("conforming")}
    assert {k for k, v in scores.items() if v == 1.0} == controls
