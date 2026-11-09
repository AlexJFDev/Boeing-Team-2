"""Checks for the B2 ARINC 429 SSM interpretation task set (WBS 3.4, issue #59).

Demo (issue #59): run a B2 task asking the model to interpret a word's SSM, and show a
correct answer pass and a swapped-SSM answer fail against the reference decoder.

Acceptance criteria checked here:
- When the task set is run with `inspect eval`, it completes end to end. Each sample
  shows two words with their labels, and its target comes from the reference decoder.
  A correct answer scores 1.
- When the SSM tables are swapped between BNR and BCD (catalogue B2), the answer fails
  on every item where the tables differ (negative case). So does reading every word
  with one table.
- When a sample's target does not match the reference decoder, the task set fails
  validation and the eval does not start (negative case).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from bench.arinc429 import b2, decoder
from bench.arinc429.b1 import PREAMBLE, canonical
from bench.corpus import manifest_entries
from bench.loader import load_preamble, load_tasks
from tasks.b2_arinc429_ssm_interpretation import b2_arinc429_ssm_interpretation

DATASET = load_tasks(b2.TASK_FILE)
SAMPLES = list(DATASET)

# The SSM tables as the task catalogue (B2) prints them, keyed by bit 31 then bit 30.
CATALOGUE = {
    "BNR": {"00": "failure_warning", "01": "no_computed_data",
            "10": "functional_test", "11": "normal_operation"},
    "BCD": {"00": "plus", "01": "no_computed_data", "10": "functional_test", "11": "minus"},
}


def with_changes(sample: Sample, **changes: Any) -> Sample:
    return sample.model_copy(update=changes, deep=True)


def meta(sample: Sample) -> dict[str, Any]:
    assert sample.metadata is not None
    return sample.metadata


def prompt(sample: Sample) -> str:
    assert isinstance(sample.input, str)
    return sample.input


def target(sample: Sample) -> dict[str, str]:
    assert isinstance(sample.target, str)
    return json.loads(sample.target)


def by_case(case: str) -> list[Sample]:
    return [s for s in SAMPLES if meta(s)["case"] == case]


def words(sample: Sample) -> list[int]:
    return [int(w, 16) for w in meta(sample)["words"]]


DISCRIMINATING = by_case("ssm_11") + by_case("ssm_00")
CONTROLS = by_case("ssm_01") + by_case("ssm_10")


# --- The committed file --------------------------------------------------------


def test_committed_file_matches_the_builder(tmp_path: Path) -> None:
    rebuilt = b2.write_task_file(tmp_path / b2.TASK_FILE.name)
    assert rebuilt.read_bytes() == b2.TASK_FILE.read_bytes(), (
        "data/tasks B2 file differs from scripts/arinc429/build_b2_tasks.py output; "
        "rebuild it (and bump B2_VERSION if a run has used this version)"
    )


def test_file_is_listed_in_the_manifest() -> None:
    assert b2.TASK_FILE.name in manifest_entries(b2.TASK_FILE.parent / "MANIFEST.json")


def test_task_file_name_carries_the_version() -> None:
    assert b2.TASK_FILE.name == f"b2_arinc429_ssm_interpretation_v{b2.B2_VERSION}.jsonl"


def test_uses_the_family_preamble_which_does_not_give_the_tables_away() -> None:
    preamble = load_preamble(PREAMBLE)
    for sample in SAMPLES:
        assert meta(sample)["preamble"] == PREAMBLE
        assert prompt(sample).startswith(preamble)
    for words_ in ("Normal Operation", "Failure Warning", "No Computed Data", "Functional Test"):
        assert words_.lower() not in preamble.lower()


# --- The SSM tables -------------------------------------------------------------


def test_meaning_codes_match_the_catalogue_tables() -> None:
    for encoding, table in CATALOGUE.items():
        for bits, code in table.items():
            assert b2.MEANING_CODES[decoder.SSM_MEANINGS[encoding][bits]] == code


def test_every_code_is_listed_in_the_prompt() -> None:
    for code in set(b2.MEANING_CODES.values()):
        assert f'"{code}"' in b2.TASK_TEXT


# --- Coverage ------------------------------------------------------------------


def test_coverage() -> None:
    cases = [meta(s)["case"] for s in SAMPLES]
    assert len(SAMPLES) == 14
    assert cases.count("ssm_11") == 4 and cases.count("ssm_00") == 4
    assert cases.count("ssm_01") == 2 and cases.count("ssm_10") == 2
    assert cases.count("parity_fault") == 2


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_each_pair_is_one_bnr_and_one_bcd_with_the_same_ssm(sample: Sample) -> None:
    assert sorted(meta(sample)["encodings"]) == ["BCD", "BNR"]
    first, second = meta(sample)["ssm_bits"]
    assert first == second


def test_word_order_alternates() -> None:
    firsts = [meta(s)["encodings"][0] for s in SAMPLES]
    assert firsts == ["BNR", "BCD"] * 7


def test_every_label_appears_and_no_word_repeats() -> None:
    labels = {label for s in SAMPLES for label in meta(s)["labels"]}
    assert labels == {"203", "204", "206", "213", "324", "012", "013"}
    all_words = [w for s in SAMPLES for w in meta(s)["words"]]
    assert len(all_words) == len(set(all_words))


def test_fault_pairs_have_exactly_one_rejected_word() -> None:
    for sample in by_case("parity_fault"):
        assert sorted(meta(sample)["expected_status"]) == ["decoded", "rejected"]
        assert list(target(sample).values()).count("reject") == 1


# --- Each sample ------------------------------------------------------------------


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_matches_the_reference_decoder(sample: Sample) -> None:
    answer = target(sample)
    for n, word in enumerate(words(sample), start=1):
        result = decoder.decode(word)
        if result.status == "rejected":
            assert answer[f"word_{n}_meaning"] == "reject"
            assert f"word_{n}_ssm" not in answer
            continue
        assert result.fields is not None and result.decoded is not None
        bits = result.fields["ssm_bits"]
        assert answer[f"word_{n}_ssm"] == bits
        assert answer[f"word_{n}_meaning"] == CATALOGUE[result.decoded["encoding"]][bits]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_prompt_shows_each_word_with_its_true_label(sample: Sample) -> None:
    for n, word in enumerate(words(sample), start=1):
        line = f"Word {n}: 0x{word:08X} (label {decoder.read_label(word)})"
        assert line in prompt(sample)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_agrees_with_the_generator(sample: Sample) -> None:
    records = {r["word_id"]: r for r in b2.generated_words()}
    for n, (word_hex, word_id) in enumerate(zip(meta(sample)["words"], meta(sample)["word_ids"]), start=1):
        record = records[word_id]
        assert record["word_hex"] == word_hex
        expected = "reject" if record["decoded"] is None else b2.MEANING_CODES[record["decoded"]["ssm_meaning"]]
        assert target(sample)[f"word_{n}_meaning"] == expected


def test_swapped_and_one_table_answers_are_wrong_exactly_where_the_tables_differ() -> None:
    for sample in DISCRIMINATING:
        for key in ("swapped_answer", "bnr_table_answer", "bcd_table_answer"):
            assert meta(sample)[key] != sample.target, (sample.id, key)
    for sample in CONTROLS:
        for key in ("swapped_answer", "bnr_table_answer", "bcd_table_answer"):
            assert meta(sample)[key] == sample.target, (sample.id, key)


def test_task_text_does_not_leak_the_answer() -> None:
    for sample in SAMPLES:
        text = meta(sample)["task_text"]
        assert "word_1_meaning\":\"" not in text and "Normal Operation" not in text


# --- Validation (negative cases) --------------------------------------------------


def test_committed_set_passes_validation() -> None:
    b2.validate_samples(SAMPLES)


def test_swapped_target_fails_validation() -> None:
    sample = DISCRIMINATING[0]
    with pytest.raises(b2.B2ValidationError, match="does not match the reference decoder"):
        b2.validate_samples([with_changes(sample, target=meta(sample)["swapped_answer"])])


def test_wrong_ssm_bits_fail_validation() -> None:
    sample = by_case("ssm_11")[0]
    wrong = {**target(sample), "word_1_ssm": "10"}
    with pytest.raises(b2.B2ValidationError):
        b2.validate_samples([with_changes(sample, target=canonical(wrong))])


def test_fault_pair_without_the_rejection_fails() -> None:
    sample = by_case("parity_fault")[0]
    answer = target(sample)
    n = next(k[5] for k, v in answer.items() if v == "reject")
    wrong = {**answer, f"word_{n}_meaning": "plus", f"word_{n}_ssm": "00"}
    with pytest.raises(b2.B2ValidationError):
        b2.validate_samples([with_changes(sample, target=canonical(wrong))])


def test_every_failing_sample_is_reported() -> None:
    bad = [with_changes(s, target=meta(s)["swapped_answer"]) for s in DISCRIMINATING[:3]]
    with pytest.raises(b2.B2ValidationError) as err:
        b2.validate_samples(bad)
    assert "3 sample(s)" in str(err.value)
    assert all(str(s.id) in str(err.value) for s in bad)


def test_wrong_label_in_the_prompt_fails() -> None:
    sample = SAMPLES[0]
    label = meta(sample)["labels"][0]
    naive = f"{words(sample)[0] & 0xFF:03o}"  # the forward read, which is a different label
    bad = with_changes(sample, input=prompt(sample).replace(f"(label {label})", f"(label {naive})", 1))
    with pytest.raises(b2.B2ValidationError, match="prompt does not contain"):
        b2.validate_samples([bad])


def test_word_swapped_for_another_generated_word_fails() -> None:
    sample, other = SAMPLES[0], SAMPLES[2]
    changed = {**meta(sample), "words": [meta(other)["words"][0], meta(sample)["words"][1]]}
    with pytest.raises(b2.B2ValidationError, match="is not word"):
        b2.validate_samples([with_changes(sample, metadata=changed)])


def test_sample_without_two_words_fails() -> None:
    sample = SAMPLES[0]
    changed = {**meta(sample), "words": meta(sample)["words"][:1]}
    with pytest.raises(b2.B2ValidationError, match="does not have two words"):
        b2.validate_samples([with_changes(sample, metadata=changed)])


def test_task_refuses_to_build_from_a_bad_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [json.loads(line) for line in b2.TASK_FILE.read_text().splitlines()]
    records[0]["target"] = records[0]["metadata"]["swapped_answer"]
    bad_file = tmp_path / "b2_bad.jsonl"
    bad_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr("tasks.b2_arinc429_ssm_interpretation.TASK_FILE", bad_file)
    monkeypatch.setattr("tasks.b2_arinc429_ssm_interpretation.load_tasks",
                        functools.partial(load_tasks, verify=False))
    with pytest.raises(b2.B2ValidationError, match=records[0]["id"]):
        b2_arinc429_ssm_interpretation()


# --- End to end with inspect eval ---------------------------------------------------


def scripted_model(answer_for):
    """mockllm answering each prompt with ``answer_for(words)``."""
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        text = messages[-1].text
        found = [int(line.split(": ", 1)[1][:10], 16) for line in text.splitlines()
                 if line.startswith(("Word 1: ", "Word 2: "))]
        output = ModelOutput.from_content("mockllm/model", answer_for(found))
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)  # no tokenizer download
        return output
    return get_model("mockllm/model", custom_outputs=respond)


def run_scores(answer_for, tmp_path: Path) -> dict[str, float]:
    [log] = inspect_eval(b2_arinc429_ssm_interpretation(), model=scripted_model(answer_for),
                         log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    assert log.samples is not None and len(log.samples) == len(SAMPLES)
    assert log.eval.task_version == b2.B2_VERSION
    return {str(s.id): float(next(iter(s.scores.values())).value) for s in log.samples}


def test_correct_answers_pass_end_to_end(tmp_path: Path) -> None:
    scores = run_scores(lambda w: canonical(b2.reference_answer(w)), tmp_path)
    assert set(scores.values()) == {1.0}


def test_swapped_ssm_answers_fail_end_to_end(tmp_path: Path) -> None:
    scores = run_scores(lambda w: canonical(b2.table_answer(w, "other")), tmp_path)
    for sample in DISCRIMINATING:
        assert scores[str(sample.id)] == 0.0, sample.id
    for sample in CONTROLS:
        assert scores[str(sample.id)] == 1.0, sample.id


@pytest.mark.parametrize("table", ["BNR", "BCD"])
def test_one_table_for_both_words_fails_end_to_end(table: str, tmp_path: Path) -> None:
    scores = run_scores(lambda w: canonical(b2.table_answer(w, table)), tmp_path)
    assert all(scores[str(s.id)] == 0.0 for s in DISCRIMINATING)


def test_not_rejecting_a_corrupt_word_fails_end_to_end(tmp_path: Path) -> None:
    def no_reject(found: list[int]) -> str:
        parts = []
        for word in found:
            bits = decoder.read_fields(word)["ssm_bits"]
            parts.append({"meaning": "no_computed_data", "ssm": bits})
        return canonical(b2.combine(parts))
    scores = run_scores(no_reject, tmp_path)
    assert all(scores[str(s.id)] == 0.0 for s in by_case("parity_fault"))
