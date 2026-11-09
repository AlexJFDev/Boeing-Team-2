"""Checks for the B1 ARINC 429 word-decode task set (WBS 3.3, issue #58).

Demo (issue #58): run a B1 task asking the model to decode a synthetic ARINC 429 word,
and show the answer scored against the reference decoder's output.

Acceptance criteria checked here:
- When the task set is run with `inspect eval`, it completes end to end. Each sample
  has a prompt with its word, and a target from the reference decoder. A correct
  answer scores 1.
- When a sample's target does not match the reference decoder, the task set fails
  validation and the eval does not start (negative case).
- When a model reads the label octet forward (the B1 discriminator) or decodes a
  corrupt word instead of rejecting it, the answer scores 0 (negative case).
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

from bench.arinc429 import b1, decoder
from bench.arinc429 import generator as gen
from bench.corpus import manifest_entries
from bench.loader import load_preamble, load_tasks
from tasks.b1_arinc429_word_decode import b1_arinc429_word_decode

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = load_tasks(b1.TASK_FILE)
SAMPLES = list(DATASET)


def with_changes(sample: Sample, **changes: Any) -> Sample:
    return sample.model_copy(update=changes, deep=True)


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


# --- The committed files -------------------------------------------------------


def test_committed_file_matches_the_builder(tmp_path: Path) -> None:
    rebuilt = b1.write_task_file(tmp_path / b1.TASK_FILE.name)
    assert rebuilt.read_bytes() == b1.TASK_FILE.read_bytes(), (
        "data/tasks B1 file differs from scripts/arinc429/build_b1_tasks.py output; "
        "rebuild it (and bump B1_VERSION if a run has used this version)"
    )


def test_committed_preamble_matches_the_label_profiles() -> None:
    assert load_preamble(b1.PREAMBLE) == b1.render_preamble(), (
        "data/preambles/arinc429_v1.0.0.txt no longer matches LABEL_PROFILES; a profile "
        "change needs a new preamble version"
    )


def test_files_are_listed_in_their_manifests() -> None:
    tasks = manifest_entries(b1.TASK_FILE.parent / "MANIFEST.json")
    preambles = manifest_entries(b1.PREAMBLE_FILE.parent / "MANIFEST.json")
    assert b1.TASK_FILE.name in tasks
    assert b1.PREAMBLE_FILE.name in preambles


def test_task_file_name_carries_the_version() -> None:
    assert b1.TASK_FILE.name == f"b1_arinc429_word_decode_v{b1.B1_VERSION}.jsonl"


# --- Coverage -----------------------------------------------------------------


def test_coverage() -> None:
    cases = [meta(s)["case"] for s in SAMPLES]
    assert len(SAMPLES) == 27
    assert cases.count("bnr") == 15 and cases.count("bcd") == 6
    assert cases.count("parity_bit_fault") == 2 and cases.count("data_bit_fault") == 2
    assert cases.count("unknown_label") == 2


def test_every_profile_appears_with_several_ssms() -> None:
    decoded = [target(s) for s in SAMPLES if "label" in target(s) and target(s)["value"] != "cannot_be_determined"]
    for profile in gen.LABEL_PROFILES:
        ssms = {t["ssm"] for t in decoded if t["label"] == profile.label_octal}
        assert len(ssms) == b1.PER_PROFILE, profile.label_octal


def test_catalogue_bit_reversal_labels_are_all_present() -> None:
    labels = {target(s).get("label") for s in SAMPLES}
    assert {"203", "204", "213", "310"} <= labels


def test_forward_read_always_gives_a_different_label() -> None:
    """The discriminator: every decoded item's naive label is wrong but well formed."""
    for sample in SAMPLES:
        if "label" in target(sample):
            naive = meta(sample)["naive_label"]
            assert naive != target(sample)["label"] and len(naive) == 3, sample.id


def test_negative_values_and_fractions_are_included() -> None:
    values = [target(s)["value"] for s in by_case("bnr")]
    assert any(v < 0 for v in values)
    assert any(isinstance(v, float) for v in values)


# --- Each sample --------------------------------------------------------------


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_sample_has_prompt_word_and_target(sample: Sample) -> None:
    assert f"Word: {meta(sample)['word']}" in prompt(sample)
    assert prompt(sample).startswith(load_preamble(b1.PREAMBLE))
    assert meta(sample)["family"] == "B" and meta(sample)["subfamily"] == "B1"
    assert sample.target


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_matches_the_reference_decoder(sample: Sample) -> None:
    result = decoder.decode(meta(sample)["word"])
    answer = target(sample)
    if result.status == "rejected":
        assert answer == b1.REJECT
        return
    assert result.fields is not None
    assert answer["label"] == result.fields["label_octal"]
    assert answer["sdi"] == result.fields["sdi"]
    assert answer["ssm"] == result.fields["ssm_bits"]
    if result.status == "unknown_label":
        assert answer["value"] == "cannot_be_determined"
    else:
        assert result.decoded is not None
        value = result.decoded["value"]
        assert answer["value"] == (abs(value) if result.decoded["encoding"] == "BCD" else value)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: str(s.id))
def test_target_agrees_with_the_generator(sample: Sample) -> None:
    """The generator's own ground truth is a second, independent check."""
    word_id = meta(sample)["word_id"]
    if word_id.startswith("unknown-"):
        return
    record = next(r for r in b1.generated_words() if r["word_id"] == word_id)
    assert record["word_hex"] == meta(sample)["word"]
    if record["decoded"] is None:
        assert target(sample) == b1.REJECT
    else:
        value = record["decoded"]["value"]
        assert target(sample)["value"] == (abs(value) if record["decoded"]["encoding"] == "BCD" else value)


def test_task_text_does_not_leak_the_answer() -> None:
    for sample in SAMPLES:
        text = meta(sample)["task_text"].replace(meta(sample)["word"], "")
        answer = target(sample)
        if "label" in answer:
            assert answer["label"] not in text, sample.id
            value = str(answer["value"])
            # Short numbers such as 4 appear in the format instructions by coincidence.
            if len(value) >= 3 and value != "cannot_be_determined":
                assert value not in text, sample.id


# --- Validation (negative cases) ---------------------------------------------


def test_committed_set_passes_validation() -> None:
    b1.validate_samples(SAMPLES)


def test_naive_label_target_fails_validation() -> None:
    sample = by_case("bnr")[0]
    wrong = {**target(sample), "label": meta(sample)["naive_label"]}
    with pytest.raises(b1.B1ValidationError, match="does not match the reference decoder"):
        b1.validate_samples([with_changes(sample, target=b1.canonical(wrong))])


def test_wrong_value_target_fails_validation() -> None:
    sample = by_case("bcd")[0]
    wrong = {**target(sample), "value": target(sample)["value"] + 1}
    with pytest.raises(b1.B1ValidationError):
        b1.validate_samples([with_changes(sample, target=b1.canonical(wrong))])


def test_corrupt_word_with_a_decoded_target_fails() -> None:
    sample = by_case("parity_bit_fault")[0]
    good = by_case("bnr")[0]
    with pytest.raises(b1.B1ValidationError, match="does not match"):
        b1.validate_samples([with_changes(sample, target=good.target)])


def test_every_failing_sample_is_reported() -> None:
    bad = [with_changes(s, target=b1.canonical(b1.REJECT)) for s in by_case("bnr")[:3]]
    with pytest.raises(b1.B1ValidationError) as err:
        b1.validate_samples(bad)
    assert "3 sample(s)" in str(err.value)
    assert all(str(s.id) in str(err.value) for s in bad)


def test_word_swapped_for_another_generated_word_fails() -> None:
    sample, other = by_case("bnr")[0], by_case("bnr")[1]
    swapped = with_changes(sample, metadata={**meta(sample), "word": meta(other)["word"]},
                           input=prompt(sample).replace(meta(sample)["word"], meta(other)["word"]),
                           target=other.target)
    with pytest.raises(b1.B1ValidationError, match=f"is not word '{meta(sample)['word_id']}'"):
        b1.validate_samples([swapped])


def test_prompt_without_its_word_fails() -> None:
    sample = SAMPLES[1]
    bad = with_changes(sample, input=prompt(sample).replace(meta(sample)["word"], "<word>"))
    with pytest.raises(b1.B1ValidationError, match="prompt does not contain the word"):
        b1.validate_samples([bad])


def test_sample_without_a_word_fails() -> None:
    sample = SAMPLES[1]
    changed = {k: v for k, v in meta(sample).items() if k != "word"}
    with pytest.raises(b1.B1ValidationError, match="metadata has no word"):
        b1.validate_samples([with_changes(sample, metadata=changed)])


def test_wrong_expected_status_fails() -> None:
    sample = by_case("bnr")[0]
    bad = with_changes(sample, metadata={**meta(sample), "expected_status": "rejected"})
    with pytest.raises(b1.B1ValidationError, match="expected_status"):
        b1.validate_samples([bad])


def test_task_refuses_to_build_from_a_bad_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = b1.TASK_FILE.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    index = next(i for i, r in enumerate(records) if r["metadata"]["case"] == "bnr")
    good = json.loads(records[index]["target"])
    records[index]["target"] = b1.canonical({**good, "sdi": (good["sdi"] + 1) % 4})
    bad_file = tmp_path / "b1_bad.jsonl"
    bad_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr("tasks.b1_arinc429_word_decode.TASK_FILE", bad_file)
    monkeypatch.setattr("tasks.b1_arinc429_word_decode.load_tasks",
                        functools.partial(load_tasks, verify=False))
    with pytest.raises(b1.B1ValidationError, match=records[index]["id"]):
        b1_arinc429_word_decode()


def test_reference_answer_refuses_a_bad_bcd_digit() -> None:
    word = gen.assemble(0o012, 0, 0b1010 << (23 - 11), "00")  # a BCD digit of 10, good parity
    with pytest.raises(b1.B1ValidationError, match="does not ask about"):
        b1.reference_answer(word)


# --- End to end with inspect eval ----------------------------------------------


def scripted_model(answer_for):
    """mockllm answering each prompt with ``answer_for(word_hex)``."""
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        text = messages[-1].text
        word_hex = text.split("Word: ", 1)[1][:10]
        output = ModelOutput.from_content("mockllm/model", answer_for(word_hex))
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)  # no tokenizer download
        return output
    return get_model("mockllm/model", custom_outputs=respond)


def run(model, tmp_path: Path):
    [log] = inspect_eval(b1_arinc429_word_decode(), model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    assert log.samples is not None and len(log.samples) == len(SAMPLES)
    return log


def scores(log) -> list[float]:
    return [float(next(iter(s.scores.values())).value) for s in log.samples]


def test_correct_answers_score_one_end_to_end(tmp_path: Path) -> None:
    log = run(scripted_model(lambda w: b1.canonical(b1.reference_answer(int(w, 16)))), tmp_path)
    assert log.eval.task_version == b1.B1_VERSION
    assert scores(log) == [1.0] * len(SAMPLES)
    for sample in log.samples:
        assert sample.metadata["word"] in sample.messages[0].text
        assert sample.store["provenance"]["corpus_hash"]


def test_naive_label_reads_score_zero_end_to_end(tmp_path: Path) -> None:
    def naive(word_hex: str) -> str:
        answer = b1.reference_answer(int(word_hex, 16))
        if "label" in answer:
            answer = {**answer, "label": b1.naive_label(int(word_hex, 16))}
        return b1.canonical(answer)
    log = run(scripted_model(naive), tmp_path)
    by_id = dict(zip((str(s.id) for s in log.samples), scores(log)))
    for sample in SAMPLES:
        expected = 1.0 if target(sample) == b1.REJECT else 0.0
        assert by_id[str(sample.id)] == expected, sample.id


def test_decoding_a_corrupt_word_scores_zero_end_to_end(tmp_path: Path) -> None:
    def never_rejects(word_hex: str) -> str:
        fields = decoder.read_fields(int(word_hex, 16))
        return b1.canonical({"label": fields["label_octal"], "sdi": fields["sdi"],
                             "ssm": fields["ssm_bits"], "value": 0})
    log = run(scripted_model(never_rejects), tmp_path)
    rejected = {str(s.id) for s in SAMPLES if target(s) == b1.REJECT}
    assert all(score == 0.0 for s, score in zip(log.samples, scores(log)) if str(s.id) in rejected)


def test_mockllm_filler_runs_and_scores_zero(tmp_path: Path) -> None:
    filler = scripted_model(lambda w: "I think this word is fine.")
    assert scores(run(filler, tmp_path)) == [0.0] * len(SAMPLES)
