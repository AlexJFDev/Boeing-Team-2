"""Inter-rater agreement pilot (WBS 3.9, issue #64).

Acceptance criteria:

1. When two team members grade the same sample of answers, agreement is computed and
   recorded.
2. (negative) When agreement is below the agreed threshold, the rubric is revised before
   the corpus freeze: the pilot writes REVISION.md and the freeze gate stays closed
   until a pilot on the revised rubric passes.

The answers come from real Inspect runs of the A1, A2 and B4 tasks on scripted stand-in
models (bench/irr_scripted.py); the grades are scripted too.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, ModelUsage, get_model
from inspect_ai.tool import ToolChoice, ToolInfo

from bench import irr
from bench.irr_scripted import standin_grades, standin_model

JUDGE_CLAUSES = {"a1.identity_overtrusted", "a1.source_misattributed", "a2.position_overtrusted",
                 "b4.fault_overattributed", "b4.attack_without_basis"}
TASKS = ["tasks/a1_adsb_decode.py", "tasks/a2_cpr_position.py", "tasks/b4_arinc429_fault_vs_injection.py"]


def _blank_model():
    def outputs(messages: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                config: GenerateConfig) -> ModelOutput:
        out = ModelOutput.from_content("mockllm/standin-blank", "")
        out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return out
    return get_model("mockllm/standin-blank", memoize=False, custom_outputs=outputs)


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("irr_logs")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(Path(__file__).resolve().parents[1])  # Inspect takes task paths relative to the cwd
        logs = _run(out)
    assert all(lg.status == "success" for lg in logs)
    return out


def _run(out: Path) -> list:
    logs = inspect_eval(TASKS,
                        model=[standin_model("standin-x"), standin_model("standin-y")],
                        log_dir=str(out), display="none", seed=42, temperature=0.0)
    logs += inspect_eval(TASKS[2], model=_blank_model(), log_dir=str(out), display="none",
                         seed=42, temperature=0.0)
    return logs


@pytest.fixture
def pilot(log_dir: Path, tmp_path: Path) -> Path:
    assert irr.main(["sample", str(log_dir), "--pilot", "p1", "--seed", "7", "--irr-dir", str(tmp_path)]) == 0
    return tmp_path / "p1"


def _write(pilot_dir: Path, rater: str, grades: dict[str, tuple[str, str]]) -> None:
    _, key = irr.read_packet(pilot_dir)
    irr._write_grades(irr.grades_path(pilot_dir, rater), list(key["items"]), grades)


# ---------------------------------------------------------------------------
# The rubric and the sample
# ---------------------------------------------------------------------------


def test_rubric_is_every_judge_clause_and_fingerprint_tracks_its_text(tmp_path: Path) -> None:
    rubric = irr.current_rubric()
    assert {k[2] for k in rubric} == JUDGE_CLAUSES
    assert all(item.text.startswith("Unsafe if ") for item in rubric.values())
    assert irr.rubric_fingerprint(rubric) == irr.rubric_fingerprint()

    edited = tmp_path / irr.TASK_FILES["b4_arinc429_fault_vs_injection"].name
    rows = [json.loads(x) for x in irr.TASK_FILES["b4_arinc429_fault_vs_injection"].read_text().splitlines()]
    for row in rows:
        for clause in row["metadata"]["unsafe_if"]:
            if clause["id"] == "b4.attack_without_basis":
                clause["text"] += " Monitoring advice alone is not a claim of attack."
    edited.write_text("".join(json.dumps(r) + "\n" for r in rows))
    revised = irr.current_rubric({**irr.TASK_FILES, "b4_arinc429_fault_vs_injection": edited})
    assert irr.rubric_fingerprint(revised) != irr.rubric_fingerprint(rubric)


def test_sample_is_blind_stratified_seeded_and_skips_blank_answers(log_dir: Path, tmp_path: Path) -> None:
    units = irr.collect_units(log_dir)
    assert {u.model for u in units} == {"mockllm/standin-x", "mockllm/standin-y"}  # blank answers skipped
    assert len(units) == 2 * len(irr.current_rubric())

    assert irr.main(["sample", str(log_dir), "--pilot", "a", "--seed", "11", "--irr-dir", str(tmp_path)]) == 0
    assert irr.main(["sample", str(log_dir), "--pilot", "b", "--seed", "11", "--irr-dir", str(tmp_path)]) == 0
    assert irr.main(["sample", str(log_dir), "--pilot", "c", "--seed", "12", "--irr-dir", str(tmp_path)]) == 0
    packet_a = (tmp_path / "a" / irr.PACKET).read_text()
    assert packet_a == (tmp_path / "b" / irr.PACKET).read_text()
    assert packet_a != (tmp_path / "c" / irr.PACKET).read_text()

    items, key = irr.read_packet(tmp_path / "a")
    assert len(items) == irr.DEFAULT_ITEMS == key["n_items"]
    assert {i["clause_id"] for i in items} == JUDGE_CLAUSES
    assert "standin" not in packet_a and "mockllm" not in packet_a  # no model names in what raters see
    assert set(items[0]) == {"item_id", "clause_id", "clause", "prompt", "response"}
    assert {v["model"] for v in key["items"].values()} <= {"mockllm/standin-x", "mockllm/standin-y"}
    assert key["seed"] == 11 and key["rubric_fingerprint"] == irr.rubric_fingerprint()
    with (tmp_path / "a" / irr.GRADES_DIR / "TEMPLATE.csv").open() as handle:
        assert [r["item_id"] for r in csv.DictReader(handle)] == [i["item_id"] for i in items]


def test_sample_refuses_a_stale_log_too_few_items_and_overwriting(log_dir: Path, tmp_path: Path, pilot: Path) -> None:
    rubric = irr.current_rubric()
    key = next(k for k in rubric if k[0] == "b4_arinc429_fault_vs_injection")
    stale = {**rubric, key: irr.RubricItem(**{**rubric[key].__dict__, "prompt": "a different question"})}
    with pytest.raises(irr.PilotError, match="no longer matches"):
        irr.collect_units(log_dir, stale)
    assert irr.main(["sample", str(log_dir), "--pilot", "x", "--seed", "1", "-n", "5",
                     "--irr-dir", str(tmp_path)]) == 2
    assert irr.main(["sample", str(log_dir), "--pilot", "p1", "--seed", "1", "--irr-dir", str(tmp_path)]) == 2
    assert irr.main(["sample", str(tmp_path / "empty"), "--pilot", "y", "--seed", "1",
                     "--irr-dir", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# Acceptance 1: two raters, agreement computed and recorded
# ---------------------------------------------------------------------------


def test_alpha_is_inspects_krippendorff_and_matches_a_hand_count() -> None:
    # 8 values: a x3, b x5. D_o = 2/8, D_e = (64 - 9 - 25) / 56, alpha = 1 - D_o/D_e = 8/15.
    assert irr.alpha([("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]) == pytest.approx(8 / 15)
    assert irr.alpha([("safe", "safe"), ("unsafe", "unsafe")] * 5) == 1.0


def test_two_raters_agreement_is_computed_and_recorded(pilot: Path, tmp_path: Path) -> None:
    _, key = irr.read_packet(pilot)
    slip = sorted(key["items"])[0]
    _write(pilot, "ss", standin_grades(key, "strict"))
    _write(pilot, "kjg", standin_grades(key, "strict", frozenset({slip})))

    assert irr.main(["score", str(pilot), "--raters", "ss", "kjg"]) == 0
    record = json.loads((pilot / irr.RECORD).read_text())
    assert record["status"] == "pass" and record["alpha"] >= irr.ALPHA_THRESHOLD
    assert record["items"] == 30 and record["agree"] == 29 and record["raters"] == ["ss", "kjg"]
    assert record["threshold"] == irr.ALPHA_THRESHOLD
    assert set(record["per_clause"]) == JUDGE_CLAUSES
    assert [d["item_id"] for d in record["disagreements"]] == [slip]
    assert record["packet_sha256"] == key["packet_sha256"]
    assert set(record["grades_sha256"]) == {"ss", "kjg"}
    assert record["rubric_fingerprint"] == irr.rubric_fingerprint()
    assert "PASS" in (pilot / irr.RECORD_MD).read_text()
    assert not (pilot / irr.REVISION).exists()
    ok, reason = irr.gate(tmp_path)
    assert ok, reason


def test_grades_must_cover_every_item_once_with_a_valid_grade(pilot: Path) -> None:
    _, key = irr.read_packet(pilot)
    grades = standin_grades(key, "strict")
    _write(pilot, "ss", grades)
    first = next(iter(grades))
    _write(pilot, "kjg", {k: v for k, v in grades.items() if k != first})
    assert irr.main(["score", str(pilot), "--raters", "ss", "kjg"]) == 2  # one item missing
    assert irr.main(["score", str(pilot), "--raters", "ss", "ss"]) == 2  # one rater twice
    assert irr.main(["score", str(pilot), "--raters", "ss", "nobody"]) == 2  # no file

    path = irr.grades_path(pilot, "kjg")
    path.write_text(path.read_text() + f"{first},maybe,\n")
    with pytest.raises(irr.PilotError, match="'maybe' is not one of safe, unsafe"):
        irr.read_grades(path, list(key["items"]))
    second = list(grades)[1]
    path.write_text(path.read_text().replace(f"{first},maybe,\n", f"{second},safe,\n"))
    with pytest.raises(irr.PilotError, match="graded twice"):
        irr.read_grades(path, list(key["items"]))
    assert not (pilot / irr.RECORD).exists()


def test_an_edited_packet_is_refused(pilot: Path) -> None:
    packet = pilot / irr.PACKET
    packet.write_text(packet.read_text().replace("Unsafe if", "Unsafe when", 1))
    with pytest.raises(irr.PilotError, match="edited after sampling"):
        irr.read_packet(pilot)


def test_grading_in_the_terminal_saves_each_answer_and_resumes(pilot: Path) -> None:
    answers = iter(["s", "", "x", "u", "too vague", "q"])
    left = irr.grade_interactive(pilot, "ss", ask=lambda _: next(answers), say=lambda _: None)
    assert left == 28
    saved = irr.read_grades(irr.grades_path(pilot, "ss"), list(irr.read_packet(pilot)[1]["items"]), complete=False)
    assert [g for g, _ in saved.values()] == ["safe", "unsafe"]
    assert list(saved.values())[1][1] == "too vague"

    rest = iter(["s", ""] * 28)
    assert irr.grade_interactive(pilot, "ss", ask=lambda _: next(rest), say=lambda _: None) == 0
    assert len(irr.read_grades(irr.grades_path(pilot, "ss"), list(irr.read_packet(pilot)[1]["items"]))) == 30
    with pytest.raises(irr.PilotError, match="rater"):
        irr.grade_interactive(pilot, "Sahiti S", ask=lambda _: "q", say=lambda _: None)


# ---------------------------------------------------------------------------
# Acceptance 2 (negative): below threshold sends the rubric back and blocks the freeze
# ---------------------------------------------------------------------------


def test_below_threshold_sends_the_rubric_back_for_revision(pilot: Path, tmp_path: Path) -> None:
    _, key = irr.read_packet(pilot)
    _write(pilot, "ss", standin_grades(key, "strict"))
    _write(pilot, "kjg", standin_grades(key, "lenient"))  # split on every borderline answer

    assert irr.main(["score", str(pilot), "--raters", "ss", "kjg"]) == 1
    record = json.loads((pilot / irr.RECORD).read_text())
    assert record["status"] == "revise_rubric" and record["alpha"] < irr.ALPHA_THRESHOLD
    assert record["disagreements"]  # recorded even when it fails
    revision = (pilot / irr.REVISION).read_text()
    assert "Rubric revision needed" in revision and "UNSAFE_IF_VERSION" in revision
    for d in record["disagreements"]:
        assert f"`{d['item_id']}`" in revision and f"`{d['clause_id']}`" in revision
    assert "BELOW THRESHOLD" in (pilot / irr.RECORD_MD).read_text()

    ok, reason = irr.gate(tmp_path)
    assert not ok and "below threshold" in reason and str(pilot / irr.REVISION) in reason
    assert irr.main(["gate", "--irr-dir", str(tmp_path)]) == 1


def test_gate_needs_a_passing_pilot_on_the_current_rubric(pilot: Path, tmp_path: Path) -> None:
    ok, reason = irr.gate(tmp_path)
    assert not ok and "no inter-rater pilot" in reason

    _, key = irr.read_packet(pilot)
    _write(pilot, "ss", standin_grades(key, "strict"))
    _write(pilot, "kjg", standin_grades(key, "strict"))
    irr.score_pilot(pilot, ["ss", "kjg"])
    assert irr.gate(tmp_path)[0]

    # After a revision the rubric text changes, so the old pass no longer counts.
    ok, reason = irr.gate(tmp_path, fingerprint="0" * 64)
    assert not ok and "earlier version of the rubric" in reason

    # A newer failing pilot on the same rubric closes the gate again.
    redo = tmp_path / "p2"
    shutil.copytree(pilot, redo)
    _write(redo, "kjg", standin_grades(key, "lenient"))
    irr.score_pilot(redo, ["ss", "kjg"])
    record = json.loads((redo / irr.RECORD).read_text())
    record["scored_at"] = "9999-01-01T00:00:00+00:00"
    (redo / irr.RECORD).write_text(json.dumps(record))
    ok, reason = irr.gate(tmp_path)
    assert not ok and "p2" in reason


def test_a_below_threshold_record_cannot_be_relabelled_to_pass_with_a_lower_threshold(pilot: Path, tmp_path: Path) -> None:
    _, key = irr.read_packet(pilot)
    _write(pilot, "ss", standin_grades(key, "strict"))
    _write(pilot, "kjg", standin_grades(key, "strict"))
    irr.score_pilot(pilot, ["ss", "kjg"])
    record = json.loads((pilot / irr.RECORD).read_text())
    record["threshold"] = 0.5
    (pilot / irr.RECORD).write_text(json.dumps(record))
    assert not irr.gate(tmp_path)[0]
