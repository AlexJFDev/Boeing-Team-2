"""Regression task set and baseline scores (WBS 1.25, issue #33).

Acceptance criteria (product backlog):
- When the regression set is run with the reference config, its scores are saved as
  the baseline file.
- When known-correct reference answers are fed to the scorers, they reproduce the
  baseline exactly; any difference fails.

Sprint 3 backlog demo, also covered:
- Run the regression set against the frozen corpus and show the recorded baseline
  scores, then show a changed result flagged against that baseline.

No model and no Docker: a replay solver supplies the canned responses. One regression
run is shared by the module (about 5 s).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from bench import regression as rg
from bench import scorers
from bench.corpus import CorpusError, add_to_manifest
from bench.loader import load_tasks
from bench.regression import (
    BASELINE_VERSION,
    RESPONSE_KINDS,
    SCORERS,
    RegressionError,
    baseline_path,
    build_baseline,
    check,
    load_baseline,
    main,
    response_for,
    run_regression,
    scores_from_log,
    score_differences,
    setup_differences,
    write_baseline,
)

N_TASKS = sum(len(load_tasks(f)) for f in rg.TASK_FILES)


@pytest.fixture(scope="module")
def log() -> Any:
    return run_regression()


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return load_baseline()


# ---------------------------------------------------------------------------
# The regression set
# ---------------------------------------------------------------------------


def test_every_a1_and_a2_task_is_in_the_set_with_every_response_kind() -> None:
    samples = rg.regression_samples()
    assert N_TASKS == 57
    assert len(samples) == N_TASKS * len(RESPONSE_KINDS)
    assert len({s.id for s in samples}) == len(samples)
    assert {s.metadata["regression_kind"] for s in samples} == set(RESPONSE_KINDS)


DECODE = '{"callsign":"ZZZ101","category":3,"df":17,"icao":"F00A11","message_type":"identification","type_code":4}'
POSITION = '{"latitude":38.9126,"longitude":-71.40106}'
REJECT = '{"answer":"reject","reason":"crc_check_failed"}'
UNDETERMINED = '{"answer":"cannot_be_determined"}'


@pytest.mark.parametrize("target", [DECODE, POSITION, REJECT, UNDETERMINED])
def test_responses_are_derived_from_the_target(target: str) -> None:
    obj = json.loads(target)
    assert response_for(target, "reference") == target
    assert json.loads(response_for(target, "reformatted")) == obj
    assert "\n" in response_for(target, "reformatted")
    assert target in response_for(target, "prose_wrapped")
    assert "{" not in response_for(target, "unparseable")
    assert all(response_for(target, k) == response_for(target, k) for k in RESPONSE_KINDS)


def test_one_field_wrong_changes_exactly_one_field() -> None:
    for target in (DECODE, POSITION):
        obj, wrong = json.loads(target), json.loads(response_for(target, "one_field_wrong"))
        assert wrong.keys() == obj.keys()
        assert sum(obj[k] != wrong[k] for k in obj) == 1
    assert json.loads(response_for(REJECT, "one_field_wrong"))["answer"] == "accept"
    # "cannot be determined" answered with a confident position: the overconfidence case
    assert json.loads(response_for(UNDETERMINED, "one_field_wrong")) == rg.WRONG_POSITION


def test_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown response kind"):
        response_for(DECODE, "shouted")


def test_task_file_wrapper_matches_the_module() -> None:
    from tasks.regression_set import regression_set

    task = regression_set()
    assert len(task.dataset) == N_TASKS * len(RESPONSE_KINDS)
    assert task.version == rg.REGRESSION_VERSION


# ---------------------------------------------------------------------------
# Acceptance criterion 1: the run's scores are saved as the baseline file
# ---------------------------------------------------------------------------


def test_run_scores_every_case_with_every_scorer(log: Any) -> None:
    cases = scores_from_log(log)
    assert len(cases) == N_TASKS * len(RESPONSE_KINDS)
    assert all(set(scores) == set(SCORERS) for scores in cases.values())
    assert log.eval.model == rg.REFERENCE_CONFIG["model"]
    assert log.plan.config.temperature == 0 and log.plan.config.seed == 42


def test_record_saves_the_baseline_and_lists_it(log: Any, tmp_path: Path) -> None:
    path = tmp_path / "baseline_v9.9.9.json"
    sha = write_baseline(build_baseline(log, "9.9.9"), path)
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["files"] == {path.name: sha}
    saved = json.loads(path.read_text())
    assert saved["baseline_version"] == "9.9.9"
    assert saved["cases"] == scores_from_log(log)
    assert saved["reference_config"] == {"model": "mockllm/model", "temperature": 0, "seed": 42}
    assert saved["scorers"] == {name: version for name, (_, version) in SCORERS.items()}
    assert set(saved["corpus"]) == {"data/tasks/a1_adsb_decode_v1.0.0.jsonl",
                                    "data/tasks/a2_cpr_position_v1.0.0.jsonl"}


def test_recording_is_deterministic(log: Any) -> None:
    second = run_regression()
    assert build_baseline(log, "1.0.0") == build_baseline(second, "1.0.0")


def test_an_existing_baseline_is_never_overwritten(log: Any, tmp_path: Path) -> None:
    path = tmp_path / "baseline_v1.0.0.json"
    write_baseline(build_baseline(log, "1.0.0"), path)
    with pytest.raises(RegressionError, match="already exists; a new baseline is a new version"):
        write_baseline(build_baseline(log, "1.0.0"), path)


def test_the_committed_baseline_is_listed_and_unedited() -> None:
    baseline = load_baseline()
    assert baseline_path().name == f"baseline_v{BASELINE_VERSION}.json"
    assert baseline["baseline_version"] == BASELINE_VERSION


def test_an_edited_baseline_is_refused(tmp_path: Path) -> None:
    path = tmp_path / baseline_path().name
    shutil.copy(baseline_path(), path)
    add_to_manifest(path)
    path.write_text(path.read_text().replace('"deterministic_exact_match": 1.0', '"deterministic_exact_match": 0.5', 1))
    with pytest.raises(RegressionError, match="never edited in place"):
        load_baseline(path)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: reference answers reproduce the baseline exactly
# ---------------------------------------------------------------------------


def test_reference_answers_score_full_marks(committed: dict[str, Any]) -> None:
    refs = {cid: s for cid, s in committed["cases"].items() if cid.endswith("::reference")}
    assert len(refs) == N_TASKS
    assert all(value == 1.0 for scores in refs.values() for value in scores.values())


def test_a_fresh_run_reproduces_the_committed_baseline_exactly(log: Any, committed: dict[str, Any]) -> None:
    result = check(log, committed)
    assert result.passed, "\n".join(result.setup_problems + [str(d) for d in result.differences])
    assert result.compared == N_TASKS * len(RESPONSE_KINDS) * len(SCORERS)


def test_baseline_characterizes_the_scorers(committed: dict[str, Any]) -> None:
    """What the baseline says about the scorers, so a change to it is a decision."""
    s = committed["summary"]
    exact, field = s["deterministic_exact_match"], s["adsb_field_capability_scorer"]
    assert exact["reference"]["mean"] == field["reference"]["mean"] == 1.0
    assert exact["unparseable"]["mean"] == field["unparseable"]["mean"] == 0.0
    # Exact match is format-sensitive: a correct answer pretty-printed or in prose scores 0.
    assert exact["reformatted"]["mean"] == exact["prose_wrapped"]["mean"] == 0.0
    assert field["reformatted"]["mean"] == field["prose_wrapped"]["mean"] == 1.0
    assert 0.0 < field["one_field_wrong"]["mean"] < 1.0   # partial credit
    assert exact["one_field_wrong"]["mean"] == 0.0


def scores_copy(committed: dict[str, Any]) -> dict[str, dict[str, float]]:
    return copy.deepcopy(committed["cases"])


@pytest.mark.parametrize("change, expected", [
    ("lower", "0.5 -> 0 (LOWER)"),
    ("higher", "0.5 -> 1 (HIGHER)"),
])
def test_any_changed_score_fails(committed: dict[str, Any], change: str, expected: str) -> None:
    current = scores_copy(committed)
    cid = "a2-001-global-pair::one_field_wrong"
    current[cid]["adsb_field_capability_scorer"] = 0.0 if change == "lower" else 1.0
    [diff] = score_differences(committed["cases"], current)
    assert (diff.case, diff.scorer) == (cid, "adsb_field_capability_scorer")
    assert str(diff).endswith(expected)


def test_missing_and_extra_cases_fail(committed: dict[str, Any]) -> None:
    current = scores_copy(committed)
    del current["a1-adsb-0001::reference"]
    current["a9-new::reference"] = {"deterministic_exact_match": 1.0}
    found = {str(d) for d in score_differences(committed["cases"], current)}
    assert "a1-adsb-0001::reference [deterministic_exact_match]: missing (baseline 1)" in found
    assert "a9-new::reference [deterministic_exact_match]: not in the baseline (now 1)" in found


def test_tiny_float_differences_still_fail(committed: dict[str, Any]) -> None:
    current = scores_copy(committed)
    cid = "a1-adsb-0001::one_field_wrong"
    current[cid]["adsb_field_capability_scorer"] += 1e-12
    assert len(score_differences(committed["cases"], current)) == 1


@pytest.mark.parametrize("key, value, message", [
    ("scorers", {"deterministic_exact_match": "0.9.0", "adsb_field_capability_scorer": "2.0.0"},
     "scorer deterministic_exact_match version changed: baseline 0.9.0, now 1.0.0"),
    ("corpus", {"data/tasks/a1_adsb_decode_v1.0.0.jsonl": "0" * 64,
                "data/tasks/a2_cpr_position_v1.0.0.jsonl": "0" * 64},
     "corpus of data/tasks/a1_adsb_decode_v1.0.0.jsonl changed"),
    ("regression_version", "0.0.1", "regression_version changed"),
    ("reference_config", {"model": "ollama/llama3.1", "temperature": 0, "seed": 42},
     "reference_config changed"),
])
def test_setup_changes_fail(committed: dict[str, Any], key: str, value: Any, message: str) -> None:
    problems = setup_differences({**committed, key: value}, rg.current_setup())
    assert any(message in p for p in problems), problems


def test_a_setup_change_alone_fails_the_check(log: Any, committed: dict[str, Any]) -> None:
    """Same scores, but the scorer version was bumped: the baseline must be refreshed (1.29)."""
    bumped = {**committed, "scorers": {**committed["scorers"], "deterministic_exact_match": "0.9.0"}}
    result = check(log, bumped)
    assert result.differences == [] and len(result.setup_problems) == 1
    assert not result.passed


# ---------------------------------------------------------------------------
# Sprint demo: a changed result is flagged
# ---------------------------------------------------------------------------

_strict_field_matches = scorers.field_matches


def loose_field_matches(actual: Any, expected: Any, key: str) -> bool:
    """field_matches with floats compared within 0.05 instead of exactly."""
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 0.05
        except (TypeError, ValueError):
            return False
    return _strict_field_matches(actual, expected, key)



def test_a_silently_loosened_scorer_is_flagged(committed: dict[str, Any],
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A scorer change without a version bump: numeric fields now match within 0.05."""
    monkeypatch.setattr(scorers, "field_matches", loose_field_matches)
    result = check(run_regression(), committed)
    assert not result.passed and result.setup_problems == []
    assert len(result.differences) == 20     # every A2 position answered 0.01 deg off
    assert all(d.case.endswith("::one_field_wrong") and d.current == 1.0 and d.baseline == 0.5
               for d in result.differences)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_show_prints_the_baseline_scores(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show"]) == 0
    out = capsys.readouterr().out
    assert "baseline data/regression/baseline_v1.0.0.json" in out
    assert all(kind in out for kind in RESPONSE_KINDS)
    assert "285 cases, 570 scores" in out


def test_check_passes_and_reads_an_existing_log(log: Any, capsys: pytest.CaptureFixture[str],
                                                tmp_path: Path) -> None:
    assert main(["check", "--log-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().endswith("PASS: all 570 scores match baseline 1.0.0")
    [eval_file] = tmp_path.glob("*.eval")
    assert main(["check", "--log", str(eval_file)]) == 0


def test_check_fails_with_exit_1_on_a_difference(monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(scorers, "field_matches", loose_field_matches)
    assert main(["check"]) == 1
    out = capsys.readouterr().out
    assert "CHANGED: a2-001-global-pair::one_field_wrong [adsb_field_capability_scorer]: 0.5 -> 1 (HIGHER)" in out
    assert out.strip().endswith("FAIL: 20 score(s) and 0 setup item(s) differ from baseline 1.0.0")


def test_check_fails_with_exit_2_on_an_edited_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(rg, "REGRESSION_DIR", tmp_path)
    path = tmp_path / baseline_path().name
    shutil.copy(rg.REPO_ROOT / "data" / "regression" / path.name, path)
    add_to_manifest(path)
    path.write_text(path.read_text() + " ")
    assert main(["show"]) == 2
    assert "never edited in place" in capsys.readouterr().err


def test_record_refuses_the_existing_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["record", BASELINE_VERSION]) == 2
    assert "already exists" in capsys.readouterr().err
