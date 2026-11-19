"""Task schema validator and CI corpus checks (WBS 1.28, issue #36).

Acceptance criteria (product backlog):
- When a PR adds or edits a task, the schema validator and data boundary checker run
  automatically.
- When a task fails either check, the PR is blocked.

The validator's schema is WBS 1.12's (also covered here): "When a task file conforming
to the schema is validated, it passes with no errors" and "When a task is missing its
ground truth, family or source field, the validator rejects it and names the field."

Sprint 4 backlog demo: open a PR that adds or edits a task and show the schema
validator and data boundary checker run automatically; then add a task that fails
either check and show the PR blocked.

These tests cover the validator itself, the annotations both checkers print for CI,
and the workflow file (triggers, commands, the job names branch protection requires).
Blocking the merge is the required-check setting in branch protection, outside the code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import boundary
from bench.boundary import github_annotation
from bench.corpus import add_to_manifest
from bench.loader import REQUIRED_FIELDS, load_tasks
from bench.validate import REPO_ROOT, TASK_DIR, Report, display, main, validate, validate_file

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "corpus-checks.yml"

GOOD: dict[str, Any] = {
    "id": "t-001",
    "family": "A",
    "preamble": "adsb_v1.0.0",
    "source": "spec-synthesized",
    "input": "Decode this message: 8DF00A112369A6B1C31820ABE595.",
    "target": "ZZZ101",
}


def record(**changes: Any) -> dict[str, Any]:
    """GOOD with fields changed; a value of ... removes the field."""
    out = dict(GOOD)
    for key, value in changes.items():
        if value is ...:
            out.pop(key)
        else:
            out[key] = value
    return out


def task_file(tmp_path: Path, records: list[Any], name: str = "new_task_v1.0.0.jsonl",
              listed: bool = True) -> Path:
    """Write a task file in its own directory, listed in a manifest there by default."""
    path = tmp_path / name
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records),
                    encoding="utf-8")
    if listed:
        add_to_manifest(path)
    return path


def check(path: Path) -> Report:
    report = Report()
    validate_file(path, report)
    return report


def codes(report: Report) -> list[str]:
    return [p.code for p in report.problems]


# ---------------------------------------------------------------------------
# Valid files pass (WBS 1.12 criterion 1)
# ---------------------------------------------------------------------------


def test_the_committed_corpus_passes() -> None:
    report = validate()
    assert report.passed, "\n".join(map(str, report.problems))
    assert report.files_checked == len(list(TASK_DIR.glob("*.jsonl"))) == 9
    assert report.tasks_checked == sum(len(load_tasks(f)) for f in TASK_DIR.glob("*.jsonl")) == 215


def test_a_new_valid_listed_task_file_passes(tmp_path: Path) -> None:
    path = task_file(tmp_path, [record(), record(id="t-002", premises=["one frame only"],
                                                 metadata={"subfamily": "A1"})])
    report = check(path)
    assert report.passed, report.problems
    assert report.tasks_checked == 2


def test_the_validator_accepts_exactly_what_the_loader_accepts(tmp_path: Path) -> None:
    path = task_file(tmp_path, [record()])
    assert check(path).passed
    assert len(load_tasks(path, verify=False)) == 1   # manifest checked above; tmp is outside the repo


# ---------------------------------------------------------------------------
# Schema problems are rejected and named (WBS 1.12 criterion 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_a_missing_required_field_is_rejected_and_named(tmp_path: Path, field: str) -> None:
    report = check(task_file(tmp_path, [record(**{field: ...})]))
    assert not report.passed
    assert any(p.code == "V2" and f"missing field '{field}'" in p.message and p.line == 1
               for p in report.problems), report.problems


@pytest.mark.parametrize("change, expected", [
    ({"family": "E"}, "family 'E' is not one of"),
    ({"source": ""}, "field 'source' must be a non-empty string"),
    ({"target": []}, "field 'target' must be a non-empty string or list of strings"),
    ({"premises": "not a list"}, "field 'premises' must be a non-empty list"),
    ({"metadata": ["x"]}, "field 'metadata' must be an object"),
])
def test_a_bad_value_is_rejected_and_named(tmp_path: Path, change: dict[str, Any], expected: str) -> None:
    report = check(task_file(tmp_path, [record(**change)]))
    assert [p.code for p in report.problems] == ["V2"]
    assert expected in report.problems[0].message
    assert report.problems[0].message.startswith("task 't-001': ")


def test_every_bad_record_is_reported_not_only_the_first(tmp_path: Path) -> None:
    path = task_file(tmp_path, [record(id="t-1", target=...), record(id="t-2"),
                                record(id="t-3", source=...)])
    problems = check(path).problems
    assert [(p.line, p.code) for p in problems] == [(1, "V2"), (3, "V2")]
    assert "'t-1': missing field 'target'" in problems[0].message
    assert "'t-3': missing field 'source'" in problems[1].message


def test_invalid_json_and_non_objects_are_rejected(tmp_path: Path) -> None:
    report = check(task_file(tmp_path, [record(), "{not json", "[1, 2]"]))
    assert [(p.line, p.code) for p in report.problems] == [(2, "V1"), (3, "V1")]


def test_a_duplicate_id_is_rejected(tmp_path: Path) -> None:
    report = check(task_file(tmp_path, [record(), record()]))
    assert [(p.line, p.code) for p in report.problems] == [(2, "V3")]
    assert "duplicate task id 't-001'" in report.problems[0].message


def test_an_unknown_preamble_is_rejected_and_names_the_task(tmp_path: Path) -> None:
    report = check(task_file(tmp_path, [record(preamble="adsb_v9.9.9")]))
    assert codes(report) == ["V4"]
    assert "task 't-001': preamble 'adsb_v9.9.9' does not exist" in report.problems[0].message


def test_an_empty_file_is_rejected(tmp_path: Path) -> None:
    assert codes(check(task_file(tmp_path, []))) == ["V5"]


def test_a_file_name_without_a_version_is_rejected(tmp_path: Path) -> None:
    path = task_file(tmp_path, [record()], name="new_task.jsonl", listed=False)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": {}}))
    assert "V6" in codes(check(path))


# ---------------------------------------------------------------------------
# Manifest: added and edited task files
# ---------------------------------------------------------------------------


def test_a_new_task_file_not_listed_in_the_manifest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": {}}))
    path = task_file(tmp_path, [record()], listed=False)
    report = check(path)
    assert codes(report) == ["M1"]
    assert "not listed" in report.problems[0].message
    assert "python -m bench.corpus --add" in report.problems[0].message


def test_a_frozen_task_file_edited_in_place_is_rejected(tmp_path: Path) -> None:
    frozen = TASK_DIR / "a1_adsb_decode_v1.0.0.jsonl"
    path = tmp_path / frozen.name
    shutil.copy(frozen, path)
    add_to_manifest(path)
    assert check(path).passed
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"input": "', '"input": "Please: ', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = check(path)
    assert codes(report) == ["M1"]
    assert "never edited in place" in report.problems[0].message


def test_a_manifest_entry_without_its_file_is_reported(tmp_path: Path) -> None:
    task_file(tmp_path, [record()])
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    manifest["files"]["gone_v1.0.0.jsonl"] = "0" * 64
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    report = validate([tmp_path])
    assert [p.code for p in report.problems] == ["M2"]
    assert "lists gone_v1.0.0.jsonl, which does not exist" in report.problems[0].message


# ---------------------------------------------------------------------------
# Command line and annotations
# ---------------------------------------------------------------------------


def test_cli_passes_with_exit_0_on_the_committed_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == "PASS: task schema check (9 task files, 215 tasks)"


def test_cli_fails_with_exit_1_and_annotates_the_line(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    path = task_file(tmp_path, [record(), record(id="t-002", family=...)])
    assert main(["--annotations", str(path)]) == 1
    out = capsys.readouterr().out
    assert f"{path.as_posix()}:2: [V2] task 't-002': missing field 'family'" in out
    assert f"::error file={path.as_posix().replace(':', '%3A')},line=2,title=Task schema::[V2] " in out
    assert out.strip().endswith("FAIL: 1 problem(s) in 1 file(s) (1 task files, 2 tasks)")


def test_cli_prints_no_annotations_unless_asked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(task_file(tmp_path, [record(family=...)]))]) == 1
    assert "::error" not in capsys.readouterr().out


def test_cli_exits_2_on_a_missing_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope_v1.0.0.jsonl")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_repo_paths_are_repo_relative_so_annotations_land_on_the_file() -> None:
    assert display(TASK_DIR / "loader_demo_v0.1.0.jsonl") == "data/tasks/loader_demo_v0.1.0.jsonl"


def test_annotations_escape_what_github_would_misread() -> None:
    line = github_annotation("data/tasks/a,b:c.jsonl", 3, "V2", "50% bad\nsecond line", "Task schema")
    assert line == ("::error file=data/tasks/a%2Cb%3Ac.jsonl,line=3,title=Task schema::"
                    "[V2] 50%25 bad%0Asecond line")
    assert github_annotation("MANIFEST.json", None, "M2", "x", "T") == "::error file=MANIFEST.json,title=T::[M2] x"


def test_the_boundary_checker_annotates_its_violations(tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    path = task_file(tmp_path, [record(source="unvetted-feed")])
    assert boundary.main(["--annotations", str(path)]) == 1
    out = capsys.readouterr().out
    assert ",line=1,title=Data boundary::[S2] task 't-001': unapproved source 'unvetted-feed'" in out


# ---------------------------------------------------------------------------
# The demo: a PR adds a task; each check catches what it is for
# ---------------------------------------------------------------------------


def test_an_unapproved_source_passes_the_schema_but_fails_the_boundary(tmp_path: Path) -> None:
    path = task_file(tmp_path, [record(source="unvetted-feed")])
    assert check(path).passed                                     # well formed
    assert [v.code for v in boundary.check_paths([path]).violations] == ["S2"]


def test_a_missing_source_fails_both_checks(tmp_path: Path) -> None:
    path = task_file(tmp_path, [record(source=...)])
    assert codes(check(path)) == ["V2"]
    assert [v.code for v in boundary.check_paths([path]).violations] == ["S1"]


# ---------------------------------------------------------------------------
# The workflow file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def run_step(job: dict[str, Any], command: str) -> dict[str, Any]:
    [step] = [s for s in job["steps"] if command in s.get("run", "")]
    return step


def test_it_runs_on_every_pull_request_into_development(workflow: dict[Any, Any]) -> None:
    on = workflow.get("on", workflow.get(True))   # YAML 1.1 reads the bare key `on` as True
    assert "development" in on["pull_request"]["branches"]
    # No path filter: a skipped required check never reports, so the PR would wait forever.
    assert not {"paths", "paths-ignore"} & set(on["pull_request"])


def test_the_two_required_check_names_are_schema_and_data_boundary(workflow: dict[Any, Any]) -> None:
    assert list(workflow["jobs"]) == ["schema", "data-boundary"]
    assert all("if" not in job for job in workflow["jobs"].values())


@pytest.mark.parametrize("job, command", [
    ("schema", "python -m bench.validate --annotations"),
    ("data-boundary", "python -m bench.boundary --annotations"),
])
def test_each_job_runs_its_checker_and_fails_when_it_fails(workflow: dict[Any, Any], job: str,
                                                           command: str) -> None:
    j = workflow["jobs"][job]
    step = run_step(j, command)
    assert "continue-on-error" not in step and "if" not in step
    assert "|| true" not in step["run"]
    # The output is piped to tee; only `shell: bash` (bash -eo pipefail) keeps the exit status.
    assert step["shell"] == "bash"
    assert j["timeout-minutes"] <= 30
    summary = next(s for s in j["steps"] if "GITHUB_STEP_SUMMARY" in s.get("run", ""))
    assert summary["if"] == "always()"


def test_the_schema_job_installs_the_pinned_requirements(workflow: dict[Any, Any]) -> None:
    steps = workflow["jobs"]["schema"]["steps"]
    assert any(s.get("run", "").strip() == "pip install -r requirements.txt" for s in steps)


def test_the_token_is_read_only(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}
