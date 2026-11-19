"""CI regression workflow (WBS 1.26, issue #34).

Acceptance criteria (product backlog):
- When a PR into development is opened, the workflow runs the regression set and
  posts pass/fail against the baseline.
- When any score falls below the baseline, the PR check fails and merge is blocked.

Sprint 4 backlog demo: open a PR into development and show the workflow run the
regression set and post pass/fail against the baseline; then lower a score below the
baseline and show the check fail and block the merge.

What runs on GitHub cannot be run here, so these tests cover the two halves that can:
the report and exit status that ``python -m bench.regression check --markdown``
produces (what the workflow posts and what fails the check), and the workflow file
itself (triggers, the command it runs, the job name branch protection requires).
Blocking the merge is the "regression" required check in branch protection, a repo
setting outside the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import regression as rg
from bench import scorers
from bench.regression import (
    BASELINE_VERSION,
    MAX_LISTED,
    REPORT_MARKER,
    CheckResult,
    Difference,
    check,
    load_baseline,
    main,
    markdown_report,
    run_regression,
)

WORKFLOW = rg.REPO_ROOT / ".github" / "workflows" / "regression.yml"
CHECK_COMMAND = "python -m bench.regression check"


@pytest.fixture(scope="module")
def log() -> Any:
    return run_regression()


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return load_baseline()


_extract = scorers.extract_adsb_json


def json_at_start_only(raw_text: str | None) -> dict[str, Any] | None:
    """A broken parser that only reads JSON at the start of the reply, so answers wrapped
    in prose stop parsing. The same regression as the demo's one-line scorer edit."""
    text = (raw_text or "").strip()
    return _extract(text) if text.startswith("{") else None


# ---------------------------------------------------------------------------
# The report the workflow posts
# ---------------------------------------------------------------------------


def test_a_passing_run_reports_pass_with_the_count_and_baseline(log: Any, committed: dict[str, Any]) -> None:
    result = check(log, committed)
    report = markdown_report(result, committed)
    assert report.startswith(REPORT_MARKER + "\n")
    assert "## Regression check: PASS" in report
    assert f"All 570 scores match baseline {BASELINE_VERSION}" in report
    assert "Changed scores" not in report and "Setup changed" not in report
    # The per-kind means: this run's equal the baseline's on every row.
    assert "| reference | 1.000 | 1.000 | 1.000 | 1.000 |" in report
    assert "| unparseable | 0.000 | 0.000 | 0.000 | 0.000 |" in report


def test_a_lowered_score_is_reported_as_fail_and_lower(monkeypatch: pytest.MonkeyPatch,
                                                       committed: dict[str, Any]) -> None:
    monkeypatch.setattr(scorers, "extract_adsb_json", json_at_start_only)
    result = check(baseline=committed)
    assert not result.passed
    report = markdown_report(result, committed)
    assert "## Regression check: FAIL" in report
    lowered = [d for d in result.differences if d.current is not None and d.baseline is not None
               and d.current < d.baseline]
    assert lowered and len(lowered) == len(result.differences)
    assert f"### Changed scores ({len(lowered)} lower than the baseline)" in report
    first = lowered[0]
    assert len(lowered) == 57                                   # every prose-wrapped answer
    assert all(d.case.endswith("::prose_wrapped") and d.scorer == "adsb_field_capability_scorer"
               for d in lowered)
    assert f"| `{first.case}` | field_scorer | 1 | 0 | LOWER |" in report
    # The table of means shows where it dropped (field_scorer baseline/now, then exact_match).
    assert "| prose_wrapped | 1.000 | 0.000 | 0.000 | 0.000 |" in report


def test_a_higher_score_also_fails(committed: dict[str, Any]) -> None:
    diff = Difference("a1-001::unparseable", "deterministic_exact_match", 0.0, 1.0)
    report = markdown_report(CheckResult([], [diff], 570), committed)
    assert "## Regression check: FAIL" in report
    assert "(0 lower than the baseline)" in report
    assert "| 0 | 1 | HIGHER |" in report
    assert "in either direction" in report


def test_setup_changes_missing_and_new_cases_are_listed(committed: dict[str, Any]) -> None:
    result = CheckResult(
        ["scorer deterministic_exact_match version changed: baseline 1.0.0, now 1.0.1"],
        [Difference("x::reference", "deterministic_exact_match", 1.0, None),
         Difference("y::reference", "deterministic_exact_match", None, 1.0)],
        569,
    )
    report = markdown_report(result, committed)
    assert "### Setup changed" in report
    assert "- scorer deterministic_exact_match version changed: baseline 1.0.0, now 1.0.1" in report
    assert "| `x::reference` | exact_match | 1 | - | missing |" in report
    assert "| `y::reference` | exact_match | - | 1 | not in baseline |" in report


def test_a_long_list_of_differences_is_capped() -> None:
    diffs = [Difference(f"t{i}::reference", "deterministic_exact_match", 1.0, 0.0)
             for i in range(MAX_LISTED + 7)]
    report = markdown_report(CheckResult([], diffs, 570))
    assert report.count("| LOWER |") == MAX_LISTED
    assert "...and 7 more." in report
    assert f"### Changed scores ({MAX_LISTED + 7} lower than the baseline)" in report


def test_pipes_in_a_case_id_do_not_break_the_table() -> None:
    diff = Difference("a|b::reference", "deterministic_exact_match", 1.0, 0.0)
    assert "| `a\\|b::reference` |" in markdown_report(CheckResult([], [diff], 1))


def test_an_error_is_reported_as_error() -> None:
    report = markdown_report(error="baseline_v1.0.0.json does not match MANIFEST.json")
    assert report.startswith(REPORT_MARKER)
    assert "## Regression check: ERROR" in report
    assert "does not match MANIFEST.json" in report


# ---------------------------------------------------------------------------
# The command the workflow runs: exit status and report file
# ---------------------------------------------------------------------------


def test_check_writes_the_report_and_exits_0_on_a_pass(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    assert main(["check", "--markdown", str(out)]) == 0
    assert "## Regression check: PASS" in out.read_text(encoding="utf-8")


def test_check_writes_the_report_and_exits_1_when_a_score_drops(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(scorers, "extract_adsb_json", json_at_start_only)
    out = tmp_path / "report.md"
    assert main(["check", "--markdown", str(out)]) == 1
    assert "(LOWER)" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert "## Regression check: FAIL" in text and "| LOWER |" in text


def test_check_writes_an_error_report_and_exits_2_without_a_baseline(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    monkeypatch.setattr(rg, "REGRESSION_DIR", tmp_path)   # no baseline, no manifest here
    out = tmp_path / "report.md"
    assert main(["check", "--markdown", str(out)]) == 2
    assert "## Regression check: ERROR" in out.read_text(encoding="utf-8")


def test_check_without_markdown_writes_no_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["check"]) == 0
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# The workflow file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


def steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    return workflow["jobs"]["regression"]["steps"]


def test_it_runs_on_pull_requests_into_development(workflow: dict[Any, Any]) -> None:
    on = triggers(workflow)
    assert "development" in on["pull_request"]["branches"]
    # No path filter: a required check that is skipped never reports, and the PR waits forever.
    assert "paths" not in on["pull_request"] and "paths-ignore" not in on["pull_request"]


def test_the_job_name_branch_protection_requires_is_regression(workflow: dict[Any, Any]) -> None:
    assert list(workflow["jobs"]) == ["regression"]
    job = workflow["jobs"]["regression"]
    assert "if" not in job                                 # never skipped
    assert job["timeout-minutes"] <= 30


def test_it_runs_the_regression_check_and_lets_its_exit_status_fail_the_job(workflow: dict[Any, Any]) -> None:
    [run] = [s for s in steps(workflow) if CHECK_COMMAND in s.get("run", "")]
    assert "--markdown regression-report.md" in run["run"]
    assert "continue-on-error" not in run and "if" not in run
    assert "|| true" not in run["run"] and "set +e" not in run["run"]


def test_it_installs_the_pinned_requirements(workflow: dict[Any, Any]) -> None:
    assert any(s.get("run", "").strip() == "pip install -r requirements.txt" for s in steps(workflow))


def test_the_report_is_posted_even_when_the_check_fails(workflow: dict[Any, Any]) -> None:
    summary = next(s for s in steps(workflow) if "GITHUB_STEP_SUMMARY" in s.get("run", ""))
    assert summary["if"] == "always()"
    assert "regression-report.md" in summary["run"]
    comment = next(s for s in steps(workflow) if str(s.get("uses", "")).startswith("actions/github-script"))
    assert comment["if"].startswith("always()") and "pull_request" in comment["if"]
    assert REPORT_MARKER in comment["with"]["script"]       # updates its own comment, not a new one per push
    assert comment["continue-on-error"] is True             # a comment failure never changes the verdict


def test_the_token_can_comment_and_nothing_else(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "write"}
