"""Guardrail profiles as a run-config field (WBS 1.17, issue #25).

Demo (issue): run the same task under two guardrail profiles selected in the run
config and show the chosen profile recorded in the EvalLog.

Inferred acceptance criteria:
- When the run config names a profile, the run starts, the profile text is the system
  message, and its name, version and sha256 are in the EvalLog.
- When the config names an unknown profile, or a profile file was edited without a
  version bump, the run is refused before any model call and no log is written.
"""

import copy
import shutil
from pathlib import Path

import pytest
import yaml
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

import runner.prompts
from runner.guardrails import (
    GuardrailProfileError,
    load_guardrail_profile,
)
from runner.preflight import load_and_check, main, run_metadata, task_args

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_TEXT = (REPO_ROOT / "prompts" / "guardrail_hardened_example.v1.txt").read_text()

BASE = {
    "task": "tasks/guardrail_example.py",
    "model": "mockllm/model",
    "generate_config": {"temperature": 0, "seed": 42},
    "metadata": {"corpus_version": "synthetic-0.1.0"},
}


def with_profile(profile: object) -> dict:
    cfg = copy.deepcopy(BASE)
    if profile is not None:
        cfg["guardrail_profile"] = profile
    return cfg


def run(tmp_path: Path, cfg: dict, monkeypatch: pytest.MonkeyPatch, name: str = "run") -> int:
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return main([str(path)])


def logs(tmp_path: Path) -> list[EvalLog]:
    return [read_eval_log(info) for info in list_eval_logs(str(tmp_path / "logs"))]


def messages(log: EvalLog) -> list[tuple[str, str]]:
    assert log.samples
    return [(m.role, m.text) for m in log.samples[0].messages if m.role in ("system", "user")]


# --- When the config names a profile, it is applied and recorded -------------


def test_selected_profile_is_applied_and_recorded(tmp_path, monkeypatch):
    cfg = with_profile({"name": "hardened_example", "version": 1})
    assert run(tmp_path, cfg, monkeypatch) == 0
    [log] = logs(tmp_path)
    meta = log.eval.metadata
    assert meta["guardrail_profile_name"] == "hardened_example"
    assert meta["guardrail_profile_version"] == 1
    assert meta["guardrail_profile_sha256"] == load_guardrail_profile("hardened_example").sha256
    assert meta["corpus_version"] == "synthetic-0.1.0"  # the config's own metadata survives
    assert ("system", PROFILE_TEXT) in messages(log)


def test_two_profiles_give_two_distinct_logs_with_the_same_task_text(tmp_path, monkeypatch):
    assert run(tmp_path, with_profile({"name": "none"}), monkeypatch, "baseline") == 0
    assert run(tmp_path, with_profile({"name": "hardened_example"}), monkeypatch, "hardened") == 0
    by_name = {log.eval.metadata["guardrail_profile_name"]: log for log in logs(tmp_path)}
    assert set(by_name) == {"none", "hardened_example"}
    baseline, hardened = by_name["none"], by_name["hardened_example"]
    assert baseline.eval.metadata["guardrail_profile_sha256"] is None
    assert not [m for m in messages(baseline) if m[0] == "system"]
    assert [m for m in messages(hardened) if m[0] == "system"]
    # Only the system message differs: the task text is identical (rule 3).
    assert [m for m in messages(baseline) if m[0] == "user"] == [
        m for m in messages(hardened) if m[0] == "user"]


def test_config_without_the_field_records_none(tmp_path, monkeypatch):
    cfg = with_profile(None)
    assert run(tmp_path, cfg, monkeypatch) == 0
    [log] = logs(tmp_path)
    assert log.eval.metadata["guardrail_profile_name"] == "none"
    assert not [m for m in messages(log) if m[0] == "system"]


def test_shipped_example_config_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    assert main(["configs/guardrail_example.yaml"]) == 0
    [log] = logs(tmp_path)
    assert log.eval.metadata["guardrail_profile_name"] == "hardened_example"


# --- Negative cases: refused before any model call, no log written -----------


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ({"name": "no_such_profile"}, "no_such_profile"),
        ({"name": "hardened_example", "version": 99}, "hardened_example"),
        ({"name": "hardened_example", "version": "one"}, "version"),
        ({"version": 1}, "name"),
        ({"name": "hardened_example", "level": 3}, "level"),
        ("hardened_example", "mapping"),
    ],
)
def test_bad_profile_is_refused_before_any_model_call(
        profile, expected, tmp_path, monkeypatch, capsys):
    assert run(tmp_path, with_profile(profile), monkeypatch) == 2
    assert expected in capsys.readouterr().err
    assert logs(tmp_path) == []


def test_edited_profile_file_is_refused(tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "prompts"
    shutil.copytree(REPO_ROOT / "prompts", prompts)
    edited = prompts / "guardrail_hardened_example.v1.txt"
    edited.write_text(edited.read_text() + "Also be extra careful.\n")
    monkeypatch.setattr(runner.prompts, "PROMPTS_DIR", prompts)
    assert run(tmp_path, with_profile({"name": "hardened_example"}), monkeypatch) == 2
    err = capsys.readouterr().err
    assert "hardened_example" in err and "version bump" in err
    assert logs(tmp_path) == []


# --- Units --------------------------------------------------------------------


def test_none_is_a_profile_with_no_text():
    profile = load_guardrail_profile("none")
    assert profile.pre_solvers() == []
    assert profile.metadata() == {
        "guardrail_profile_name": "none",
        "guardrail_profile_version": 0,
        "guardrail_profile_sha256": None,
    }


def test_text_profile_adds_one_system_message():
    profile = load_guardrail_profile("hardened_example")
    assert len(profile.pre_solvers()) == 1
    assert profile.text == PROFILE_TEXT


def test_unknown_profile_error_names_it():
    with pytest.raises(GuardrailProfileError, match="no_such_profile"):
        load_guardrail_profile("no_such_profile")


def test_guardrail_metadata_keys_do_not_collide_with_prompt_template_keys():
    keys = set(load_guardrail_profile("hardened_example").metadata())
    assert not any(k.startswith("prompt_template") for k in keys)


def test_task_args_are_passed_only_when_the_config_selects_a_profile(tmp_path):
    assert task_args(with_profile(None)) == {}
    assert task_args(with_profile({"name": "hardened_example"})) == {
        "guardrail_profile": "hardened_example", "guardrail_version": 1}


def test_run_metadata_includes_the_profile(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(with_profile({"name": "hardened_example"})))
    meta = run_metadata(load_and_check(path))
    assert meta["guardrail_profile_name"] == "hardened_example"
    assert meta["corpus_version"] == "synthetic-0.1.0"
