"""The 'bench run' CLI entry point (WBS 1.19, issue #27).

Demo (issue): run the smoke test through the ``bench run`` command from a run config,
then remove a required field and show the command refuse to start with an error
naming it.

Inferred acceptance criteria:
- When ``bench run`` is given a complete run config, the run starts, finishes and
  writes a log.
- When a required field is missing, the command exits with code 2, names the field
  and writes no log.
"""

import copy
import subprocess
import sys

import pytest
import yaml
from inspect_ai.log import list_eval_logs, read_eval_log

from runner.cli import main

COMPLETE = {
    "task": "tasks/smoke_test.py",
    "model": "mockllm/model",
    "generate_config": {"temperature": 0, "seed": 42},
    "metadata": {"corpus_version": "synthetic-0.1.0"},
}


def run(tmp_path, cfg, monkeypatch):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return main(["run", str(path)])


def logs(tmp_path):
    return list_eval_logs(str(tmp_path / "logs"))


def test_run_executes_the_smoke_test_from_a_run_config(tmp_path, monkeypatch):
    assert run(tmp_path, COMPLETE, monkeypatch) == 0
    [info] = logs(tmp_path)
    log = read_eval_log(info)
    assert log.status == "success"
    assert log.eval.task == "smoke_test"
    assert log.eval.metadata["corpus_version"] == "synthetic-0.1.0"


def test_shipped_run_yaml_runs_through_the_command(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    assert main(["run", "configs/run.yaml"]) == 0
    assert len(logs(tmp_path)) == 1


@pytest.mark.parametrize(
    "field",
    ["model", "task", "generate_config.temperature", "generate_config.seed",
     "metadata.corpus_version"],
)
def test_missing_required_field_is_refused_and_named(field, tmp_path, monkeypatch, capsys):
    cfg = copy.deepcopy(COMPLETE)
    parent = cfg
    *path, leaf = field.split(".")
    for part in path:
        parent = parent[part]
    del parent[leaf]
    assert run(tmp_path, cfg, monkeypatch) == 2
    assert field in capsys.readouterr().err
    assert logs(tmp_path) == []


def test_unknown_config_key_is_refused(tmp_path, monkeypatch, capsys):
    cfg = {**copy.deepcopy(COMPLETE), "temprature": 0}
    assert run(tmp_path, cfg, monkeypatch) == 2
    assert "temprature" in capsys.readouterr().err
    assert logs(tmp_path) == []


def test_missing_config_file_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    assert main(["run", str(tmp_path / "nope.yaml")]) == 2
    assert "nope.yaml" in capsys.readouterr().err
    assert logs(tmp_path) == []


@pytest.mark.parametrize("argv", [[], ["run"], ["frobnicate"], ["run", "a.yaml", "b.yaml"]])
def test_bad_command_line_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "usage: bench" in capsys.readouterr().err


def test_module_entry_point_matches_the_command(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "runner.cli", "run", str(tmp_path / "missing.yaml")],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 2
    assert "missing.yaml" in result.stderr
