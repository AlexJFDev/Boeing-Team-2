import copy

import pytest
import yaml
from inspect_ai.log import list_eval_logs, read_eval_log

from runner.preflight import main

COMPLETE = {
    "task": "tasks/smoke_test.py",
    "model": "mockllm/model",
    "generate_config": {"temperature": 0, "seed": 42},
    "metadata": {"corpus_version": "synthetic-0.1.0"},
}


def _run(tmp_path, cfg, monkeypatch):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return main([str(p)])


def _logs(tmp_path):
    return list_eval_logs(str(tmp_path / "logs"))


def test_complete_config_runs_and_is_logged(tmp_path, monkeypatch):
    assert _run(tmp_path, COMPLETE, monkeypatch) == 0
    [info] = _logs(tmp_path)
    log = read_eval_log(info)
    assert log.eval.model == "mockllm/model"
    assert log.eval.model_generate_config.temperature == 0
    assert log.eval.model_generate_config.seed == 42
    assert log.eval.metadata["corpus_version"] == "synthetic-0.1.0"


def test_shipped_run_yaml_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path / "logs"))
    assert main(["configs/run.yaml"]) == 0


@pytest.mark.parametrize(
    "field",
    ["model", "generate_config.temperature", "generate_config.seed", "metadata.corpus_version"],
)
def test_missing_field_refused_before_any_model_call(field, tmp_path, monkeypatch, capsys):
    cfg = copy.deepcopy(COMPLETE)
    parent = cfg
    *path, leaf = field.split(".")
    for p in path:
        parent = parent[p]
    del parent[leaf]
    assert _run(tmp_path, cfg, monkeypatch) == 2
    assert field in capsys.readouterr().err
    assert _logs(tmp_path) == []


def test_zero_is_not_missing(tmp_path, monkeypatch):
    cfg = copy.deepcopy(COMPLETE)
    cfg["generate_config"]["seed"] = 0
    assert _run(tmp_path, cfg, monkeypatch) == 0
