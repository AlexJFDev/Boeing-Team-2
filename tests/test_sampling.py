import json
import socket

import pytest
from inspect_ai import eval as inspect_eval

from runner.config import RunConfigError, load_run_config, parse_run_config
from runner.run import main

TASK = "tasks/smoke_test.py"


def _write(tmp_path, data):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    return p


@pytest.mark.parametrize("missing", ["temperature", "seed"])
def test_missing_field_is_refused_and_named(missing):
    data = {"model": "mockllm/model", "temperature": 0, "seed": 42}
    del data[missing]
    with pytest.raises(RunConfigError, match=missing):
        parse_run_config(data)


def test_runner_refuses_to_start_without_seed(tmp_path, capsys):
    cfg = _write(tmp_path, {"model": "mockllm/model", "temperature": 0})
    assert main([str(cfg), TASK]) == 2
    assert "seed" in capsys.readouterr().err


def test_baseline_config_loads():
    cfg = load_run_config("configs/baseline.json")
    assert (cfg.temperature, cfg.seed) == (0.0, 42)


def test_settings_recorded_in_evallog(tmp_path):
    cfg = load_run_config("configs/baseline.json")
    [log] = inspect_eval(
        TASK,
        model="mockllm/model",
        temperature=cfg.temperature,
        seed=cfg.seed,
        log_dir=str(tmp_path),
    )
    assert log.eval.model_generate_config.temperature == 0
    assert log.eval.model_generate_config.seed == 42


def test_mockllm_five_identical_completions(tmp_path):
    cfg = load_run_config("configs/baseline.json")
    [log] = inspect_eval(
        TASK,
        model="mockllm/model",
        temperature=cfg.temperature,
        seed=cfg.seed,
        epochs=5,
        log_dir=str(tmp_path),
    )
    outputs = [s.output.completion for s in log.samples]
    assert len(outputs) == 5 and len(set(outputs)) == 1


def _ollama_up() -> bool:
    try:
        socket.create_connection(("localhost", 11434), timeout=0.5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="no local Ollama server")
def test_ollama_five_identical_completions(tmp_path):
    cfg = load_run_config("configs/baseline.json")
    [log] = inspect_eval(
        TASK,
        model=cfg.model,
        temperature=cfg.temperature,
        seed=cfg.seed,
        epochs=5,
        log_dir=str(tmp_path),
    )
    assert log.status == "success"
    outputs = [s.output.completion for s in log.samples]
    assert len(outputs) == 5 and len(set(outputs)) == 1
