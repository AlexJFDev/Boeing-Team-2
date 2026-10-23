import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml
from inspect_ai.log import list_eval_logs, read_eval_log

from runner.preflight import main, run_metadata

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


def test_unknown_top_level_key_refused_and_named(tmp_path, monkeypatch, capsys):
    cfg = copy.deepcopy(COMPLETE)
    cfg["limit"] = 1
    assert _run(tmp_path, cfg, monkeypatch) == 2
    assert "limit" in capsys.readouterr().err
    assert _logs(tmp_path) == []


def test_unknown_generate_config_key_refused_and_named(tmp_path, monkeypatch, capsys):
    cfg = copy.deepcopy(COMPLETE)
    cfg["generate_config"]["temprature"] = 0
    assert _run(tmp_path, cfg, monkeypatch) == 2
    assert "generate_config.temprature" in capsys.readouterr().err
    assert _logs(tmp_path) == []


def test_unreachable_vllm_endpoint_refused_before_any_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    cfg = copy.deepcopy(COMPLETE)
    cfg["model"] = "vllm/test-model"
    assert _run(tmp_path, cfg, monkeypatch) == 2
    assert "could not reach" in capsys.readouterr().err
    assert _logs(tmp_path) == []


def test_endpoint_info_is_merged_into_run_metadata(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps({"data": [{"id": "test-model", "info": {"updated_at": 2000}}]}).encode()
            )

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("VLLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    cfg = copy.deepcopy(COMPLETE)
    cfg["model"] = "vllm/test-model"
    cfg["model_info_fields"] = ["info.updated_at"]
    try:
        meta = run_metadata(cfg)
    finally:
        server.shutdown()
    assert meta["corpus_version"] == "synthetic-0.1.0"
    assert meta["endpoint_model"]["info.updated_at"] == 2000
