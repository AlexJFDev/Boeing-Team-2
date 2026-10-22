import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from runner.endpoint import EndpointError, vllm_model_metadata
from runner.run import main

TASK = "tasks/smoke_test.py"

LISTING = {
    "data": [
        {
            "id": "test-model",
            "created": 1000,
            "owned_by": "test",
            "info": {"updated_at": 2000},
        }
    ]
}


@pytest.fixture
def endpoint(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(LISTING).encode())

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("VLLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("VLLM_API_KEY", "key")
    yield
    server.shutdown()


def test_standard_and_extra_fields_are_recorded(endpoint):
    meta = vllm_model_metadata("vllm/test-model", ("info.updated_at", "info.absent"))
    assert meta == {
        "endpoint_model": {
            "id": "test-model",
            "created": 1000,
            "owned_by": "test",
            "info.updated_at": 2000,
        }
    }


def test_non_vllm_model_has_no_endpoint_metadata():
    assert vllm_model_metadata("mockllm/model") is None


def test_unlisted_model_is_refused_and_named(endpoint):
    with pytest.raises(EndpointError, match="missing-model"):
        vllm_model_metadata("vllm/missing-model")


def test_unreachable_endpoint_writes_no_log(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"model": "vllm/test-model", "temperature": 0, "seed": 42}))
    assert main([str(cfg), TASK]) == 2
    assert "could not reach" in capsys.readouterr().err
    assert list(tmp_path.glob("*.eval")) == []
