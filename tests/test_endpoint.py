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


# --- Ollama model identity (D2 harness, item 6.2) ---------------------------
import socket  # noqa: E402

from runner.endpoint import ollama_model_metadata  # noqa: E402

OLLAMA_DIGEST = "a" * 64


@pytest.fixture
def ollama(monkeypatch):
    seen: list[tuple[str, dict | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            seen.append((self.path, None))
            if self.path == "/api/version":
                self._reply({"version": "0.0.1-test"})
            elif self.path == "/api/tags":
                self._reply({"models": [{"name": "test-model:latest", "digest": OLLAMA_DIGEST, "size": 123}]})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            self._reply({"details": {"family": "llama", "parameter_size": "8.0B",
                                     "quantization_level": "Q4_K_M", "format": "gguf"},
                         "modified_at": "2026-01-01T00:00:00Z"})

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    yield seen
    server.shutdown()


def test_ollama_identity_is_recorded(ollama):
    meta = ollama_model_metadata("ollama/test-model")
    assert meta == {"endpoint_model": {
        "id": "test-model", "status": "ok", "ollama_version": "0.0.1-test",
        "name": "test-model:latest", "digest": OLLAMA_DIGEST, "size_bytes": 123,
        "format": "gguf", "family": "llama", "parameter_size": "8.0B",
        "quantization_level": "Q4_K_M", "modified_at": "2026-01-01T00:00:00Z",
    }}
    assert ("/api/show", {"model": "test-model:latest"}) in ollama
    assert "127.0.0.1" not in json.dumps(meta), "the base URL is not recorded"


def test_ollama_model_not_pulled_is_recorded_explicitly(ollama):
    meta = ollama_model_metadata("ollama/other-model")
    assert meta is not None
    assert meta["endpoint_model"]["status"] == "model_not_found"
    assert "other-model:latest" in meta["endpoint_model"]["error"]


def test_unreachable_ollama_is_recorded_explicitly(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{port}/v1")
    meta = ollama_model_metadata("ollama/test-model")
    assert meta is not None
    assert meta["endpoint_model"]["status"] == "unreachable"
    assert "could not query Ollama" in meta["endpoint_model"]["error"]


def test_non_ollama_model_has_no_ollama_metadata():
    assert ollama_model_metadata("mockllm/model") is None
