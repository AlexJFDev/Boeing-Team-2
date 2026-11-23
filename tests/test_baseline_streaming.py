"""Checks for the streamed VT ARC baseline (3.8, issue #63, baseline v1.3.0).

v1.2.0 called the ARC models through Inspect's ``vllm/`` provider, which can't stream,
so every request hit ARC's 8000-token cap for non-streaming output and 203 answers came
back blank. v1.3.0 calls them through Inspect's OpenAI-compatible provider
(``openai-api/arc/...``) with ``model_args: {stream: true}``.

These tests run a real ``eval_set`` against a local stand-in for an OpenAI-compatible
endpoint, which lists models on ``GET /models`` and streams ``POST /chat/completions``
as server-sent events, recording each request. Acceptance criteria:
- When the baseline has ``model_args: {stream: true}``, every request to the endpoint
  asks to stream and carries the configured max_tokens, temperature and seed, and the
  answers are scored (positive case).
- When a model stops on its token limit with no text, or partway through its answer,
  the report records it as no answer or cut off, and the sample is not flagged for
  review (negative case).
- When the endpoint's base URL is missing or the endpoint doesn't list a model, the
  run is refused with exit code 2 before any model call (negative case).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import baseline_report
from runner import baseline
from runner.endpoint import EndpointError, openai_api_model_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
V130 = REPO_ROOT / "configs" / "baseline_v1.3.0.yaml"
B3 = "tasks/b3_arinc429_bus_conformance.py"

# Model id -> (content to stream, finish_reason). "answers" says no violation on every
# stream, so it is right on B3's three conforming controls and wrong on the rest.
BEHAVIOUR = {
    "answers": ('{"violations":[]}', "stop"),
    "blank": ("", "length"),
    "truncated": ('{"violations":[[3,"ra', "length"),
}


class FakeEndpoint:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = {"object": "list", "data": [{"id": m, "object": "model", "created": 1,
                                                    "owned_by": "arc"} for m in BEHAVIOUR]}
                self._send(200, json.dumps(body).encode(), "application/json")

            def do_POST(self) -> None:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(request)
                content, finish = BEHAVIOUR[request["model"]]
                if not request.get("stream"):  # behave like ARC: refuse what it would cap
                    self._send(400, b'{"error":"this stand-in only streams"}', "application/json")
                    return
                chunks = [{"role": "assistant", "content": ""}]
                chunks += [{"content": content[i:i + 7]} for i in range(0, len(content), 7)]
                events = [{"id": "c1", "object": "chat.completion.chunk", "created": 1,
                           "model": request["model"],
                           "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                          for d in chunks]
                events.append({"id": "c1", "object": "chat.completion.chunk", "created": 1,
                               "model": request["model"],
                               "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                         "total_tokens": 15}})
                body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
                self._send(200, body.encode(), "text/event-stream")

            def _send(self, status: int, body: bytes, kind: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture(scope="module")
def endpoint() -> Iterator[FakeEndpoint]:
    fake = FakeEndpoint()
    yield fake
    fake.server.shutdown()


@pytest.fixture
def arc_env(endpoint: FakeEndpoint, monkeypatch: pytest.MonkeyPatch) -> FakeEndpoint:
    monkeypatch.setenv("FAKEARC_BASE_URL", endpoint.url)
    monkeypatch.setenv("FAKEARC_API_KEY", "test-key")
    return endpoint


def v130(**changes: Any) -> dict[str, Any]:
    cfg = yaml.safe_load(V130.read_text())
    cfg.update(changes)
    return cfg


# --- The config --------------------------------------------------------------------


def test_v130_streams_the_three_arc_models_through_the_openai_compatible_provider() -> None:
    cfg = baseline.load_and_check(V130)
    assert cfg["models"] == ["openai-api/arc/gpt-oss-120b", "openai-api/arc/GLM-5.3",
                             "openai-api/arc/DeepSeek-V4.1-Flash"]
    assert cfg["model_args"] == {"stream": True}
    gen = cfg["generate_config"]
    assert gen["max_tokens"] > 8000  # past ARC's non-streaming cap, which v1.2.0 hit
    assert gen["max_connections"] * len(cfg["models"]) <= 20 and gen["max_connections"] <= 10
    assert gen["temperature"] == 0 and gen["seed"] == 42
    assert cfg["tasks"] == baseline.load_and_check(REPO_ROOT / "configs" / "baseline_v1.2.0.yaml")["tasks"]


def test_model_args_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(v130(model_args=["stream"])))
    with pytest.raises(baseline.BaselineConfigError, match="model_args must be a mapping"):
        baseline.load_and_check(path)


# --- The endpoint check ---------------------------------------------------------------


def test_openai_api_endpoint_is_checked_and_recorded(arc_env: FakeEndpoint) -> None:
    meta = openai_api_model_metadata("openai-api/fakearc/answers")
    assert meta == {"endpoint_model": {"service": "fakearc", "id": "answers", "created": 1,
                                       "owned_by": "arc"}}
    assert openai_api_model_metadata("vllm/answers") is None


def test_unlisted_model_or_missing_url_is_refused(arc_env: FakeEndpoint,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(EndpointError, match="does not list model 'nope'"):
        openai_api_model_metadata("openai-api/fakearc/nope")
    with pytest.raises(EndpointError, match="openai-api/<service>/<model>"):
        openai_api_model_metadata("openai-api/fakearc")
    monkeypatch.delenv("FAKEARC_BASE_URL")
    with pytest.raises(EndpointError, match="FAKEARC_BASE_URL is not set"):
        openai_api_model_metadata("openai-api/fakearc/answers")


def test_missing_arc_url_exits_2_before_any_model_call(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARC_BASE_URL", raising=False)
    monkeypatch.setattr(baseline, "load_dotenv", lambda: None)  # never read a developer's .env
    monkeypatch.setattr(baseline, "run", lambda *a, **k: pytest.fail("the run must not start"))
    assert baseline.main([str(V130)]) == 2


# --- A streamed run ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def streamed(endpoint: FakeEndpoint, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FAKEARC_BASE_URL", endpoint.url)
        mp.setenv("FAKEARC_API_KEY", "test-key")
        cfg = v130(tasks=[B3], models=[f"openai-api/fakearc/{m}" for m in BEHAVIOUR],
                   generate_config={"temperature": 0, "seed": 42, "max_tokens": 32000,
                                    "max_connections": 2})
        log_dir = tmp_path_factory.mktemp("streamed") / "v1.3.0"
        endpoints = baseline.endpoint_metadata(cfg["models"])
        assert baseline.run(cfg, log_dir, display="none", extra_metadata=endpoints)
        return baseline_report.write_report(log_dir)


def test_every_request_streams_with_the_configured_settings(streamed: dict[str, Any],
                                                            endpoint: FakeEndpoint) -> None:
    assert len(endpoint.requests) >= 16 * 3
    for request in endpoint.requests:
        assert request["stream"] is True
        assert request.get("max_tokens", request.get("max_completion_tokens")) == 32000
        assert request["temperature"] == 0 and request["seed"] == 42


def test_streamed_answers_are_scored(streamed: dict[str, Any]) -> None:
    run = next(r for r in streamed["runs"] if r["model"] == "openai-api/fakearc/answers")
    assert run["model_args"] == {"stream": True} and run["max_tokens"] == 32000
    assert run["no_answer"] == 0 and run["cut_off"] == 0
    assert run["capability_mean"] == pytest.approx(3 / 16)  # the three conforming controls


def test_token_limit_stops_are_no_answer_or_cut_off(streamed: dict[str, Any]) -> None:
    runs = {r["model"].rsplit("/", 1)[1]: r for r in streamed["runs"]}
    assert (runs["blank"]["no_answer"], runs["blank"]["cut_off"]) == (16, 0)
    assert runs["blank"]["stop_reasons"] == {"max_tokens": 16}
    assert (runs["truncated"]["no_answer"], runs["truncated"]["cut_off"]) == (0, 16)
    sample = next(s for s in streamed["samples"] if s["sample_id"] == "b3-03")
    truncated = sample["results"]["openai-api/fakearc/truncated"]
    assert truncated["cut_off"] is True and truncated["no_answer"] is False
    assert truncated["answer"].startswith('{"violations":[[3')  # the partial text is kept


def test_cut_off_samples_are_not_flagged(streamed: dict[str, Any]) -> None:
    assert streamed["review"] == []
    assert len(streamed["unreviewable"]) == 13  # every model scored 0 on the 13 non-controls
    md = baseline_report.render_markdown(streamed)
    assert "(cut off)" in md and "stream=True" in md
