"""Tests for the EvalLog provenance schema (Sprint 1 item 7.1, issue #16).

Acceptance criteria:
- When a mock run writes an EvalLog, it contains task ID, corpus hash, model hash,
  prompt template hash, raw output, parsed answer and token usage.
- When a log is missing any required field, schema validation fails.

Most tests run on a small dict that has the same field paths as a real Inspect
0.3.273 log (the JSON from `inspect log dump`). The two tests whose names start
with test_mock_run run real mock evals through Inspect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from bench.provenance import (
    REQUIRED_FIELDS,
    ProvenanceError,
    corpus_hash_from_manifest,
    model_hash,
    sha256_file,
    sha256_text,
    validate_log,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_mock_eval(task_file: str, log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run a task from the repo root with the mock model and return its EvalLog.

    Inspect globs the task path relative to the working directory, and pathlib's
    glob rejects absolute patterns on Python 3.11, so the path must be relative.
    """
    from inspect_ai import eval as run_eval

    monkeypatch.chdir(REPO_ROOT)
    logs = run_eval(task_file, model="mockllm/model", log_dir=str(log_dir))
    assert logs[0].status == "success"
    return logs[0]


def make_log() -> dict[str, Any]:
    return {
        "eval": {
            "task_id": "oW5AQKw4eD8LKZgFfPpECt",
            "task": "provenance_mock",
            "model": "mockllm/model",
            "model_args": {},
            "model_generate_config": {},
        },
        "samples": [
            {
                "id": 1,
                "store": {
                    "provenance": {
                        "corpus_hash": sha256_text("corpus"),
                        "prompt_template_hash": sha256_text("template"),
                    }
                },
                "output": {
                    "completion": "Default output from mockllm/model",
                    "usage": {"input_tokens": 7, "output_tokens": 33, "total_tokens": 40},
                },
                "scores": {
                    "includes": {"value": "I", "answer": "default output from mockllm/model"}
                },
            }
        ],
    }


# Each entry removes the one place a required field is read from.
BREAKERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "task_id": lambda log: log["eval"].pop("task_id"),
    "corpus_hash": lambda log: log["samples"][0]["store"]["provenance"].pop("corpus_hash"),
    "model_hash": lambda log: log["eval"].pop("model"),
    "prompt_template_hash": lambda log: log["samples"][0]["store"]["provenance"].pop(
        "prompt_template_hash"
    ),
    "raw_output": lambda log: log["samples"][0]["output"].pop("completion"),
    "parsed_answer": lambda log: log["samples"][0]["scores"]["includes"].pop("answer"),
    "token_usage": lambda log: log["samples"][0]["output"].pop("usage"),
}


def test_every_required_field_has_a_breaker() -> None:
    """Adding a field to the schema forces a matching test case here."""
    assert set(BREAKERS) == set(REQUIRED_FIELDS)


# --- Acceptance criterion 1: all fields present ------------------------------


def test_valid_log_yields_a_complete_record() -> None:
    [record] = validate_log(make_log())
    assert record.task_id == "oW5AQKw4eD8LKZgFfPpECt"
    assert record.corpus_hash == sha256_text("corpus")
    assert record.prompt_template_hash == sha256_text("template")
    assert record.model_hash == model_hash("mockllm/model")
    assert record.raw_output == "Default output from mockllm/model"
    assert record.parsed_answer == "default output from mockllm/model"
    assert record.token_usage.total_tokens == 40


def test_mock_run_writes_all_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 1 end to end: run a mock eval and check the EvalLog it writes."""
    log = run_mock_eval("tasks/provenance_mock.py", tmp_path, monkeypatch)
    [record] = validate_log(log)
    assert record.raw_output
    assert record.token_usage.total_tokens > 0


# --- Acceptance criterion 2 (negative case): missing field fails -------------


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_field_fails_validation(field: str) -> None:
    log = make_log()
    BREAKERS[field](log)
    with pytest.raises(ProvenanceError) as excinfo:
        validate_log(log)
    assert excinfo.value.fields == [field]


def test_null_value_counts_as_missing() -> None:
    log = make_log()
    log["samples"][0]["scores"]["includes"]["answer"] = None
    with pytest.raises(ProvenanceError) as excinfo:
        validate_log(log)
    assert excinfo.value.fields == ["parsed_answer"]


def test_malformed_hash_fails() -> None:
    log = make_log()
    log["samples"][0]["store"]["provenance"]["corpus_hash"] = "not-a-hash"
    with pytest.raises(ProvenanceError) as excinfo:
        validate_log(log)
    assert excinfo.value.fields == ["corpus_hash"]


def test_negative_token_count_fails() -> None:
    log = make_log()
    log["samples"][0]["output"]["usage"]["input_tokens"] = -1
    with pytest.raises(ProvenanceError) as excinfo:
        validate_log(log)
    assert excinfo.value.fields == ["token_usage"]


def test_log_without_samples_fails() -> None:
    log = make_log()
    log["samples"] = []
    with pytest.raises(ProvenanceError):
        validate_log(log)


def test_every_sample_is_checked() -> None:
    log = make_log()
    second = make_log()["samples"][0]
    second["id"] = 2
    second["output"].pop("completion")
    log["samples"].append(second)
    with pytest.raises(ProvenanceError, match="sample 2"):
        validate_log(log)


def test_mock_run_without_provenance_solver_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing smoke test never records the hashes, so its log must fail."""
    log = run_mock_eval("tasks/smoke_test.py", tmp_path, monkeypatch)
    with pytest.raises(ProvenanceError) as excinfo:
        validate_log(log)
    assert {"corpus_hash", "prompt_template_hash"} <= set(excinfo.value.fields)


# --- Parsed answer and scorer choice ------------------------------------------


def test_scorer_name_selects_the_parsed_answer() -> None:
    log = make_log()
    log["samples"][0]["scores"]["exact"] = {"value": 1.0, "answer": "DF17"}
    [default_record] = validate_log(log)
    [chosen_record] = validate_log(log, scorer_name="exact")
    assert default_record.parsed_answer == "default output from mockllm/model"
    assert chosen_record.parsed_answer == "DF17"


# --- Hash helpers -------------------------------------------------------------


def test_sha256_text_known_vector() -> None:
    assert (
        sha256_text("abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_model_hash_tracks_settings_but_not_key_order() -> None:
    base = model_hash("mockllm/model", {}, {"temperature": 0, "seed": 1})
    assert base != model_hash("mockllm/model", {}, {"temperature": 1, "seed": 1})
    assert base != model_hash("mockllm/other", {}, {"temperature": 0, "seed": 1})
    assert model_hash("m", {"a": 1, "b": 2}) == model_hash("m", {"b": 2, "a": 1})


def test_corpus_hash_from_manifest(tmp_path: Path) -> None:
    data = tmp_path / "frames.jsonl"
    data.write_text('{"frame_id": "f1"}\n')
    digest = sha256_file(data)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(f'{{"file": "frames.jsonl", "sha256": "{digest}"}}')
    assert corpus_hash_from_manifest(manifest) == digest

    data.write_text('{"frame_id": "tampered"}\n')
    with pytest.raises(ValueError, match="does not match"):
        corpus_hash_from_manifest(manifest)