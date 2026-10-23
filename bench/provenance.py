"""Provenance schema for EvalLogs (Sprint 1 item 7.1, issue #16).

Inspect AI already writes most of what a rerun needs. This module adds the
domain-specific parts: hashing helpers and a schema check that fails when any
required field is missing from a sample.

It has no Inspect imports on purpose, so it works on a plain dict (the JSON
from `inspect log dump`) as well as on an EvalLog object.

Where each required field comes from in an Inspect 0.3.273 EvalLog:

    task_id               eval.task_id
    corpus_hash           samples[i].store.provenance.corpus_hash
                          (written by bench.provenance_solver.record_provenance)
    model_hash            derived from eval.model, eval.model_args and
                          eval.model_generate_config
    prompt_template_hash  samples[i].store.provenance.prompt_template_hash
                          (written by bench.provenance_solver.record_provenance)
    raw_output            samples[i].output.completion
    parsed_answer         samples[i].scores[<scorer>].answer
    token_usage           samples[i].output.usage
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

PROVENANCE_VERSION = "1.0.0"
STORE_KEY = "provenance"

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ProvenanceRecord(BaseModel):
    """One record per sample. Every field is required; there are no defaults."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    corpus_hash: Sha256Hex
    model_hash: Sha256Hex
    prompt_template_hash: Sha256Hex
    raw_output: str
    parsed_answer: str
    token_usage: TokenUsage


REQUIRED_FIELDS: tuple[str, ...] = tuple(ProvenanceRecord.model_fields)


class ProvenanceError(ValueError):
    """Raised when a log is missing a required provenance field."""

    def __init__(self, message: str, fields: list[str]) -> None:
        super().__init__(message)
        self.fields = fields


# --- Hashing ----------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_hash_from_manifest(manifest_path: Path | str) -> str:
    """Return the sha256 in a MANIFEST.json after checking it against the data file."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    actual = sha256_file(manifest_path.parent / manifest["file"])
    if actual != manifest["sha256"]:
        raise ValueError(
            f"{manifest['file']} does not match the sha256 recorded in {manifest_path.name}"
        )
    return str(manifest["sha256"])


def model_hash(
    model: str,
    model_args: dict[str, Any] | None = None,
    generate_config: dict[str, Any] | None = None,
) -> str:
    """Hash of the model name and the settings that can change a result.

    This identifies the model and its configuration, not its weights. A log
    cannot tell us the weights, so a weights digest (for example an Ollama model
    digest) would have to be passed in through model_args by the caller.
    """
    canonical = json.dumps(
        {
            "model": model,
            "model_args": model_args or {},
            "generate_config": generate_config or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


# --- Extraction and validation ----------------------------------------------


def _get(data: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def build_record(
    log: dict[str, Any], sample: dict[str, Any], scorer_name: str | None = None
) -> ProvenanceRecord:
    """Build and validate the record for one sample.

    Raises ProvenanceError, with the offending field names, if anything required
    is missing or malformed. `scorer_name` picks which score supplies the parsed
    answer; by default the first score on the sample is used.
    """
    eval_spec = log.get("eval") or {}
    model = eval_spec.get("model")
    stored = _get(sample, "store", STORE_KEY)
    if not isinstance(stored, dict):
        stored = {}
    scores = sample.get("scores") or {}
    score = scores.get(scorer_name) if scorer_name else next(iter(scores.values()), None)

    candidate = {
        "task_id": eval_spec.get("task_id"),
        "corpus_hash": stored.get("corpus_hash"),
        "model_hash": (
            model_hash(
                model,
                eval_spec.get("model_args"),
                eval_spec.get("model_generate_config"),
            )
            if model
            else None
        ),
        "prompt_template_hash": stored.get("prompt_template_hash"),
        "raw_output": _get(sample, "output", "completion"),
        "parsed_answer": score.get("answer") if isinstance(score, dict) else None,
        "token_usage": _get(sample, "output", "usage"),
    }
    try:
        return ProvenanceRecord.model_validate(candidate)
    except ValidationError as err:
        fields = sorted({str(error["loc"][0]) for error in err.errors()})
        raise ProvenanceError(
            f"sample {sample.get('id')!r}: missing or invalid provenance field(s): "
            f"{', '.join(fields)}",
            fields,
        ) from err


def validate_log(log: Any, scorer_name: str | None = None) -> list[ProvenanceRecord]:
    """Check every sample in a log and return one ProvenanceRecord per sample.

    `log` is an Inspect EvalLog or the dict that `inspect log dump` produces.
    """
    data = log if isinstance(log, dict) else log.model_dump(mode="json", exclude_none=True)
    samples = data.get("samples") or []
    if not samples:
        raise ProvenanceError(
            "log has no samples, so there is nothing to check", list(REQUIRED_FIELDS)
        )
    return [build_record(data, sample, scorer_name) for sample in samples]