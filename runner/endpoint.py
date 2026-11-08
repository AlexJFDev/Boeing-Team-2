"""Record what an OpenAI-compatible endpoint says about a ``vllm/...`` model.

A hosted endpoint exposes no weights digest, so the best available identity is
whatever ``GET /models`` lists for the model. The standard fields are always
recorded; ``extra_fields`` (dotted paths from the run config) add anything
endpoint-specific. The base URL and key are never recorded.
"""

import json
import os
import urllib.request

STANDARD_FIELDS = ("created", "owned_by", "root", "max_model_len")


class EndpointError(RuntimeError):
    """The endpoint is unreachable, or does not list the model."""


def _dig(entry: dict, path: str):
    for key in path.split("."):
        entry = entry.get(key) if isinstance(entry, dict) else None
    return entry


def vllm_model_metadata(model: str, extra_fields: tuple[str, ...] = ()) -> dict | None:
    """EvalLog metadata for a ``vllm/...`` model, or None for any other model."""
    if not model.startswith("vllm/"):
        return None
    name = model.removeprefix("vllm/")
    base_url = os.environ.get("VLLM_BASE_URL")
    if not base_url:
        raise EndpointError("VLLM_BASE_URL is not set")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            listing = json.load(response)
    except (OSError, ValueError) as e:
        raise EndpointError(f"could not reach the endpoint: {e}") from e

    entry = next((m for m in listing.get("data", []) if m.get("id") == name), None)
    if entry is None:
        raise EndpointError(f"endpoint does not list model '{name}'")
    fields = {"id": name}
    for path in (*STANDARD_FIELDS, *extra_fields):
        if (value := _dig(entry, path)) is not None:
            fields[path] = value
    return {"endpoint_model": fields}


OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DETAIL_FIELDS = ("format", "family", "parameter_size", "quantization_level")


def _ollama_root() -> str:
    """Server root, from the same OLLAMA_BASE_URL Inspect's Ollama provider reads."""
    base = os.environ.get("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_URL
    base = base.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _ollama_get(root: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{root}{path}", data=data)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def ollama_model_metadata(model: str) -> dict | None:
    """EvalLog metadata identifying an ``ollama/...`` model, or None for any other model.

    Records the Ollama server version, the model's full digest (``/api/tags``)
    and its details (``/api/show``). Never raises: if the server cannot be
    reached or does not have the model, ``status`` says so and ``error`` gives
    the reason, so a log without an identity is explicit rather than silent.
    The base URL is not recorded.
    """
    if not model.startswith("ollama/"):
        return None
    name = model.removeprefix("ollama/")
    tagged = name if ":" in name else f"{name}:latest"
    info: dict = {"id": name, "status": "ok"}
    root = _ollama_root()
    try:
        info["ollama_version"] = _ollama_get(root, "/api/version").get("version")
        tags = _ollama_get(root, "/api/tags").get("models", [])
        entry = next((m for m in tags if m.get("name") in (name, tagged)), None)
        if entry is None:
            info.update(status="model_not_found", error=f"Ollama does not list model '{tagged}'")
            return {"endpoint_model": info}
        info.update(name=entry.get("name"), digest=entry.get("digest"), size_bytes=entry.get("size"))
        shown = _ollama_get(root, "/api/show", {"model": tagged})
        details = shown.get("details") or {}
        for key in OLLAMA_DETAIL_FIELDS:
            if details.get(key) is not None:
                info[key] = details[key]
        if shown.get("modified_at") is not None:
            info["modified_at"] = shown["modified_at"]
    except (OSError, ValueError) as e:
        info.update(status="unreachable", error=f"could not query Ollama: {e}")
    return {"endpoint_model": info}
