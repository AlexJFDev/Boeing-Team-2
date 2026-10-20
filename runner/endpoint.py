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
