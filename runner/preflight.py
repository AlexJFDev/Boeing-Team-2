"""Pre-flight check for a YAML run config, then launch the run.

    python -m runner.preflight configs/run.yaml

The file uses the same keys as ``inspect eval --run-config`` (``task``, ``model``,
``generate_config``, ``metadata``). Inspect already handles loading it; this module
only refuses to start when a reproducibility field is missing, or a key is not
recognised, before any model call. ``metadata.corpus_version`` has no Inspect
equivalent, so it travels in the run metadata and shows up in
``EvalLog.eval.metadata``.

For ``vllm/...`` models the endpoint is queried first (see ``runner.endpoint``): an
unreachable endpoint aborts before any log is written, and what it lists for the
model is recorded in the log. The optional ``model_info_fields`` key names extra
endpoint-specific fields to record.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig

from runner.endpoint import EndpointError, vllm_model_metadata

ALLOWED = ("task", "model", "generate_config", "metadata", "model_info_fields")
REQUIRED = (
    "task",
    "model",
    "generate_config.temperature",
    "generate_config.seed",
    "metadata.corpus_version",
)


class PreflightError(ValueError):
    pass


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            return None
        node = node[part]
    return node


def load_and_check(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise PreflightError(f"{path}: run config must be a YAML mapping")
    missing = [f for f in REQUIRED if _get(cfg, f) is None]
    if missing:
        raise PreflightError(f"run config is missing required field(s): {', '.join(missing)}")
    unknown = [k for k in cfg if k not in ALLOWED]
    unknown += [
        f"generate_config.{k}" for k in cfg["generate_config"] if k not in GenerateConfig.model_fields
    ]
    if unknown:
        raise PreflightError(f"run config has unknown field(s): {', '.join(unknown)}")
    return cfg


def run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Metadata for the EvalLog: the config's own, plus the endpoint's for vllm models."""
    endpoint = vllm_model_metadata(cfg["model"], tuple(cfg.get("model_info_fields", ())))
    return {**cfg["metadata"], **(endpoint or {})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        cfg = load_and_check(args.config)
        metadata = run_metadata(cfg)
    except (PreflightError, EndpointError, OSError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    logs = inspect_eval(
        cfg["task"],
        model=cfg["model"],
        metadata=metadata,
        **cfg["generate_config"],
    )
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    sys.exit(main())
