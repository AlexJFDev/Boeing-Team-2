"""Pre-flight check for a YAML run config, then launch the run.

    python -m runner.preflight configs/run.yaml

The file uses the same keys as ``inspect eval --run-config`` (``task``, ``model``,
``generate_config``, ``metadata``). Inspect already handles loading it; this module
only refuses to start when a reproducibility field is missing, before any model
call. ``metadata.corpus_version`` has no Inspect equivalent, so it travels in the
run metadata and shows up in ``EvalLog.eval.metadata``.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import eval as inspect_eval

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
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args(argv)

    try:
        cfg = load_and_check(args.config)
    except (PreflightError, OSError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    logs = inspect_eval(
        cfg["task"],
        model=cfg["model"],
        metadata=cfg["metadata"],
        **cfg["generate_config"],
    )
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    sys.exit(main())
