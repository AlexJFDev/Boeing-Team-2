"""Baseline evaluation of the ADS-B and ARINC 429 suites (WBS 3.8, issue #63).

    bench baseline configs/baseline_v1.0.0.yaml
    bench baseline --report-only logs/baseline/v1.0.0

Runs every task in the config against every model with Inspect's ``eval_set``, which
already handles the batch, retries and the EvalLogs. Then it writes the per-task
report and the ground-truth review list (``bench/baseline_report.py``). Those two are
the project-specific parts.

The config is checked before any model call, with the same reproducibility rules as
``bench run`` (CLAUDE.md rule 3): ``models``, ``generate_config.temperature``,
``generate_config.seed`` and ``metadata.corpus_version`` are required, and an unknown
key is refused. A refused config exits with code 2 and writes no log. As with
``bench run``, each ``vllm/...`` or ``openai-api/<service>/...`` model's endpoint is
queried first: an unreachable endpoint also exits with 2, and what it lists for the
model is recorded in every log.

``model_args`` is passed to every model, e.g. ``{stream: true}`` for an
``openai-api/...`` model on an endpoint that caps non-streaming output (VT ARC caps it
at 8000 tokens). Inspect records it in each EvalLog.

Logs go to ``logs/baseline/v<baseline_version>`` (git-ignored). Running the same
config again resumes: ``eval_set`` skips task-model pairs that already have a complete
log and retries the rest, so an interrupted baseline is finished, not restarted.

Exit codes: 0 every task-model pair succeeded, 1 a pair failed (the report is still
written for the pairs that finished), 2 the config was refused.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig, Model

from bench import baseline_report
from runner.endpoint import EndpointError, openai_api_model_metadata, vllm_model_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs" / "baseline"

ALLOWED = ("baseline_version", "tasks", "models", "generate_config", "model_args", "metadata",
           "retry_attempts", "max_tasks")
REQUIRED = ("baseline_version", "tasks", "models", "generate_config.temperature",
            "generate_config.seed", "metadata.corpus_version")


class BaselineConfigError(ValueError):
    """The baseline config is incomplete or names something that does not exist."""


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            return None
        node = node[part]
    return node


def load_and_check(path: str | Path) -> dict[str, Any]:
    """The parsed config, or BaselineConfigError naming every problem."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise BaselineConfigError(f"{path}: baseline config must be a YAML mapping")
    problems = [f"missing required field {f}" for f in REQUIRED if _get(cfg, f) is None]
    problems += [f"unknown field {k}" for k in cfg if k not in ALLOWED]
    if isinstance(cfg.get("generate_config"), dict):
        problems += [f"unknown field generate_config.{k}" for k in cfg["generate_config"]
                     if k not in GenerateConfig.model_fields]
    for key in ("tasks", "models"):
        value = cfg.get(key)
        if value is not None and not (isinstance(value, list) and value
                                      and all(isinstance(v, str) and v.strip() for v in value)):
            problems.append(f"{key} must be a non-empty list of strings")
        elif value is not None and len(set(value)) != len(value):
            problems.append(f"{key} lists the same entry twice")
    model_args = cfg.get("model_args")
    if model_args is not None and not isinstance(model_args, dict):
        problems.append("model_args must be a mapping, e.g. {stream: true}")
    for task in cfg.get("tasks") or []:
        if isinstance(task, str) and not (REPO_ROOT / task.split("@")[0]).is_file():
            problems.append(f"task file {task} does not exist")
    if problems:
        raise BaselineConfigError("baseline config refused: " + "; ".join(problems))
    return cfg


def endpoint_metadata(models: list[str]) -> dict[str, Any]:
    """What each hosted model's endpoint lists for it, as ``bench run`` records.

    Covers ``vllm/...`` and ``openai-api/<service>/...`` models. Raises EndpointError,
    before any model call, when an endpoint can't be reached or doesn't list the model.
    Other providers record nothing here.
    """
    found = {m: meta["endpoint_model"] for m in models
             if (meta := vllm_model_metadata(m) or openai_api_model_metadata(m))}
    return {"endpoint_models": found} if found else {}


def log_dir_for(cfg: dict[str, Any], root: Path = LOG_ROOT) -> Path:
    return root / f"v{cfg['baseline_version']}"


def run(cfg: dict[str, Any], log_dir: Path, models: list[str | Model] | None = None,
        display: str | None = None, extra_metadata: dict[str, Any] | None = None) -> bool:
    """Run every task against every model. ``models`` overrides the config's (for rehearsals)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    success, _ = eval_set(
        # Inspect resolves task files relative to the working directory, not absolutely.
        [os.path.relpath(REPO_ROOT / t) for t in cfg["tasks"]],
        model=models if models is not None else cfg["models"],
        log_dir=str(log_dir),
        retry_attempts=cfg.get("retry_attempts", 3),
        model_args=cfg.get("model_args") or {},
        max_tasks=cfg.get("max_tasks"),
        metadata={**cfg["metadata"], "baseline_version": cfg["baseline_version"],
                  **(extra_metadata or {})},
        display=display,  # type: ignore[arg-type]
        **cfg["generate_config"],
    )
    return bool(success)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench baseline", description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", help="baseline config, e.g. configs/baseline_v1.0.0.yaml")
    parser.add_argument("--report-only", metavar="LOG_DIR",
                        help="skip the run and rebuild the report from the logs in LOG_DIR")
    args = parser.parse_args(argv)

    if args.report_only:
        report = baseline_report.write_report(Path(args.report_only))
        print(baseline_report.summary_text(report))
        return 0
    if not args.config:
        parser.error("give a baseline config, or --report-only LOG_DIR")

    load_dotenv()  # provider settings such as OLLAMA_BASE_URL; never printed
    try:
        cfg = load_and_check(args.config)
        endpoints = endpoint_metadata(cfg["models"])
    except (BaselineConfigError, EndpointError, OSError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    log_dir = log_dir_for(cfg)
    ok = run(cfg, log_dir, extra_metadata=endpoints)
    report = baseline_report.write_report(log_dir)
    print(baseline_report.summary_text(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
