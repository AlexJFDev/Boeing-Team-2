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

The optional ``guardrail_profile`` key (``name`` and ``version``) selects a guardrail
profile (see ``runner.guardrails``). An unknown or edited profile is refused before
any model call. The profile is passed to the task as the ``guardrail_profile`` and
``guardrail_version`` task parameters, so the task must accept them; it is recorded in
the log under ``guardrail_profile_*``. Without the key the log records ``none``.
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai import list_tasks
from inspect_ai.model import GenerateConfig

from runner.endpoint import EndpointError, vllm_model_metadata
from runner.guardrails import NONE, GuardrailProfile, GuardrailProfileError, load_guardrail_profile

ALLOWED = ("task", "model", "generate_config", "metadata", "model_info_fields", "guardrail_profile")
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
    guardrail_profile(cfg)  # refuse a bad or edited profile now, before any model call
    check_task_accepts_profile(cfg)
    return cfg


GUARDRAIL_PARAMS = ("guardrail_profile", "guardrail_version")


def check_task_accepts_profile(cfg: dict[str, Any]) -> None:
    """Refuse a config that names a guardrail profile for a task that cannot apply it.

    Inspect only warns about a task parameter the task does not use, so without this
    check the run would go ahead with no system message while the log said otherwise.
    """
    if cfg.get("guardrail_profile") is None:
        return
    spec = str(cfg["task"])
    path, _, name = spec.partition("@")
    if not path.endswith(".py") or not Path(path).is_file():
        raise PreflightError(
            f"cannot check that task {spec!r} accepts a guardrail profile: "
            "name a task file (path/to/file.py or path/to/file.py@task_name)"
        )
    try:
        tasks = [t for t in list_tasks(Path(path)) if not name or t.name == name]
    except Exception as e:
        raise PreflightError(f"cannot load task {spec!r} to check its parameters: {e}") from e
    if not tasks:
        raise PreflightError(f"no task {name or ''!r} found in {path}")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    for info in tasks:
        accepted = inspect.signature(getattr(module, info.name)).parameters
        missing = [p for p in GUARDRAIL_PARAMS if p not in accepted]
        if missing:
            raise PreflightError(
                f"task {info.name!r} in {path} does not accept {', '.join(missing)}, so it "
                "cannot apply the guardrail profile; add the parameters and "
                "profile.pre_solvers() to the task (see tasks/guardrail_example.py)"
            )


def guardrail_profile(cfg: dict[str, Any]) -> GuardrailProfile:
    """The profile the config selects; ``none`` when it selects nothing."""
    selected = cfg.get("guardrail_profile")
    if selected is None:
        return load_guardrail_profile(NONE)
    if not isinstance(selected, dict):
        raise PreflightError("guardrail_profile must be a mapping with 'name' and 'version'")
    extra = [k for k in selected if k not in ("name", "version")]
    if extra:
        raise PreflightError(f"guardrail_profile has unknown field(s): {', '.join(extra)}")
    name, version = selected.get("name"), selected.get("version", 1)
    if not isinstance(name, str) or not name:
        raise PreflightError("guardrail_profile.name must be a non-empty string")
    if isinstance(version, bool) or not isinstance(version, int):
        raise PreflightError("guardrail_profile.version must be an integer")
    try:
        return load_guardrail_profile(name, version)
    except GuardrailProfileError as e:
        raise PreflightError(str(e)) from e


def run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Metadata for the EvalLog: the config's own, the guardrail profile's, and the
    endpoint's for vllm models."""
    endpoint = vllm_model_metadata(cfg["model"], tuple(cfg.get("model_info_fields", ())))
    return {**cfg["metadata"], **guardrail_profile(cfg).metadata(), **(endpoint or {})}


def task_args(cfg: dict[str, Any]) -> dict[str, Any]:
    """Task parameters for the selected profile; empty when the config selects none."""
    if cfg.get("guardrail_profile") is None:
        return {}
    profile = guardrail_profile(cfg)
    return {"guardrail_profile": profile.name, "guardrail_version": profile.version}


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
        task_args=task_args(cfg),
        **cfg["generate_config"],
    )
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    sys.exit(main())
