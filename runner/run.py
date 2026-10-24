"""Run an Inspect task from a run config.

    python -m runner.run configs/baseline.json tasks/smoke_test.py [--model mockllm/model]

The config is validated before anything starts; a missing ``temperature`` or
``seed`` aborts with a message naming the field. For ``vllm/...`` models the
endpoint is queried first, so an unreachable endpoint aborts before any log is
written, and what it lists for the model is recorded in the log (``model_info_fields`` in the
config names extra endpoint-specific fields).
"""

import argparse
import sys

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

from runner.config import RunConfigError, load_run_config
from runner.endpoint import EndpointError, vllm_model_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("task")
    parser.add_argument("--model", help="override the config's model (e.g. mockllm/model)")
    args = parser.parse_args(argv)

    try:
        cfg = load_run_config(args.config)
    except RunConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    model = args.model or cfg.model
    load_dotenv()
    try:
        metadata = vllm_model_metadata(model, cfg.model_info_fields)
    except EndpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    logs = inspect_eval(
        args.task,
        model=model,
        metadata=metadata,
        temperature=cfg.temperature,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
    )
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    sys.exit(main())
