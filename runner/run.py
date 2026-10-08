"""Run an Inspect task from a run config.

    python -m runner.run configs/baseline.json tasks/smoke_test.py [--model mockllm/model]

The config is validated before anything starts; a missing ``temperature`` or
``seed`` aborts with a message naming the field.
"""

import argparse
import sys

from inspect_ai import eval as inspect_eval

from runner.config import RunConfigError, load_run_config


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

    logs = inspect_eval(
        args.task,
        model=args.model or cfg.model,
        temperature=cfg.temperature,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
    )
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    sys.exit(main())
