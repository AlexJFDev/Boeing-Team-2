"""The ``bench`` command (WBS 1.19, issue #27).

    bench run configs/run.yaml

``bench run`` is the front door to ``runner.preflight``: it checks the run config
and refuses to start, before any model call, when a required field is missing or a
key is not recognised, naming the field and exiting with code 2. Otherwise it
launches the run. ``python -m runner.preflight <config>`` does the same thing.

Exit codes: 0 the run succeeded, 1 the run failed, 2 the config was refused or the
command line was wrong.

The ``bench`` script is registered in ``pyproject.toml``; run ``pip install -e .``
once after pulling to get it. ``python -m runner.cli`` works without that.
"""

import argparse
import sys

from runner import preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run = commands.add_parser(
        "run",
        help="check a run config, then run it",
        description="Check a YAML run config and refuse to start if it is incomplete, "
        "then run it.",
    )
    run.add_argument("config", help="path to the YAML run config, e.g. configs/run.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return preflight.main([args.config])
    raise AssertionError(f"unhandled command: {args.command}")  # argparse requires one


if __name__ == "__main__":
    sys.exit(main())
