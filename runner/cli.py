"""The ``bench`` command (WBS 1.19, issue #27).

    bench run configs/run.yaml
    bench baseline configs/baseline_v1.0.0.yaml

``bench run`` is the front door to ``runner.preflight``: it checks the run config
and refuses to start, before any model call, when a required field is missing or a
key is not recognised, naming the field and exiting with code 2. Otherwise it
launches the run. ``--resume`` picks up an interrupted run (WBS 1.20, issue #28). ``python -m runner.preflight <config>`` does the same thing.

``bench baseline`` runs every task in a baseline config against every model, then
writes the per-task report and the ground-truth review list (WBS 3.8, see
``runner.baseline``). ``bench baseline --report-only <log dir>`` rebuilds the report.

Exit codes: 0 the run succeeded, 1 the run failed, 2 the config was refused or the
command line was wrong.

The ``bench`` script is registered in ``pyproject.toml``; run ``pip install -e .``
once after pulling to get it. ``python -m runner.cli`` works without that.
"""

import argparse
import sys

from runner import baseline, preflight


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
    base = commands.add_parser(
        "baseline",
        help="run every task in a baseline config against every model, then report",
        description="Run a baseline config with Inspect's eval_set, then write the per-task "
        "report and the ground-truth review list.",
    )
    base.add_argument("config", nargs="?", help="baseline config, e.g. configs/baseline_v1.0.0.yaml")
    base.add_argument("--report-only", metavar="LOG_DIR",
                      help="skip the run and rebuild the report from LOG_DIR")
    run.add_argument(
        "--resume",
        action="store_true",
        help="start, or pick up an interrupted run: skip tasks already completed in --log-dir",
    )
    run.add_argument("--log-dir", help="log directory for --resume (default: logs)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        extra = (["--resume"] if args.resume else []) + (
            ["--log-dir", args.log_dir] if args.log_dir else []
        )
        return preflight.main([args.config, *extra])
    if args.command == "baseline":
        forwarded = [args.config] if args.config else []
        if args.report_only:
            forwarded += ["--report-only", args.report_only]
        return baseline.main(forwarded)
    raise AssertionError(f"unhandled command: {args.command}")  # argparse requires one


if __name__ == "__main__":
    sys.exit(main())
