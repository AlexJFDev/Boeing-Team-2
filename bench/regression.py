"""Regression task set and baseline scores (WBS 1.25, issue #33).

A fixed set of tasks, each answered by fixed canned responses, scored by the
benchmark's own scorers through a normal Inspect eval. The scores are saved once as
a baseline file. Later runs must reproduce that baseline exactly; any difference
fails and is listed. This is what tells us a harness or scorer change altered scoring
(the CI gate that runs it on every PR is WBS 1.26).

No model is involved. A replay solver puts each canned response in place of a model
reply, so the run is deterministic and needs no server or Docker.

The regression set (``REGRESSION_VERSION``) is:

- every task in the A1 and A2 v1.0.0 task files (the frozen corpus, pinned by path;
  each file is verified against its manifest when loaded),
- answered five ways (``RESPONSE_KINDS``), each derived from the task's own target:

      reference        the target itself: the known-correct reference answer
      reformatted      the same JSON, pretty-printed
      prose_wrapped    the target JSON inside a sentence
      one_field_wrong  one field changed; for "cannot be determined" a confident made-up
                       position, and for a rejected frame an "accept"
      unparseable      a sentence with no JSON

- scored by the A-family scorers in ``SCORERS``.

The baseline (``data/regression/baseline_vX.Y.Z.json``) records every case's score from
every scorer, the scorer versions, the corpus hash of each task file and the reference
config. It is listed in ``data/regression/MANIFEST.json``, so an in-place edit is refused
like any frozen corpus file (CLAUDE.md rule 4). A new baseline is a new version
(refresh procedure: WBS 1.29).

Command line (exit status 1 if anything differs from the baseline):

    python -m bench.regression show       # print the recorded baseline scores
    python -m bench.regression check      # run the regression set and compare
    python -m bench.regression check --log logs/<file>.eval    # compare an existing run
    python -m bench.regression record 1.0.1                    # save a new baseline version

The same task runs under Inspect directly:

    PYTHONPATH=. inspect eval tasks/regression_set.py --model mockllm/model
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import inspect_ai
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from bench.corpus import REPO_ROOT, CorpusError, add_to_manifest, relpath, task_corpus, verify_file
from bench.loader import load_tasks
from bench.scorers import (
    ADSB_SCORER_VERSION,
    EXACT_MATCH_VERSION,
    adsb_field_capability_scorer,
    deterministic_exact_match,
)

REGRESSION_VERSION = "1.0.0"
BASELINE_VERSION = "1.0.0"  # the baseline `check` compares against
REGRESSION_DIR = REPO_ROOT / "data" / "regression"

# Pinned by path, not taken from bench.adsb.a1/a2: a new task version must not silently
# change what the regression set measures. Moving to new task files is a new
# REGRESSION_VERSION and a new baseline.
TASK_FILES: tuple[Path, ...] = (
    REPO_ROOT / "data" / "tasks" / "a1_adsb_decode_v1.0.0.jsonl",
    REPO_ROOT / "data" / "tasks" / "a2_cpr_position_v1.0.0.jsonl",
)
RESPONSE_KINDS = ("reference", "reformatted", "prose_wrapped", "one_field_wrong", "unparseable")

# Scorer name in the EvalLog -> (scorer factory, version recorded in the baseline).
SCORERS: dict[str, tuple[Callable[[], Scorer], str]] = {
    "deterministic_exact_match": (deterministic_exact_match, EXACT_MATCH_VERSION),
    "adsb_field_capability_scorer": (adsb_field_capability_scorer, ADSB_SCORER_VERSION),
}

# The run that produces the baseline. No model call is made, but the sampling
# settings are still pinned and recorded (CLAUDE.md rule 3).
REFERENCE_CONFIG = {"model": "mockllm/model", "temperature": 0, "seed": 42}

CASE_SEPARATOR = "::"
UNPARSEABLE_TEXT = "I could not decode this message."
WRONG_POSITION = {"latitude": 0.0, "longitude": 0.0}


class RegressionError(ValueError):
    """The baseline is missing or unreadable, or a run could not be compared."""


def baseline_path(version: str = BASELINE_VERSION) -> Path:
    return REGRESSION_DIR / f"baseline_v{version}.json"


# ---------------------------------------------------------------------------
# The regression set
# ---------------------------------------------------------------------------


def _perturb(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return round(value + 0.01, 5)
    text = str(value)
    return text[:-1] + ("1" if text.endswith("0") else "0") if text else "0"


def response_for(target: str, kind: str) -> str:
    """The canned response of one kind for one task, derived from its target only."""
    obj = json.loads(target)
    if kind == "reference":
        return target
    if kind == "reformatted":
        return json.dumps(obj, indent=2)
    if kind == "prose_wrapped":
        return f"Here is my answer: {target} Let me know if you need more detail."
    if kind == "unparseable":
        return UNPARSEABLE_TEXT
    if kind == "one_field_wrong":
        if obj.get("answer") == "cannot_be_determined":
            return json.dumps(WRONG_POSITION, separators=(",", ":"))
        if "answer" in obj:
            return json.dumps({**obj, "answer": "accept"}, separators=(",", ":"), sort_keys=True)
        key = sorted(obj)[0]
        return json.dumps({**obj, key: _perturb(obj[key])}, separators=(",", ":"), sort_keys=True)
    raise ValueError(f"unknown response kind {kind!r}")


def case_id(task_id: str, kind: str) -> str:
    return f"{task_id}{CASE_SEPARATOR}{kind}"


def regression_samples() -> list[Sample]:
    """One sample per (task, response kind), in a fixed order."""
    samples = []
    for task_file in TASK_FILES:
        for sample in load_tasks(task_file):   # verifies the file against its manifest
            target = sample.target if isinstance(sample.target, str) else sample.target[0]
            for kind in RESPONSE_KINDS:
                samples.append(Sample(
                    id=case_id(str(sample.id), kind),
                    input=sample.input,
                    target=sample.target,
                    metadata={**(sample.metadata or {}), "task_id": str(sample.id),
                              "task_file": relpath(task_file), "regression_kind": kind,
                              "regression_response": response_for(target, kind)},
                ))
    return samples


@solver
def replay_response() -> Solver:
    """Answer with the sample's canned response instead of calling a model."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        text = state.metadata["regression_response"]
        state.output = ModelOutput.from_content(model=str(state.model), content=text)
        state.messages.append(state.output.message)
        return state

    return solve


def regression_task() -> Task:
    return Task(
        dataset=regression_samples(),
        solver=replay_response(),
        scorer=[factory() for factory, _ in SCORERS.values()],
        version=REGRESSION_VERSION,
        metadata={"regression_version": REGRESSION_VERSION},
    )


def run_regression(log_dir: str | Path | None = None) -> EvalLog:
    """Run the regression set with the reference config and return the EvalLog."""
    with tempfile.TemporaryDirectory() as tmp:
        [log] = inspect_eval(
            regression_task(), model=REFERENCE_CONFIG["model"],
            temperature=REFERENCE_CONFIG["temperature"], seed=REFERENCE_CONFIG["seed"],
            log_dir=str(log_dir or tmp), display="none",
        )
        # Samples are read lazily from the .eval file, so load them before the
        # temporary directory is removed.
        log = read_eval_log(log.location)
    if log.status != "success" or not log.samples:
        raise RegressionError(f"regression run failed: {log.error}")
    return log


# ---------------------------------------------------------------------------
# Baseline: build, save, load
# ---------------------------------------------------------------------------


def scores_from_log(log: EvalLog) -> dict[str, dict[str, float]]:
    """Case id -> scorer name -> score, for every sample in the log."""
    cases: dict[str, dict[str, float]] = {}
    for sample in log.samples or []:
        scores = sample.scores or {}
        cases[str(sample.id)] = {name: float(scores[name].value) for name in SCORERS if name in scores}
    return dict(sorted(cases.items()))


def summarize(cases: dict[str, dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    """Scorer -> response kind -> {"n": count, "mean": mean score}."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cid, scores in cases.items():
        kind = cid.rsplit(CASE_SEPARATOR, 1)[-1]
        for name, value in scores.items():
            by[name][kind].append(value)
    return {name: {kind: {"n": len(v), "mean": round(sum(v) / len(v), 6)}
                   for kind, v in sorted(kinds.items(), key=lambda kv: RESPONSE_KINDS.index(kv[0]))}
            for name, kinds in sorted(by.items())}


def current_setup() -> dict[str, Any]:
    """Everything besides the scores that must match for a comparison to mean anything."""
    return {
        "regression_version": REGRESSION_VERSION,
        "response_kinds": list(RESPONSE_KINDS),
        "scorers": {name: version for name, (_, version) in SCORERS.items()},
        "corpus": {relpath(f): task_corpus(f)["corpus_hash"] for f in TASK_FILES},
        "reference_config": REFERENCE_CONFIG,
    }


def build_baseline(log: EvalLog, version: str) -> dict[str, Any]:
    cases = scores_from_log(log)
    return {
        "baseline_version": version,
        **current_setup(),
        "inspect_ai_version": inspect_ai.__version__,   # recorded, not compared
        "summary": summarize(cases),
        "cases": cases,
    }


def write_baseline(baseline: dict[str, Any], path: Path) -> str:
    """Save a new baseline file and list it in the manifest. Never overwrites."""
    if path.exists():
        raise RegressionError(f"{path.name} already exists; a new baseline is a new version")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
    return add_to_manifest(path)


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    path = path or baseline_path()
    try:
        verify_file(path)   # an in-place edit fails here
    except CorpusError as exc:
        raise RegressionError(str(exc)) from None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Difference:
    case: str
    scorer: str
    baseline: float | None
    current: float | None

    def __str__(self) -> str:
        if self.baseline is None:
            return f"{self.case} [{self.scorer}]: not in the baseline (now {self.current:g})"
        if self.current is None:
            return f"{self.case} [{self.scorer}]: missing (baseline {self.baseline:g})"
        direction = "LOWER" if self.current < self.baseline else "HIGHER"
        return f"{self.case} [{self.scorer}]: {self.baseline:g} -> {self.current:g} ({direction})"


def setup_differences(baseline: dict[str, Any], setup: dict[str, Any]) -> list[str]:
    """Version and corpus changes that make the baseline the wrong one to compare to."""
    problems = []
    for key in ("regression_version", "response_kinds", "reference_config"):
        if baseline.get(key) != setup[key]:
            problems.append(f"{key} changed: baseline {baseline.get(key)!r}, now {setup[key]!r}")
    for name, version in setup["scorers"].items():
        old = baseline.get("scorers", {}).get(name)
        if old != version:
            problems.append(f"scorer {name} version changed: baseline {old}, now {version}")
    for path, digest in setup["corpus"].items():
        old = baseline.get("corpus", {}).get(path)
        if old != digest:
            problems.append(f"corpus of {path} changed: baseline {old}, now {digest}")
    return problems


def score_differences(baseline_cases: dict[str, dict[str, float]],
                      current_cases: dict[str, dict[str, float]]) -> list[Difference]:
    diffs = []
    for cid in sorted(set(baseline_cases) | set(current_cases)):
        old, new = baseline_cases.get(cid, {}), current_cases.get(cid, {})
        for name in sorted(set(old) | set(new)):
            if old.get(name) != new.get(name):
                diffs.append(Difference(cid, name, old.get(name), new.get(name)))
    return diffs


@dataclass
class CheckResult:
    setup_problems: list[str]
    differences: list[Difference]
    compared: int

    @property
    def passed(self) -> bool:
        return not self.setup_problems and not self.differences


def check(log: EvalLog | None = None, baseline: dict[str, Any] | None = None) -> CheckResult:
    """Compare a regression run (a fresh one by default) with the baseline."""
    baseline = baseline if baseline is not None else load_baseline()
    log = log or run_regression()
    current = scores_from_log(log)
    compared = sum(len(v) for v in current.values())
    return CheckResult(setup_differences(baseline, current_setup()),
                       score_differences(baseline["cases"], current), compared)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

SHORT_NAMES = {"deterministic_exact_match": "exact_match", "adsb_field_capability_scorer": "field_scorer"}


def summary_table(summary: dict[str, dict[str, dict[str, float]]]) -> str:
    names = list(summary)
    header = f"{'response kind':<17}" + "".join(f"{SHORT_NAMES.get(n, n):>14}" for n in names) + f"{'cases':>8}"
    lines = [header, "-" * len(header)]
    for kind in RESPONSE_KINDS:
        row = [summary[n].get(kind, {}) for n in names]
        if not any(row):
            continue
        n_cases = next((r["n"] for r in row if r), 0)
        lines.append(f"{kind:<17}" + "".join(f"{r.get('mean', float('nan')):>14.3f}" for r in row)
                     + f"{n_cases:>8}")
    return "\n".join(lines)


def _show(baseline: dict[str, Any], path: Path) -> None:
    print(f"baseline {relpath(path)} (regression set {baseline['regression_version']})")
    for f, digest in baseline["corpus"].items():
        print(f"  corpus {f}: {digest[:16]}...")
    print("  scorers: " + ", ".join(f"{n} {v}" for n, v in baseline["scorers"].items()))
    print()
    print(summary_table(baseline["summary"]))
    print(f"\n{len(baseline['cases'])} cases, "
          f"{sum(len(v) for v in baseline['cases'].values())} scores (mean score per response kind)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.regression",
                                     description="Regression task set and baseline scores (WBS 1.25).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print the recorded baseline scores")
    p_check = sub.add_parser("check", help="run the regression set and compare with the baseline")
    p_check.add_argument("--log", help="compare this .eval log instead of running")
    p_check.add_argument("--log-dir", help="keep the run's .eval log here")
    p_record = sub.add_parser("record", help="run the regression set and save a NEW baseline version")
    p_record.add_argument("version", help="baseline version, e.g. 1.0.1")
    args = parser.parse_args(argv)

    try:
        if args.command == "show":
            path = baseline_path()
            _show(load_baseline(path), path)
            return 0
        if args.command == "record":
            path = baseline_path(args.version)
            baseline = build_baseline(run_regression(), args.version)
            sha = write_baseline(baseline, path)
            print(f"saved {relpath(path)} (sha256 {sha}); listed in {relpath(path.parent / 'MANIFEST.json')}")
            print(summary_table(baseline["summary"]))
            if args.version != BASELINE_VERSION:
                print(f"note: check still compares against {BASELINE_VERSION}; "
                      f"set BASELINE_VERSION to {args.version} to switch")
            return 0
        log = read_eval_log(args.log) if args.log else run_regression(args.log_dir)
        result = check(log)
    except (RegressionError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for problem in result.setup_problems:
        print(f"SETUP CHANGED: {problem}")
    for diff in result.differences:
        print(f"CHANGED: {diff}")
    if result.passed:
        print(f"PASS: all {result.compared} scores match baseline {BASELINE_VERSION}")
        return 0
    print(f"FAIL: {len(result.differences)} score(s) and {len(result.setup_problems)} setup "
          f"item(s) differ from baseline {BASELINE_VERSION}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
