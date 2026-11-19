"""Task schema validator (WBS 1.28, issue #36; schema from WBS 1.12/1.13).

Checks every task file in data/tasks against the task schema the loader enforces
(bench/loader.py) and against the corpus manifest (bench/corpus.py). CI runs it on
every pull request (.github/workflows/corpus-checks.yml), so a task that would stop a
run is caught before it is merged, not when someone runs the suite.

Unlike the loader, which stops at the first bad record, this reports every problem in
every file:

    V1  a line is not valid JSON, or not a JSON object
    V2  a record breaks the schema: a required field (id, family, preamble, source,
        input, target) is missing or empty, the family is not A-D, or premises /
        metadata have the wrong type. The message names the field (loader rules).
    V3  a task ID is used twice in the same file
    V4  a record names a preamble that does not exist in data/preambles
    V5  the file has no tasks
    V6  the file name carries no version (expected <name>_vX.Y.Z.jsonl)
    M1  the task file, or a preamble it names, is not listed in its MANIFEST.json or
        does not match it (an unlisted new file, or a frozen file edited in place)
    M2  data/tasks/MANIFEST.json lists a file that does not exist

The data boundary checker (WBS 1.27, bench/boundary.py) checks where the data came
from; this checks that it is well formed. CI runs both.

Command line (exit status 0 pass, 1 a problem was found, 2 a path does not exist):

    python -m bench.validate                         # every task file in data/tasks
    python -m bench.validate data/tasks/x_v1.0.0.jsonl
    python -m bench.validate --annotations           # also print GitHub annotations
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bench.boundary import github_annotation
from bench.corpus import MANIFEST_NAME, VERSIONED_NAME, CorpusError, manifest_entries, verify_file
from bench.loader import PREAMBLE_DIR, REPO_ROOT, load_preamble, record_problems

VALIDATOR_VERSION = "1.0.0"

TASK_DIR = REPO_ROOT / "data" / "tasks"
TASK_SUFFIX = ".jsonl"


@dataclass(frozen=True)
class Problem:
    path: str
    line: int | None
    code: str
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: [{self.code}] {self.message}"


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    files_checked: int = 0
    tasks_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.problems

    def failed_files(self) -> list[str]:
        return sorted({p.path for p in self.problems})

    def summary(self) -> str:
        counts = f"{self.files_checked} task files, {self.tasks_checked} tasks"
        if self.passed:
            return f"PASS: task schema check ({counts})"
        return (f"FAIL: {len(self.problems)} problem(s) in {len(self.failed_files())} file(s) "
                f"({counts})")


def display(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo (what GitHub annotations need)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def validate_file(path: Path, report: Report, preamble_dir: Path = PREAMBLE_DIR) -> None:
    """Check one task file and add what is wrong to ``report``."""
    name = display(path)
    report.files_checked += 1
    found: list[Problem] = []

    if not VERSIONED_NAME.search(path.name):
        found.append(Problem(name, None, "V6", "file name has no version (expected <name>_vX.Y.Z.jsonl)"))

    seen: set[str] = set()
    preambles: dict[str, int] = {}   # preamble name -> first line that names it
    tasks = 0
    text = path.read_bytes().decode("utf-8", errors="replace")
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            found.append(Problem(name, n, "V1", "line is not valid JSON"))
            continue
        if not isinstance(record, dict):
            found.append(Problem(name, n, "V1", "line is not a JSON object"))
            continue
        tasks += 1
        task_id = record.get("id")
        who = f"task {task_id!r}: " if isinstance(task_id, str) and task_id else ""
        found += [Problem(name, n, "V2", who + p) for p in record_problems(record)]
        if isinstance(task_id, str) and task_id:
            if task_id in seen:
                found.append(Problem(name, n, "V3", f"duplicate task id {task_id!r}"))
            seen.add(task_id)
        preamble = record.get("preamble")
        if isinstance(preamble, str) and preamble.strip() and preamble not in preambles:
            preambles[preamble] = n
            try:
                load_preamble(preamble, preamble_dir)
            except FileNotFoundError:
                found.append(Problem(name, n, "V4",
                                     f"{who}preamble {preamble!r} does not exist in {display(preamble_dir)}"))
                continue
            try:
                verify_file(preamble_dir / f"{preamble}.txt")
            except CorpusError as exc:
                found.append(Problem(name, n, "M1", f"preamble {preamble!r}: {exc}"))

    report.tasks_checked += tasks
    if tasks == 0:
        found.append(Problem(name, None, "V5", "file has no tasks"))
    try:
        verify_file(path)
    except CorpusError as exc:
        found.append(Problem(name, None, "M1", str(exc)))
    report.problems += found


def manifest_problems(task_dir: Path = TASK_DIR) -> list[Problem]:
    """Files the task manifest lists that are not there."""
    manifest = task_dir / MANIFEST_NAME
    try:
        entries = manifest_entries(manifest)
    except CorpusError as exc:
        return [Problem(display(manifest), None, "M2", str(exc))]
    return [Problem(display(manifest), None, "M2", f"lists {name}, which does not exist")
            for name in sorted(entries) if not (task_dir / name).is_file()]


def task_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.is_file() and f.suffix == TASK_SUFFIX)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    return files


def validate(paths: Iterable[str | Path] | None = None, preamble_dir: Path = PREAMBLE_DIR) -> Report:
    """Validate the given task files and directories (default: every file in data/tasks)."""
    roots = [Path(p) for p in paths] if paths else [TASK_DIR]
    report = Report()
    for f in task_files(roots):
        validate_file(f, report, preamble_dir)
    for d in {r for r in roots if r.is_dir()}:
        if (d / MANIFEST_NAME).exists() or not paths:
            report.problems += manifest_problems(d)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.validate",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="task files or directories (default: data/tasks)")
    parser.add_argument("--annotations", action="store_true",
                        help="also print each problem as a GitHub Actions error annotation")
    args = parser.parse_args(argv)
    try:
        report = validate(args.paths)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for p in report.problems:
        print(p)
        if args.annotations:
            print(github_annotation(p.path, p.line, p.code, p.message, "Task schema"))
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
