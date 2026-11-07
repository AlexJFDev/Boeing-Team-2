"""Synthetic data boundary checker (WBS 1.27, issue #35).

Checks that the corpus stays inside the project's data boundary (CLAUDE.md rule 1:
synthetic or sanitized data only). Two kinds of check:

Source (S codes), on JSON Lines records and JSON documents:

    S1  a task record has no source
    S2  a source is not one of APPROVED_SOURCES

    A task record is any record with the loader's task keys (id, family, input,
    target; see bench/loader.py), wherever the file lives. Any other record or
    document that declares a top-level ``source`` string must use an approved tag
    too. A ``source`` object (the ACARS manifest form) must list only approved tags
    in ``external_sources_used``.

Content (R and K codes), on every text file under the scanned paths:

    R1  a real-looking U.S. registration (N-number)
    R2  a real-looking hyphenated registration (forms like G-QXZQ, VH-QXZ, 9V-QXZ)
    K1  a key- or token-like string (private key, cloud or API token, assignment)

R1 follows the N-number grammar in 14 CFR 47.15(b): "N", then at most five symbols,
the first a digit 1-9, no letters I or O, and at most two letters, at the end (1-5
digits, 1-4 digits and a letter, or 1-3 digits and two letters). One-symbol marks
such as N1 and N2 are not flagged, because N1 and N2 are also engine spool speeds in
maintenance text; that skips a handful of real registrations.

R2 matches a one- or two-character nationality mark, a hyphen and three or four
letters. It is letters only after the hyphen so that document names such as DO-260B
are not flagged, which means all-digit marks (B-1234, RA-12345) are not caught.

A key is allowed when it contains SYNTHETIC (any case), so tasks can plant clearly
fake secrets for the tool-misuse probes (WBS 10.4).

These are pattern checks. They catch the common ways real data leaks in (a pasted
tail number, a real key); they cannot prove a file is synthetic. Generating data
from the spec is still what keeps real data out. Secret scanning on every commit
stays with gitleaks in CI (.github/workflows/secret-scan.yml).

Files that are not UTF-8 text (pcaps, images) are skipped and counted in the report.

Command line (exit status 1 if anything fails, 0 if all pass):

    python -m bench.boundary                 # scans data/ and prompts/
    python -m bench.boundary data/tasks      # or any files and directories
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

CHECKER_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("data", "prompts")

# Product backlog 1.27: spec-synthesized, OpenSky, airplanes.live, adsb.lol, Airframes.io.
APPROVED_SOURCES = ("spec-synthesized", "opensky", "airplanes.live", "adsb.lol", "airframes.io")

# The loader's required task keys, minus source (checked here) and preamble.
TASK_KEYS = frozenset({"id", "family", "input", "target"})

SYNTHETIC_MARKER = "SYNTHETIC"

_NOT_ALNUM_BEFORE = r"(?<![A-Za-z0-9])"
_NOT_ALNUM_AFTER = r"(?![A-Za-z0-9])"
_LETTER = "[A-HJ-NP-Z]"  # no I or O

US_REGISTRATION = re.compile(
    _NOT_ALNUM_BEFORE
    + rf"N(?:[1-9]\d{{1,4}}|[1-9]\d{{0,3}}{_LETTER}|[1-9]\d{{0,2}}{_LETTER}{{2}})"
    + _NOT_ALNUM_AFTER
)
HYPHEN_REGISTRATION = re.compile(
    r"(?<![A-Za-z0-9-])(?:[A-Z]{1,2}|[0-9][A-Z]|[A-Z][0-9])-[A-Z]{3,4}(?![A-Za-z0-9-])"
)
# Phonetic words and common compounds with the same shape as a registration.
NOT_REGISTRATIONS = frozenset({"X-RAY", "E-MAIL"})

KEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})")),
    ("API secret key", re.compile(r"(?<![A-Za-z0-9])sk-(?:[A-Za-z]+-)?[A-Za-z0-9_\-]{20,}")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}")),
    ("Slack token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}")),
    ("secret assignment", re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token"
        r"|password|passwd)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+_.\-]{16,}")),
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int | None
    code: str
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: [{self.code}] {self.message}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0
    task_records: int = 0
    binary_skipped: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def failed_files(self) -> list[str]:
        return sorted({v.path for v in self.violations})

    def summary(self) -> str:
        skipped = f", {len(self.binary_skipped)} binary files skipped" if self.binary_skipped else ""
        counts = f"{self.files_scanned} files, {self.task_records} task records{skipped}"
        if self.passed:
            return f"PASS: data boundary check ({counts})"
        return (f"FAIL: {len(self.violations)} problem(s) in {len(self.failed_files())} file(s) "
                f"({counts})")


# ---------------------------------------------------------------------------
# Source checks
# ---------------------------------------------------------------------------


def source_problem(value: Any) -> str | None:
    """Why a declared source is not acceptable, or None if it is."""
    if isinstance(value, dict):
        bad = [s for s in value.get("external_sources_used") or [] if s not in APPROVED_SOURCES]
        if bad:
            return f"unapproved external sources {bad}; approved: {', '.join(APPROVED_SOURCES)}"
        return None
    if value in APPROVED_SOURCES:
        return None
    return f"unapproved source {value!r}; approved: {', '.join(APPROVED_SOURCES)}"


def is_task_record(record: Any) -> bool:
    return isinstance(record, dict) and TASK_KEYS <= record.keys()


def check_record(record: Any, path: str, line: int | None) -> tuple[list[Violation], bool]:
    """Source violations for one record or document, and whether it is a task record."""
    if not isinstance(record, dict):
        return [], False
    task = is_task_record(record)
    where = f"task {record.get('id')!r}: " if task else ""
    if "source" not in record or record["source"] in (None, ""):
        return ([Violation(path, line, "S1", f"{where}no source")] if task else []), task
    problem = source_problem(record["source"])
    return ([Violation(path, line, "S2", where + problem)] if problem else []), task


# ---------------------------------------------------------------------------
# Content checks
# ---------------------------------------------------------------------------


def scan_line(text: str) -> Iterator[tuple[str, str]]:
    """(code, message) for each real-looking registration or key in one line of text."""
    for m in US_REGISTRATION.finditer(text):
        yield "R1", f"real-looking U.S. registration {m.group(0)!r}"
    for m in HYPHEN_REGISTRATION.finditer(text):
        if m.group(0) not in NOT_REGISTRATIONS:
            yield "R2", f"real-looking registration {m.group(0)!r}"
    for kind, rx in KEY_PATTERNS:
        for m in rx.finditer(text):
            if SYNTHETIC_MARKER not in m.group(0).upper():
                yield "K1", f"{kind}-like string {_redact(m.group(0))!r}"


def _redact(secret: str) -> str:
    return secret if len(secret) <= 8 else secret[:6] + "..." + f"({len(secret)} chars)"


def check_text(text: str, path: str) -> list[Violation]:
    return [Violation(path, n, code, msg)
            for n, line in enumerate(text.splitlines(), start=1)
            for code, msg in scan_line(line)]


def check_file(path: Path, report: Report, display: str | None = None) -> None:
    name = display or _display(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        report.binary_skipped.append(name)
        return
    report.files_scanned += 1
    report.violations += check_text(text, name)
    if path.suffix == ".jsonl":
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                report.violations.append(Violation(name, n, "J1", "line is not valid JSON"))
                continue
            found, task = check_record(record, name, n)
            report.violations += found
            report.task_records += task
    elif path.suffix == ".json":
        try:
            doc = json.loads(text)
        except ValueError:
            report.violations.append(Violation(name, None, "J1", "file is not valid JSON"))
            return
        found, task = check_record(doc, name, None)
        report.violations += found
        report.task_records += task


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.is_file()
                              and not any(part.startswith(".") for part in f.relative_to(p).parts))
        elif p.is_file():
            yield p
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")


def check_paths(paths: Iterable[str | Path] | None = None) -> Report:
    """Run every check on the given files and directories (default: data/ and prompts/)."""
    roots = [Path(p) for p in paths] if paths else [REPO_ROOT / d for d in DEFAULT_PATHS]
    report = Report()
    for f in _files(roots):
        check_file(f, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.boundary", description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files or directories (default: data/ and prompts/)")
    args = parser.parse_args(argv)
    try:
        report = check_paths(args.paths)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for v in report.violations:
        print(v)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
