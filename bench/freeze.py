"""Freeze protocol corpus v1.0 (WBS 3.11, issue #66).

Protocol corpus v1.0 is the six ADS-B and ARINC 429 task families (A1, A2, B1-B4) at
their current task files, with the preambles they name. Freezing it writes a record
that is never changed again, ``data/frozen/protocol_v1.0.0.json``, and tags the commit
that adds it:

    python -m bench.freeze check      # run every freeze check, write nothing
    python -m bench.freeze freeze     # check, then write the frozen record
    git add data/frozen && git commit # commit the record (through a PR, like any change)
    python -m bench.freeze tag        # annotated tag corpus-protocol-v1.0.0 on that commit
    git push origin corpus-protocol-v1.0.0
    python -m bench.freeze verify     # every frozen record still matches its files (CI)

The freeze is blocked, and nothing is written, unless every check passes:

    manifests    every corpus file matches its MANIFEST.json entry (bench.corpus)
    schema       the task files pass the task schema check (bench.validate)
    boundary     the task files and preambles pass the data-boundary check (bench.boundary)
    docs         docs/corpus.md documents every family, source and oracle (bench.corpus_doc)
    irr          a pilot on the current judge-clause rubric passed (bench.irr gate, WBS 3.9)
    reviews      no open GitHub issue labelled ``task-review`` is on a task in the corpus

A review issue is "on a task" when its title or body names a task ID in the corpus
(``b3-07``), a corpus task file, or a family ID (``A2``). An issue naming no task at all
also blocks, since it can't be ruled out; one naming only tasks outside this corpus
does not. If the issues can't be read (no network, rate limit), the freeze is blocked:
pass ``--review-issues FILE`` (a JSON list of {number, title, body}) to check offline.

The record holds the corpus hash (one sha256 over every file, ``bench.corpus`` scheme),
each family's run-level corpus hash (the one its EvalLogs record, so a log can be
matched to the frozen corpus), the files and their sha256, the task, scorer and oracle
versions, the sha256 of docs/corpus.md, and the checks that passed. A record is never
overwritten: a change after the freeze is a new corpus version (CLAUDE.md rule 4).

Exit codes: 0 pass, 1 blocked or a mismatch, 2 an error (bad arguments, a git failure).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.corpus import CORPUS_VERSION, CorpusError, corpus_hash, relpath, task_corpus
from bench.provenance import sha256_file

FREEZE_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = REPO_ROOT / "data" / "frozen"
DOC = REPO_ROOT / "docs" / "corpus.md"
CORPUS_NAME = "protocol"
CORPUS_RELEASE = "1.0.0"
REVIEW_LABEL = "task-review"
DEFAULT_REPO = "AlexJFDev/Boeing-Team-2"

TASK_LIKE = re.compile(r"(?<![\w-])[a-z]\d+-[a-z0-9]+(?:-[a-z0-9]+)*(?![\w-])|\b[A-D]\d\b|_v\d+\.\d+\.\d+\.jsonl\b",
                       re.IGNORECASE)


class FreezeError(RuntimeError):
    """The freeze cannot go ahead for a reason that is not a failed check (exit 2)."""


@dataclass(frozen=True)
class Family:
    id: str
    task_file: Path
    task_version: str
    capability_scorer: str  # name==version
    safety_scorer: str | None


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def protocol_families() -> list[Family]:
    """The families in protocol corpus v1.0, at their current task files and scorers."""
    from bench import safety, scorers
    from bench.adsb import a1, a2
    from bench.arinc429 import b1, b2, b3, b4

    adsb = f"adsb_field_capability_scorer=={scorers.ADSB_SCORER_VERSION}"
    coords = f"a2_cpr_coordinate_scorer=={scorers.A2_COORDINATE_SCORER_VERSION}"
    arinc = f"arinc429_bnr_bcd_capability_scorer=={scorers.ARINC429_SCORER_VERSION}"
    unsafe = f"unsafe_if_rule_scorer=={safety.SAFETY_SCORER_VERSION}"
    return [
        Family("A1", a1.TASK_FILE, a1.A1_TASK_VERSION, adsb, unsafe),
        Family("A2", a2.TASK_FILE, a2.A2_TASK_VERSION, coords, unsafe),
        Family("B1", b1.TASK_FILE, b1.B1_TASK_VERSION, arinc, None),
        Family("B2", b2.TASK_FILE, b2.B2_TASK_VERSION, arinc, None),
        Family("B3", b3.TASK_FILE, b3.B3_TASK_VERSION, arinc, None),
        Family("B4", b4.TASK_FILE, b4.B4_VERSION, f"b4_triage_scorer=={b4.B4_SCORER_VERSION}", unsafe),
    ]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def show(path: Path) -> str:
    """A path for messages: repo-relative when inside the repository, else as given."""
    try:
        return relpath(path)
    except ValueError:
        return str(path)


def record_path(out_dir: Path = FROZEN_DIR, name: str = CORPUS_NAME, release: str = CORPUS_RELEASE) -> Path:
    return out_dir / f"{name}_v{release}.json"


def tag_name(name: str = CORPUS_NAME, release: str = CORPUS_RELEASE) -> str:
    return f"corpus-{name}-v{release}"


# ---------------------------------------------------------------------------
# Review issues
# ---------------------------------------------------------------------------


def fetch_review_issues(repo: str = DEFAULT_REPO, label: str = REVIEW_LABEL, timeout: float = 20) -> list[dict[str, Any]]:
    """Open GitHub issues with the review label (pull requests left out). Raises on failure."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "boeing-team-2-freeze"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    issues: list[dict[str, Any]] = []
    for page in range(1, 51):
        url = (f"https://api.github.com/repos/{repo}/issues?state=open&labels={label}"
               f"&per_page=100&page={page}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            batch = json.loads(response.read().decode("utf-8"))
        issues += [{"number": i["number"], "title": i.get("title") or "", "body": i.get("body") or "",
                    "url": i.get("html_url", "")} for i in batch if "pull_request" not in i]
        if len(batch) < 100:
            break
    return issues


def issue_targets(issue: dict[str, Any], task_ids: set[str], files: set[str], families: set[str]) -> tuple[list[str], bool]:
    """(the corpus tasks, files and families the issue names; whether it names anything task-like)."""
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"
    named = [t for t in sorted(task_ids)
             if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", text, re.IGNORECASE)]
    named += [f for f in sorted(files) if f in text]
    named += [f for f in sorted(families) if re.search(rf"\b{f}\b", text)]
    return named, bool(TASK_LIKE.search(text))


def review_blockers(issues: list[dict[str, Any]], families: list[Family]) -> list[tuple[dict[str, Any], list[str]]]:
    """The open review issues that block this corpus, each with what it names."""
    task_ids = {str(r["id"]) for f in families for r in _rows(f.task_file)}
    files = {f.task_file.name for f in families}
    ids = {f.id for f in families}
    blocking = []
    for issue in issues:
        named, task_like = issue_targets(issue, task_ids, files, ids)
        if named or not task_like:
            blocking.append((issue, named or ["(names no task)"]))
    return blocking


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run_checks(families: list[Family], *, issues: list[dict[str, Any]] | None = None,
               issues_source: str = "github", irr_dir: Path | None = None,
               repo: str = DEFAULT_REPO, fetch: Callable[..., list[dict[str, Any]]] = fetch_review_issues) -> list[Check]:
    """Every freeze check, in order. Never raises for a failed check."""
    from bench import boundary, corpus_doc, irr, validate

    task_files = [f.task_file for f in families]
    checks: list[Check] = []

    bad = []
    for path in task_files:
        try:
            task_corpus(path)
        except CorpusError as exc:
            bad.append(str(exc))
    checks.append(Check("manifests", not bad, bad[0] if bad else
                        f"{len(task_files)} task files and their preambles match their manifests"))

    report = validate.validate(task_files)
    checks.append(Check("schema", report.passed, report.summary() if report.passed else
                        f"{report.summary()}: {report.problems[0]}"))

    preambles = sorted({REPO_ROOT / p for path in task_files for p in task_corpus_files(path)
                        if p.startswith("data/preambles/")})
    breport = boundary.check_paths([*task_files, *preambles])
    checks.append(Check("boundary", breport.passed, breport.summary() if breport.passed else
                        f"{breport.summary()}: {breport.violations[0]}"))

    problems = corpus_doc.check()
    checks.append(Check("docs", not problems, "docs/corpus.md covers every task file, source and oracle"
                        if not problems else f"{len(problems)} problem(s): {problems[0]}"))

    ok, reason = irr.gate(irr_dir) if irr_dir is not None else irr.gate()
    checks.append(Check("irr", ok, reason))

    if issues is None:
        try:
            issues = fetch(repo)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            checks.append(Check("reviews", False,
                                f"could not read the open '{REVIEW_LABEL}' issues of {repo} ({exc}); the freeze "
                                "stays blocked until they can be checked (or pass --review-issues FILE)"))
            return checks
    blocking = review_blockers(issues, families)
    if blocking:
        listed = "; ".join(f"#{i['number']} {i['title']!r} ({', '.join(named)})" for i, named in blocking)
        checks.append(Check("reviews", False, f"{len(blocking)} open review issue(s) on corpus tasks: {listed}"))
    else:
        checks.append(Check("reviews", True, f"no open '{REVIEW_LABEL}' issue on a corpus task "
                            f"({len(issues)} open review issue(s) checked, from {issues_source})"))
    return checks


def task_corpus_files(task_file: Path) -> list[str]:
    try:
        return list(task_corpus(task_file)["corpus_files"])
    except CorpusError:
        return []


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _oracles(rows: list[dict[str, Any]]) -> list[str]:
    from bench.corpus_doc import ORACLE_KEYS

    found = set()
    for row in rows:
        meta = row.get("metadata") or {}
        for key, cite in ORACLE_KEYS.items():
            if isinstance(meta.get(key), str):
                found.add(cite(meta[key]))
    return sorted(found)


def _git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def build_record(families: list[Family], checks: list[Check], *, issues_source: str,
                 name: str = CORPUS_NAME, release: str = CORPUS_RELEASE) -> dict[str, Any]:
    files: dict[str, str] = {}
    entries: dict[str, Any] = {}
    for fam in families:
        corpus = task_corpus(fam.task_file)
        files.update(corpus["corpus_files"])
        rows = _rows(fam.task_file)
        entries[fam.id] = {
            "task_file": relpath(fam.task_file),
            "task_version": fam.task_version,
            "tasks": len(rows),
            "run_corpus_hash": corpus["corpus_hash"],
            "run_corpus_files": sorted(corpus["corpus_files"]),
            "sources": sorted({str(r.get("source")) for r in rows}),
            "oracles": _oracles(rows),
            "capability_scorer": fam.capability_scorer,
            "safety_scorer": fam.safety_scorer,
        }
    try:
        commit = _git(["rev-parse", "HEAD"])
    except FreezeError:
        commit = None
    return {
        "corpus": name,
        "release": release,
        "corpus_hash": corpus_hash(files),
        "corpus_hash_scheme": CORPUS_VERSION,
        "tag": tag_name(name, release),
        "tasks": sum(e["tasks"] for e in entries.values()),
        "families": entries,
        "files": dict(sorted(files.items())),
        "documentation": {relpath(DOC): sha256_file(DOC)},
        "checks": [asdict(c) for c in checks],
        "review_label": REVIEW_LABEL,
        "review_issues_from": issues_source,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_from_commit": commit,
        "freeze_version": FREEZE_VERSION,
    }


def freeze(*, out_dir: Path = FROZEN_DIR, issues: list[dict[str, Any]] | None = None, issues_source: str = "github",
           irr_dir: Path | None = None, repo: str = DEFAULT_REPO,
           fetch: Callable[..., list[dict[str, Any]]] = fetch_review_issues) -> tuple[list[Check], dict[str, Any] | None]:
    """Run the checks; if all pass, write and return the record. Never overwrites one."""
    path = record_path(out_dir)
    if path.exists():
        raise FreezeError(f"{show(path)} already exists: protocol corpus v{CORPUS_RELEASE} is frozen and its "
                          "record is never rewritten. A change is a new corpus version.")
    families = protocol_families()
    checks = run_checks(families, issues=issues, issues_source=issues_source, irr_dir=irr_dir, repo=repo, fetch=fetch)
    if not all(c.ok for c in checks):
        return checks, None
    record = build_record(families, checks, issues_source=issues_source)
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checks, record


# ---------------------------------------------------------------------------
# Verify and tag
# ---------------------------------------------------------------------------


def verify(out_dir: Path = FROZEN_DIR, files_root: Path = REPO_ROOT) -> list[str]:
    """Problems with any frozen record: a file changed or gone, or a hash that doesn't recompute."""
    problems: list[str] = []
    for path in sorted(out_dir.glob("*_v*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            files: dict[str, str] = record["files"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{show(path)}: unreadable frozen record ({exc})")
            continue
        for rel, sha in files.items():
            target = files_root / rel
            if not target.is_file():
                problems.append(f"{show(path)}: frozen file {rel} is missing")
            elif sha256_file(target) != sha:
                problems.append(f"{show(path)}: frozen file {rel} has changed since the freeze "
                                f"(sha256 {sha256_file(target)[:12]}..., frozen {sha[:12]}...)")
        if corpus_hash(files) != record.get("corpus_hash"):
            problems.append(f"{show(path)}: corpus_hash does not match its file list")
        for fam, entry in (record.get("families") or {}).items():
            # The run-level hash covers the family's task file and the preambles it names.
            subset = {k: files.get(k, "") for k in entry.get("run_corpus_files", [])}
            if not subset or corpus_hash(subset) != entry.get("run_corpus_hash"):
                problems.append(f"{show(path)}: {fam} run_corpus_hash does not recompute")
    return problems


def tag(out_dir: Path = FROZEN_DIR, git_root: Path = REPO_ROOT, files_root: Path = REPO_ROOT) -> str:
    """Put the annotated tag on HEAD, which must hold the committed, unchanged record. Returns the tag."""
    path = record_path(out_dir)
    if not path.is_file():
        raise FreezeError(f"no frozen record at {path}; run `python -m bench.freeze freeze` first")
    record = json.loads(path.read_text(encoding="utf-8"))
    rel = path.resolve().relative_to(git_root.resolve()).as_posix()
    try:
        committed = _git(["show", f"HEAD:{rel}"], cwd=git_root)
    except FreezeError as exc:
        raise FreezeError(f"{rel} is not committed at HEAD; commit it, then tag") from exc
    if json.loads(committed) != record:
        raise FreezeError(f"{rel} differs from the committed copy at HEAD; commit or restore it, then tag")
    mismatches = [p for p in verify(out_dir, files_root) if p.startswith(show(path)) or rel in p]
    if mismatches:
        raise FreezeError("the frozen record no longer matches its files: " + "; ".join(mismatches))
    name = record["tag"]
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{name}"], cwd=git_root,
                      capture_output=True).returncode == 0:
        raise FreezeError(f"tag {name} already exists; a corpus tag is never moved")
    message = (f"Protocol corpus v{record['release']} frozen (WBS 3.11)\n\n"
               f"corpus_hash: {record['corpus_hash']}\n"
               f"record: {rel}\n"
               f"families: {', '.join(record['families'])}; tasks: {record['tasks']}; files: {len(record['files'])}\n")
    _git(["tag", "-a", name, "-m", message, "HEAD"], cwd=git_root)
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_checks(checks: list[Check]) -> None:
    for c in checks:
        print(f"  {'PASS' if c.ok else 'FAIL':<5} {c.name:<10} {c.detail}")


def _load_issues(path: Path | None) -> tuple[list[dict[str, Any]] | None, str]:
    if path is None:
        return None, "github"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise FreezeError(f"{path}: expected a JSON list of issues")
    return data, show(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.freeze", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("command", choices=("check", "freeze", "tag", "verify"))
    parser.add_argument("--review-issues", type=Path, help="JSON list of open review issues, instead of GitHub")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo for review issues (default {DEFAULT_REPO})")
    parser.add_argument("--irr-dir", type=Path, help="inter-rater pilot records (default data/irr)")
    parser.add_argument("--out-dir", type=Path, default=FROZEN_DIR, help="frozen records (default data/frozen)")
    parser.add_argument("--git-root", type=Path, default=REPO_ROOT, help="git repository to tag (default this one)")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            problems = verify(args.out_dir)
            for p in problems:
                print(p)
            count = len(list(args.out_dir.glob("*_v*.json"))) if args.out_dir.is_dir() else 0
            print(f"{'FAIL' if problems else 'PASS'}: frozen corpus check ({count} frozen record(s))")
            return 1 if problems else 0
        if args.command == "tag":
            name = tag(args.out_dir, args.git_root)
            print(f"Tagged {name} on {_git(['rev-parse', '--short', 'HEAD'], cwd=args.git_root)}. "
                  f"Publish it with: git push origin {name}")
            return 0
        issues, source = _load_issues(args.review_issues)
        if args.command == "check":
            checks = run_checks(protocol_families(), issues=issues, issues_source=source,
                                irr_dir=args.irr_dir, repo=args.repo)
            print(f"Freeze checks for protocol corpus v{CORPUS_RELEASE}:")
            _print_checks(checks)
            ok = all(c.ok for c in checks)
            print("READY to freeze" if ok else "BLOCKED: the freeze would not go ahead")
            return 0 if ok else 1
        checks, record = freeze(out_dir=args.out_dir, issues=issues, issues_source=source,
                                irr_dir=args.irr_dir, repo=args.repo)
        print(f"Freeze checks for protocol corpus v{CORPUS_RELEASE}:")
        _print_checks(checks)
        if record is None:
            print("BLOCKED: nothing was written")
            return 1
        print(f"FROZEN: {show(record_path(args.out_dir))}\n  corpus_hash {record['corpus_hash']}\n"
              f"  {record['tasks']} tasks in {len(record['families'])} families, {len(record['files'])} files\n"
              f"Next: commit the record, then `python -m bench.freeze tag` on that commit.")
        return 0
    except (FreezeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
