"""Protocol corpus documentation check and lookup (WBS 3.10, issue #65).

``docs/corpus.md`` documents every task family in the protocol corpus: its data
source, source specification, oracle, scorer and scoring tolerance. This module
reads that page and the task files and fails when a task cites something the page
doesn't cover, so review fails in CI (the ``schema`` job of corpus-checks.yml).

    python -m bench.corpus_doc                    # check docs/corpus.md against data/tasks
    python -m bench.corpus_doc --family A2        # print one family's entry and its sources
    python -m bench.corpus_doc --annotations      # also print GitHub error annotations

What it checks:

    D1  a task family (a task file named ``<letter><digit>_..._vX.Y.Z.jsonl``) has no
        ``## <ID> · <name>`` section
    D2  a family entry is missing a required field (Task files, Data source, Source
        spec, Oracle, Scorer, Tolerance) or leaves it empty
    D3  a task file is not named in its family's Task files row
    D4  a task's ``source`` is not a key in the Sources table
    D5  a task's ``source`` is not named in its family's Data source row
    D6  an oracle a task cites in its metadata, as ``name==version``, is not named in
        its family's Oracle row (``decoder``, ``reference``, ``checker``;
        ``oracle_version`` is bench.adsb.oracle, ``verifier_version`` bench.adsb.cpr)
    D7  a key in a Data source or Source spec row is not in the Sources table
    D8  a Task files row names a file that does not exist
    D9  a scorer named in a Scorer row is not used by the family's task (tasks/<id>_*.py)

Cells are read through their backticked tokens, so prose around them is free.
``loader_demo`` is not part of the scored corpus and is skipped. Standard library
only, like bench/boundary.py, so CI needs no install to run it.

Exit codes: 0 pass, 1 a problem was found, 2 the page or task directory is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CHECKER_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "corpus.md"
TASKS_DIR = REPO_ROOT / "data" / "tasks"
TASK_CODE_DIR = REPO_ROOT / "tasks"

NOT_IN_CORPUS = ("loader_demo",)
REQUIRED_FIELDS = ("Task files", "Data source", "Source spec", "Oracle", "Scorer", "Tolerance")
KEYED_FIELDS = ("Data source", "Source spec")  # every token must be a Sources key

TASK_FILE = re.compile(r"^(?P<family>[a-z]\d+)_[a-z0-9_]+_v\d+\.\d+\.\d+\.jsonl$")
FAMILY_HEADING = re.compile(r"^##\s+(?P<family>[A-Z]\d+)\s+·\s+\S")
SOURCES_HEADING = re.compile(r"^##\s+Sources\s*$")
HEADING = re.compile(r"^#{1,2}\s")
TOKEN = re.compile(r"`([^`]+)`")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Metadata keys that name an oracle, and how each becomes a ``name==version`` citation.
ORACLE_KEYS = {
    "decoder": lambda v: v,
    "reference": lambda v: v,
    "checker": lambda v: v,
    "oracle_version": lambda v: f"bench.adsb.oracle=={v}",
    "verifier_version": lambda v: f"bench.adsb.cpr=={v}",
}


@dataclass
class Problem:
    code: str
    path: str
    line: int | None
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: [{self.code}] {self.message}"


@dataclass
class Cell:
    text: str
    line: int

    @property
    def tokens(self) -> list[str]:
        return TOKEN.findall(self.text)


@dataclass
class Entry:
    family: str
    title: str
    line: int
    fields: dict[str, Cell] = field(default_factory=dict)


@dataclass
class Doc:
    path: Path
    entries: dict[str, Entry]
    sources: dict[str, Cell]  # key -> the whole table row


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _cells(line: str) -> list[str] | None:
    """The cells of a pipe-table body row; None for a non-row or the separator row."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [c.strip() for c in stripped[1:-1].split("|")]
    if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
        return None
    return cells


def parse_doc(path: Path = DOC) -> Doc:
    """The family entries (field -> cell) and the Sources table of the corpus page."""
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: dict[str, Entry] = {}
    sources: dict[str, Cell] = {}
    current: Entry | None = None
    in_sources = False
    for n, line in enumerate(lines, start=1):
        heading = FAMILY_HEADING.match(line)
        if heading:
            current = Entry(heading["family"], line.lstrip("# ").strip(), n)
            entries.setdefault(current.family, current)
            in_sources = False
            continue
        if SOURCES_HEADING.match(line):
            current, in_sources = None, True
            continue
        if HEADING.match(line):
            current, in_sources = None, False
            continue
        cells = _cells(line)
        if not cells or len(cells) < 2:
            continue
        if current is not None and cells[0] not in ("Field",):
            current.fields.setdefault(cells[0], Cell(cells[1], n))
        elif in_sources:
            keys = TOKEN.findall(cells[0])
            if keys and cells[0] != "Key":
                sources.setdefault(keys[0], Cell(line.strip(), n))
    return Doc(path, entries, sources)


def task_files(tasks_dir: Path = TASKS_DIR) -> dict[str, list[Path]]:
    """Corpus task files by family ID (A1, B3, ...), skipping files not in the corpus."""
    families: dict[str, list[Path]] = {}
    for path in sorted(tasks_dir.glob("*.jsonl")):
        if path.name.startswith(NOT_IN_CORPUS):
            continue
        match = TASK_FILE.match(path.name)
        if match:
            families.setdefault(match["family"].upper(), []).append(path)
    return families


def check(doc_path: Path = DOC, tasks_dir: Path = TASKS_DIR, code_dir: Path = TASK_CODE_DIR) -> list[Problem]:
    """Every way the corpus page fails to cover the task files."""
    if not doc_path.is_file():
        raise FileNotFoundError(f"no corpus documentation at {doc_path}")
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"no task directory at {tasks_dir}")
    doc = parse_doc(doc_path)
    docp = _rel(doc_path)
    problems: list[Problem] = []

    for entry in doc.entries.values():
        for name in REQUIRED_FIELDS:
            cell = entry.fields.get(name)
            if cell is None or not cell.text.strip() or not cell.tokens and name != "Tolerance":
                problems.append(Problem("D2", docp, cell.line if cell else entry.line,
                                        f"{entry.family} entry has no {name!r}"
                                        + (" with backticked values" if cell and name != "Tolerance" else "")))
        for name in KEYED_FIELDS:
            cell = entry.fields.get(name)
            for key in cell.tokens if cell else []:
                if key not in doc.sources:
                    problems.append(Problem("D7", docp, cell.line if cell else entry.line,
                                            f"{entry.family} {name} cites {key!r}, which is not in the Sources table"))
        listed = entry.fields.get("Task files")
        for name in listed.tokens if listed else []:
            if name.endswith(".jsonl") and not (tasks_dir / name).is_file():
                problems.append(Problem("D8", docp, listed.line if listed else entry.line,
                                        f"{entry.family} Task files names {name}, which does not exist"))
        scorer = entry.fields.get("Scorer")
        code = sorted(code_dir.glob(f"{entry.family.lower()}_*.py"))
        text = "\n".join(p.read_text(encoding="utf-8") for p in code)
        for name in scorer.tokens if scorer else []:
            if IDENTIFIER.match(name) and f"{name}(" not in text:
                where = ", ".join(_rel(p) for p in code) or f"tasks/{entry.family.lower()}_*.py (none found)"
                problems.append(Problem("D9", docp, scorer.line if scorer else entry.line,
                                        f"{entry.family} Scorer names {name}, which {where} does not use"))

    for family, paths in sorted(task_files(tasks_dir).items()):
        entry = doc.entries.get(family)
        if entry is None:
            problems.append(Problem("D1", _rel(paths[0]), None,
                                    f"task family {family} has no '## {family} · <name>' entry in {docp}"))
            continue
        listed = entry.fields.get("Task files")
        data = entry.fields.get("Data source")
        oracle = entry.fields.get("Oracle")
        for path in paths:
            if listed is None or path.name not in listed.tokens:
                problems.append(Problem("D3", _rel(path), None,
                                        f"{path.name} is not listed in the {family} Task files row of {docp}"))
            seen: set[tuple[str, str]] = set()
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError:
                    continue  # malformed lines are bench.validate's job
                tid = task.get("id", f"line {n}")
                source = task.get("source")
                if isinstance(source, str) and ("source", source) not in seen:
                    seen.add(("source", source))
                    if source not in doc.sources:
                        problems.append(Problem("D4", _rel(path), n,
                                                f"task {tid} cites source {source!r}, which is not in the "
                                                f"Sources table of {docp}"))
                    if data is None or source not in data.tokens:
                        problems.append(Problem("D5", _rel(path), n,
                                                f"task {tid} cites source {source!r}, which the {family} "
                                                f"Data source row of {docp} does not name"))
                meta = task.get("metadata") or {}
                for key, cite in ORACLE_KEYS.items():
                    if not isinstance(meta.get(key), str):
                        continue
                    citation = cite(meta[key])
                    if ("oracle", citation) in seen:
                        continue
                    seen.add(("oracle", citation))
                    if oracle is None or citation not in oracle.tokens:
                        problems.append(Problem("D6", _rel(path), n,
                                                f"task {tid} cites oracle {citation} ({key}), which the {family} "
                                                f"Oracle row of {docp} does not name"))
    return problems


def describe(family: str, doc_path: Path = DOC) -> str:
    """One family's entry as plain text, with the Sources rows it cites."""
    doc = parse_doc(doc_path)
    entry = doc.entries.get(family.upper())
    if entry is None:
        known = ", ".join(sorted(doc.entries)) or "none"
        raise KeyError(f"no family {family!r} in {_rel(doc_path)} (documented: {known})")
    width = max(len(k) for k in entry.fields)
    out = [entry.title, "=" * len(entry.title)]
    out += [f"{name.ljust(width)}  {cell.text}" for name, cell in entry.fields.items()]
    cited = [k for name in KEYED_FIELDS if name in entry.fields for k in entry.fields[name].tokens]
    if cited:
        out += ["", "Sources cited:"]
        for key in dict.fromkeys(cited):
            row = doc.sources.get(key)
            cells = _cells(row.text) if row else None
            out.append(f"  {key}: " + (" | ".join(cells[1:]) if cells else "NOT IN THE SOURCES TABLE"))
    return "\n".join(out)


def github_annotation(problem: Problem) -> str:
    def esc(text: str, props: bool = False) -> str:
        text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        return text.replace(":", "%3A").replace(",", "%2C") if props else text
    where = f"file={esc(problem.path, True)}" + (f",line={problem.line}" if problem.line else "")
    return f"::error {where},title={esc('Corpus documentation', True)}::[{problem.code}] {esc(problem.message)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.corpus_doc", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--family", help="print one family's entry (A1, B3, ...) instead of checking")
    parser.add_argument("--doc", type=Path, default=DOC, help="the corpus page (default: docs/corpus.md)")
    parser.add_argument("--tasks", type=Path, default=TASKS_DIR, help="task directory (default: data/tasks)")
    parser.add_argument("--annotations", action="store_true", help="also print GitHub Actions error annotations")
    args = parser.parse_args(argv)
    try:
        if args.family:
            print(describe(args.family, args.doc))
            return 0
        problems = check(args.doc, args.tasks)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem)
        if args.annotations:
            print(github_annotation(problem))
    families = task_files(args.tasks)
    count = f"{len(families)} families, {sum(len(v) for v in families.values())} task files"
    if problems:
        print(f"FAIL: corpus documentation check, {len(problems)} problem(s) ({count})")
        return 1
    print(f"PASS: corpus documentation check ({count})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
