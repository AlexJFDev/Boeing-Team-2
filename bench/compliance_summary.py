"""Airworthiness compliance summary (WBS 6.4, issue #163).

Lists every safety finding that finished Inspect logs graded (an unsafe-if clause or a D2
vector) with the source it is graded against, from data/safety/citations_v<version>.json,
next to the 6.3 refusal and injection-compliance rates from bench.safety_telemetry.aggregate().
A graded finding with no citation stops generation. Not a DO-326A compliance determination.

    python -m bench.compliance_summary logs/ [--citations PATH] [--out DIR] [--no-write]

Exit codes: 0 summary written (or printed with --no-write); 1 nothing safety-labeled;
2 refused, nothing written (no logs, bad path, bad catalogue, or an uncited graded finding).

Details: bench/acars/README.md ("Airworthiness compliance summary (Task 6.4)").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.analysis import SampleColumn, samples_df
from inspect_ai.log import EvalSample

from bench import safety_telemetry as st

# Bump on any change to finding extraction, graded/pending rules, citation checks or output schema (rule 4).
SUMMARY_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOGUE = REPO_ROOT / "data" / "safety" / "citations_v1.0.0.json"
DEFAULT_OUT_DIR = Path("logs") / "compliance_summary"

SECTION, SOURCE_NO_SECTION, TAXONOMY = "section", "source_no_section", "taxonomy"
UNCITED_PENDING = "uncited_pending"
CITATION_STATUSES = (SECTION, SOURCE_NO_SECTION, TAXONOMY)
HEADER_STATUSES = (*CITATION_STATUSES, UNCITED_PENDING)
# Standards the issue asks to trace to; counted in the header (section-level citations only).
TRACKED_STANDARDS = ("DO-326A", "ARINC 429", "ARINC 664")

ENTRY_KEYS = ("finding_id", "family", "kind", "standard", "section", "tag", "source_file", "source_locator",
              "status", "notes")
KINDS = {"unsafe_if": "unsafe_if_clause", "injection": "d2_vector"}

# Per-attempt outcome of one finding
UNSAFE, SAFE, UNPARSEABLE_OUTCOME, PENDING, CLAUSE_ERROR = "unsafe", "safe", "unparseable", "pending", "error"
GRADED_OUTCOMES = (UNSAFE, SAFE, UNPARSEABLE_OUTCOME)
_D2_OUTCOMES = {st.OBEYED: UNSAFE, st.RESISTED: SAFE, st.UNPARSEABLE: UNPARSEABLE_OUTCOME,
                st.NEEDS_JUDGE: PENDING}
MISSING_ID = "<missing finding id>"

DISCLAIMER = ("Traceability summary of graded safety findings to documented sources. It is not a "
              "DO-326A or any other airworthiness or compliance determination. Citations with status "
              "source_no_section or taxonomy do not name a section of an aviation standard.")

GROUP_KEYS = st.GROUP_KEYS
FINDING_COLUMNS = (
    *GROUP_KEYS, "finding_id", "family", "kind", "scorers", "tasks",
    "n", "graded", "unsafe", "safe", "unparseable", "pending", "clause_errors",
    "citation_status", "standard", "section", "tag", "source_file", "source_locator",
)
_NULLABLE = ("scripted_persona", "guardrail_profile_version", "guardrail_profile_sha256",
             "family", "kind", "standard", "section", "tag", "source_file", "source_locator")

DEFINITIONS = {
    "finding": "an unsafe-if clause ID or D2 vector ID graded by a scorer in SAFETY_SCORERS, per group",
    "n": "sample attempts (each epoch counts once) that carried the finding",
    "graded": "attempts with a verdict: unsafe + safe + unparseable",
    "unsafe": "rule clause flagged, or D2 injected attempt obeyed",
    "safe": "rule clause passed, or D2 injected attempt resisted",
    "unparseable": "D2 injected attempt whose reply could not be parsed",
    "pending": "judge clause or D2 needs_judge attempt: waits for the judge model (WBS 6.7)",
    "clause_errors": "rule clause the scorer could not run (unknown rule); not graded",
    "citation_status": "section / source_no_section / taxonomy from the catalogue, or uncited_pending "
                       "when the finding is not in the catalogue and has not been graded",
    "tag": "the item named in a taxonomy citation (for example an OWASP ID); null otherwise",
    "standard_section_counts": "findings with a section-level citation to each tracked standard",
}


class CitationCatalogueError(ValueError):
    """The citation catalogue is missing, malformed, or cites a file that does not exist."""


class UncitedFindingError(ValueError):
    """A graded safety finding has no entry in the citation catalogue."""

    def __init__(self, findings: list[dict[str, Any]], catalogue: Path) -> None:
        self.findings = findings
        listed = "; ".join(
            f"{f['finding_id']} ({', '.join(f['scorers'])}; task {', '.join(f['tasks'])}, "
            f"{', '.join(f['logs'])}, {f['graded']} graded sample(s))"
            for f in findings
        )
        super().__init__(
            f"summary not generated: {len(findings)} graded safety finding(s) have no citation in "
            f"{catalogue.name}: {listed}. Add each to a new citation catalogue version "
            f"(never edit {catalogue.name} in place)."
        )


# -- catalogue -----------------------------------------------------------------------
@dataclass(frozen=True)
class CitationCatalogue:
    path: Path
    version: str
    sha256: str
    entries: dict[str, dict[str, Any]]

    def provenance(self) -> dict[str, Any]:
        try:
            shown = self.path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            shown = str(self.path)
        return {"path": shown, "version": self.version, "sha256": self.sha256}


def _entry_problems(entry: Any, index: int) -> list[str]:
    if not isinstance(entry, dict):
        return [f"entries[{index}] is not an object"]
    where = f"entries[{index}] ({entry.get('finding_id', '?')})"
    problems = [f"{where}: missing key {k!r}" for k in ENTRY_KEYS if k not in entry]
    problems += [f"{where}: unknown key {k!r}" for k in entry if k not in ENTRY_KEYS]
    if problems:
        return problems
    for key in ("finding_id", "family", "kind", "standard", "source_file", "source_locator"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            problems.append(f"{where}: {key} must be a non-empty string")
    if entry["kind"] not in KINDS.values():
        problems.append(f"{where}: kind must be one of {sorted(KINDS.values())}")
    status = entry["status"]
    if status not in CITATION_STATUSES:
        problems.append(f"{where}: status {status!r} is not one of {list(CITATION_STATUSES)}")
    section = entry["section"]
    if status == SECTION and (not isinstance(section, str) or not section.strip()):
        problems.append(f"{where}: status 'section' needs a section")
    if status != SECTION and section is not None:
        problems.append(f"{where}: only status 'section' may give a section")
    tag = entry["tag"]
    if status == TAXONOMY and (not isinstance(tag, str) or not tag.strip()):
        problems.append(f"{where}: status 'taxonomy' needs a tag")
    if status != TAXONOMY and tag is not None:
        problems.append(f"{where}: only status 'taxonomy' may give a tag")
    if isinstance(entry["source_file"], str) and entry["source_file"].strip():
        source = REPO_ROOT / entry["source_file"]
        if Path(entry["source_file"]).is_absolute() or not source.is_file():
            problems.append(f"{where}: source_file {entry['source_file']!r} is not a file in the repo")
    return problems


def load_catalogue(path: str | Path = DEFAULT_CATALOGUE) -> CitationCatalogue:
    """Load and check a citation catalogue. Raises CitationCatalogueError on any problem."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise CitationCatalogueError(f"cannot read citation catalogue {path}: {e}") from e
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CitationCatalogueError(f"citation catalogue {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("version"), str) \
            or not isinstance(data.get("entries"), list):
        raise CitationCatalogueError(f"citation catalogue {path} needs a string 'version' and an 'entries' list")
    problems: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(data["entries"]):
        found = _entry_problems(entry, i)
        problems += found
        if not found:
            if entry["finding_id"] in entries:
                problems.append(f"entries[{i}]: duplicate finding_id {entry['finding_id']!r}")
            entries[entry["finding_id"]] = entry
    if problems:
        raise CitationCatalogueError(f"citation catalogue {path} is invalid: " + "; ".join(problems))
    return CitationCatalogue(path=path, version=data["version"], sha256=hashlib.sha256(raw).hexdigest(),
                             entries=entries)


# -- finding extraction --------------------------------------------------------------
def _name(scorer: str) -> str:
    return scorer.rsplit("/", 1)[-1]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as dicts keyed by column name (pandas types record keys as Hashable)."""
    return [{str(k): v for k, v in rec.items()} for rec in frame.to_dict("records")]


def _clause_id(error: Any) -> str:
    """Clause ID from an evaluate_clauses error string ("<id>: unknown rule ...")."""
    text = str(error)
    if ":" not in text:
        return MISSING_ID
    return text.split(":", 1)[0].strip() or MISSING_ID


def findings_of(scorer: str, value: Any, metadata: dict[str, Any]) -> list[tuple[str, str]]:
    """(finding_id, outcome) pairs one safety score records. Controls give none."""
    spec = st.SAFETY_SCORERS[scorer]
    if spec.kind == "injection":
        outcome = _D2_OUTCOMES.get(str(value))
        if outcome is None:  # not_applicable (control) or an unknown value: nothing graded
            return []
        return [(str(metadata.get("vector_id") or MISSING_ID), outcome)]
    pairs: list[tuple[str, str]] = []
    for key, outcome in (("flagged", UNSAFE), ("passed", SAFE)):
        for item in metadata.get(key) or []:
            fid = item.get("id") if isinstance(item, dict) else None
            pairs.append((str(fid) if fid not in (None, "", "?") else MISSING_ID, outcome))
    pairs += [(str(fid), PENDING) for fid in metadata.get("judge_pending") or []]
    pairs += [(_clause_id(err), CLAUSE_ERROR) for err in metadata.get("errors") or []]
    return pairs


def _finding_scores(sample: EvalSample) -> Any:
    """Only registered safety scorers, so no capability score is read."""
    out: dict[str, Any] = {}
    for key, score in (sample.scores or {}).items():
        name = _name(key)
        if name in st.SAFETY_SCORERS:
            out[name] = findings_of(name, score.value, score.metadata or {})
    return out


_COLUMNS = [
    SampleColumn("epoch", path="epoch", required=True),
    SampleColumn("finding_scores", path=_finding_scores, full=True),
]


def collect_findings(logs: st.LogInput | Iterable[st.LogInput]) -> pd.DataFrame:
    """One row per (sample attempt, finding) with the 6.3 group keys of that attempt."""
    paths = st.resolve_logs(logs)
    classified = st.classify_samples(paths)
    groups = {(r["log"], r["sample_uuid"], int(r["epoch"])): r for r in _records(classified)}
    raw = samples_df([str(p) for p in paths], columns=_COLUMNS, quiet=True)
    assert isinstance(raw, pd.DataFrame)
    rows: list[dict[str, Any]] = []
    for rec in _records(raw):
        base = groups[(str(rec["log"]), rec["sample_id"], int(rec["epoch"]))]
        scores: dict[str, Any] = st._json(rec.get("finding_scores")) or {}
        for scorer, pairs in sorted(scores.items()):
            for fid, outcome in pairs:
                rows.append({
                    **{k: base[k] for k in GROUP_KEYS},
                    "log": base["log"], "task_name": base["task_name"], "sample_id": base["sample_id"],
                    "epoch": base["epoch"], "scorer": scorer, "kind": KINDS[st.SAFETY_SCORERS[scorer].kind],
                    "finding_id": fid, "outcome": outcome,
                })
    columns = [*GROUP_KEYS, "log", "task_name", "sample_id", "epoch", "scorer", "kind", "finding_id", "outcome"]
    return pd.DataFrame(rows, columns=columns)


# -- summary ---------------------------------------------------------------------------
def _citation(entry: dict[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return {"citation_status": UNCITED_PENDING, "standard": None, "section": None, "tag": None,
                "source_file": None, "source_locator": None}
    return {"citation_status": entry["status"], "standard": entry["standard"], "section": entry["section"],
            "tag": entry["tag"], "source_file": entry["source_file"], "source_locator": entry["source_locator"]}


def _family(finding_id: str, kind: str, entry: dict[str, Any] | None) -> str | None:
    """From the catalogue; for an uncited clause, the "<subfamily>." prefix of its ID
    (bench/adsb/unsafe_if.py convention). None when neither applies."""
    if entry is not None:
        return entry["family"]
    if kind == KINDS["unsafe_if"] and "." in finding_id:
        return finding_id.split(".", 1)[0].upper()
    return None


def _finding_rows(attempts: pd.DataFrame, catalogue: CitationCatalogue) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in _records(attempts):
        key = json.dumps([*(rec[k] for k in GROUP_KEYS), rec["finding_id"]], default=str)
        buckets.setdefault(key, []).append(rec)
    rows = []
    for key in sorted(buckets):
        recs = buckets[key]
        outcomes = [r["outcome"] for r in recs]
        fid = recs[0]["finding_id"]
        entry = catalogue.entries.get(fid)
        graded = sum(outcomes.count(o) for o in GRADED_OUTCOMES)
        rows.append({
            **{k: recs[0][k] for k in GROUP_KEYS},
            "finding_id": fid,
            "family": _family(fid, recs[0]["kind"], entry),
            "kind": entry["kind"] if entry else recs[0]["kind"],
            "scorers": sorted({r["scorer"] for r in recs}),
            "tasks": sorted({r["task_name"] for r in recs}),
            "n": len(recs),
            "graded": graded,
            "unsafe": outcomes.count(UNSAFE),
            "safe": outcomes.count(SAFE),
            "unparseable": outcomes.count(UNPARSEABLE_OUTCOME),
            "pending": outcomes.count(PENDING),
            "clause_errors": outcomes.count(CLAUSE_ERROR),
            **_citation(entry),
            "_logs": sorted({Path(r["log"]).name for r in recs}),
        })
    return rows


def _uncited(rows: list[dict[str, Any]], catalogue: CitationCatalogue) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["graded"] and row["finding_id"] not in catalogue.entries:
            f = by_id.setdefault(row["finding_id"], {"finding_id": row["finding_id"], "scorers": set(),
                                                     "tasks": set(), "logs": set(), "graded": 0})
            f["scorers"].update(row["scorers"])
            f["tasks"].update(row["tasks"])
            f["logs"].update(row["_logs"])
            f["graded"] += row["graded"]
    return [{**f, "scorers": sorted(f["scorers"]), "tasks": sorted(f["tasks"]), "logs": sorted(f["logs"])}
            for _, f in sorted(by_id.items())]


def _status_counts(findings: pd.DataFrame) -> dict[str, int]:
    """Distinct finding IDs per citation status."""
    if findings.empty:
        return {s: 0 for s in HEADER_STATUSES}
    per_id = findings.drop_duplicates("finding_id").set_index("finding_id")["citation_status"]
    return {s: int((per_id == s).sum()) for s in HEADER_STATUSES}


def _standard_counts(findings: pd.DataFrame) -> dict[str, int]:
    """Distinct findings with a section-level citation to each tracked standard."""
    if findings.empty:
        return {s: 0 for s in TRACKED_STANDARDS}
    cited = findings.loc[findings["citation_status"] == SECTION].drop_duplicates("finding_id")
    return {s: int(cited["standard"].astype(str).str.startswith(s).sum()) for s in TRACKED_STANDARDS}


@dataclass(frozen=True)
class ComplianceSummary:
    """The 6.3 safety report plus each graded finding and its citation. No combined score."""

    telemetry: st.SafetyReport
    findings: pd.DataFrame
    catalogue: CitationCatalogue

    @property
    def citation_status_counts(self) -> dict[str, int]:
        return _status_counts(self.findings)

    @property
    def standard_section_counts(self) -> dict[str, int]:
        return _standard_counts(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return st._plain({
            "summary_version": SUMMARY_VERSION,
            "disclaimer": DISCLAIMER,
            "citation_catalogue": self.catalogue.provenance(),
            "citation_status_counts": self.citation_status_counts,
            "standard_section_counts": self.standard_section_counts,
            "definitions": DEFINITIONS,
            "findings": self.findings.to_dict("records"),
            "safety_telemetry": self.telemetry.to_dict(),
        })

    def write(self, out_dir: str | Path = DEFAULT_OUT_DIR, stem: str | None = None) -> tuple[Path, Path]:
        """Write ``<stem>.json`` (full summary) and ``<stem>.csv`` (findings). Returns both paths."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"compliance_summary_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        json_path, csv_path = out / f"{stem}.json", out / f"{stem}.csv"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        csv_frame = self.findings.copy()
        for col in ("scorers", "tasks"):
            csv_frame[col] = [";".join(values) for values in self.findings[col]]
        csv_frame.to_csv(csv_path, index=False, lineterminator="\n")
        return json_path, csv_path


def summarize(logs: st.LogInput | Iterable[st.LogInput],
              catalogue: CitationCatalogue | str | Path | None = None) -> ComplianceSummary:
    """Compliance summary from finished logs. Raises UncitedFindingError before anything is
    written if a graded finding has no catalogue entry."""
    cat = catalogue if isinstance(catalogue, CitationCatalogue) else load_catalogue(catalogue or DEFAULT_CATALOGUE)
    paths = st.resolve_logs(logs)
    rows = _finding_rows(collect_findings(paths), cat)
    uncited = _uncited(rows, cat)
    if uncited:
        raise UncitedFindingError(uncited, cat.path)
    report = st.aggregate(paths)
    findings = pd.DataFrame([{k: r[k] for k in FINDING_COLUMNS} for r in rows], columns=list(FINDING_COLUMNS))
    for col in _NULLABLE:
        findings[col] = pd.Series([r.get(col) for r in rows], dtype=object, index=findings.index)
    return ComplianceSummary(telemetry=report, findings=findings, catalogue=cat)


# -- command line ----------------------------------------------------------------------
def _cite_text(row: dict[str, Any]) -> str:
    status = row["citation_status"]
    if status == UNCITED_PENDING:
        return "none (not graded yet)"
    if status == TAXONOMY:
        return f"{row['standard']}, {row['tag']} (not an aviation standard)"
    section = row["section"] or "section not cited"
    return f"{row['standard']}, {section}"


def format_summary(summary: ComplianceSummary) -> str:
    counts, standards = summary.citation_status_counts, summary.standard_section_counts
    lines = [
        f"Airworthiness compliance summary (summary {SUMMARY_VERSION}, citation catalogue "
        f"{summary.catalogue.version}, sha256 {summary.catalogue.sha256[:12]}...)",
        "Citations: " + " | ".join(f"{s} {counts[s]}" for s in HEADER_STATUSES)
        + " | " + ", ".join(f"{s} {standards[s]}" for s in TRACKED_STANDARDS),
        DISCLAIMER,
        "",
        st.format_report(summary.telemetry),
        "",
        "Safety findings and the source each is graded against:",
    ]
    if summary.findings.empty:
        lines.append("  none")
        return "\n".join(lines)
    f = summary.findings
    keys = f[list(GROUP_KEYS)].apply(lambda r: json.dumps(list(r), default=str), axis=1)
    for key, group in f.groupby(keys, sort=True):
        model, persona, profile, version, _sha = json.loads(str(key))
        label = f"{model}" + (f" [{persona}]" if persona else "") + f", guardrail {profile}" \
            + (f" v{version}" if version is not None else "")
        lines += ["", f"  {label}"]
        table = group.assign(citation=[_cite_text(r) for r in _records(group)],
                             source=group["source_file"].fillna("-"))
        cols = ["finding_id", "family", "n", "graded", "unsafe", "safe", "unparseable", "pending",
                "citation_status", "citation", "source"]
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
            body = table[cols].to_string(index=False, na_rep="-")
        lines += ["    " + line for line in body.splitlines()]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.compliance_summary",
                                description=(__doc__ or "").partition("\n")[0])
    p.add_argument("logs", nargs="+", help=".eval log files or directories")
    p.add_argument("--citations", default=str(DEFAULT_CATALOGUE),
                   help="citation catalogue JSON (default data/safety/citations_v1.0.0.json)")
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR), help=f"output directory (default {DEFAULT_OUT_DIR})")
    p.add_argument("--no-write", action="store_true", help="print the summary without writing files")
    args = p.parse_args(argv)
    try:
        summary = summarize(args.logs, args.citations)
    except (FileNotFoundError, st.NoLogsError, CitationCatalogueError, UncitedFindingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_summary(summary))
    if summary.telemetry.rates.empty:
        return 1
    if not args.no_write:
        json_path, csv_path = summary.write(args.out)
        print(f"\nwrote {json_path}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
