"""Inter-rater agreement pilot on the graded items (WBS 3.9, issue #64).

The graded items are the unsafe-if clauses whose ``detect`` is "judge": the ones no
rule can check, so a person (and later the judge model, WBS 6.7) has to read the
answer and decide. In the protocol suites they are:

    a1.identity_overtrusted    a1.source_misattributed    a2.position_overtrusted
    b4.fault_overattributed    b4.attack_without_basis

Each clause's text is the rubric for that item. Before the judge is built and before
the protocol corpus is frozen (WBS 3.11), this pilot checks that the rubric is clear
enough that two team members, grading the same answers apart, reach the same verdict.

    python -m bench.irr sample logs/baseline/v1.3.0 --pilot pilot-01 --seed 3909
    python -m bench.irr grade data/irr/pilot-01 --rater ss      # each rater, separately
    python -m bench.irr grade data/irr/pilot-01 --rater kjg
    python -m bench.irr score data/irr/pilot-01 --raters ss kjg
    python -m bench.irr gate                                    # before the freeze

``sample`` reads the baseline EvalLogs (WBS 3.8) and draws a seeded sample of
(answer, judge clause) items, stratified so every clause kind appears. The packet is
blind: it holds the clause, the task prompt and the model's full answer, never the
model's name or any score. The model behind each item is in ``key.json``. Clause text
comes from the *current* task files, not from the log, so after a rubric revision the
same answers can be graded again against the new wording without rerunning the
models. The model never sees the clauses, so its answer stays valid; ``sample`` checks
that each logged prompt still matches the current task.

Each rater grades every item "safe" or "unsafe" (with an optional note) into
``grades/<rater>.csv``, by hand or with ``grade``, which shows one item at a time and
can be stopped and resumed. Raters must not see each other's file until both are done.

``score`` computes Krippendorff's alpha (nominal) with Inspect's built-in
``krippendorff_alpha`` metric, plus raw percent agreement and a per-clause breakdown,
and records them in ``agreement.json`` and ``agreement.md``. The pilot passes when
alpha >= ``ALPHA_THRESHOLD`` (0.800, Krippendorff's level for relying on the data;
0.667 is his floor for tentative conclusions only). Below it, ``score`` also writes
``REVISION.md``, which sends the rubric back: it lists every clause kind with
disagreements, worst first, with each disputed answer and both raters' notes.

``gate`` blocks the corpus freeze: it passes only if the newest pilot graded against
the current rubric text passed. The rubric is identified by a fingerprint of every
judge clause's text in the current task files, so revising any clause (which also
needs a task version bump, CLAUDE.md rule 4) makes the earlier pilot stale and a new
one is needed.

Agreement is about the rubric, not the models: it is not a safety score and is never
combined with capability (CLAUDE.md rule 2).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.scorer import SampleScore, Score, krippendorff_alpha

from bench.adsb import a1, a2
from bench.adsb import unsafe_if as adsb_unsafe_if
from bench.arinc429 import b4
from bench.arinc429 import unsafe_if as arinc429_unsafe_if

IRR_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
IRR_DIR = REPO_ROOT / "data" / "irr"

ALPHA_THRESHOLD = 0.800
"""Krippendorff (2004): rely on data with alpha >= .800; .667 is the floor for tentative
conclusions. Fixed here, not a flag, so a pilot cannot pass by lowering it."""
TENTATIVE_FLOOR = 0.667
MIN_ITEMS = 20
DEFAULT_ITEMS = 30
GRADES = ("safe", "unsafe")

TASK_FILES: dict[str, Path] = {
    "a1_adsb_decode": a1.TASK_FILE,
    "a2_cpr_position": a2.TASK_FILE,
    "b4_arinc429_fault_vs_injection": b4.TASK_FILE,
}
"""Protocol-suite tasks that carry judge clauses, by Inspect task name."""

RUBRIC_VERSIONS = {"adsb": adsb_unsafe_if.UNSAFE_IF_VERSION, "arinc429": arinc429_unsafe_if.UNSAFE_IF_VERSION}

PACKET = "packet.jsonl"
KEY = "key.json"
GRADES_DIR = "grades"
RECORD = "agreement.json"
RECORD_MD = "agreement.md"
REVISION = "REVISION.md"

RATER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class PilotError(ValueError):
    """The pilot cannot go ahead: bad input, a stale log, an edited packet or incomplete grades."""


# ---------------------------------------------------------------------------
# The rubric: every judge clause in the current task files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricItem:
    task: str
    task_version: str
    task_file: str
    sample_id: str
    clause_id: str
    text: str
    prompt: str  # the task's input, as the model saw it after the preamble


def current_rubric(task_files: dict[str, Path] | None = None) -> dict[tuple[str, str, str], RubricItem]:
    """Every judge clause in the current task files, keyed by (task, sample id, clause id)."""
    rubric: dict[tuple[str, str, str], RubricItem] = {}
    for task, path in sorted((task_files or TASK_FILES).items()):
        version = re.search(r"_v(\d+\.\d+\.\d+)\.jsonl$", path.name)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for clause in row["metadata"].get("unsafe_if") or []:
                if clause.get("detect") != "judge":
                    continue
                key = (task, str(row["id"]), str(clause["id"]))
                rubric[key] = RubricItem(task, version.group(1) if version else "?", path.name,
                                         str(row["id"]), str(clause["id"]), str(clause["text"]),
                                         str(row["input"]))
    return rubric


def rubric_fingerprint(rubric: dict[tuple[str, str, str], RubricItem] | None = None) -> str:
    """sha256 over every judge clause's text. Any rubric edit changes it."""
    rubric = current_rubric() if rubric is None else rubric
    material = [[*key, item.text] for key, item in sorted(rubric.items())]
    return hashlib.sha256(json.dumps(material, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# sample: draw the items from baseline EvalLogs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One gradable item: one model's answer to one sample, against one judge clause."""

    task: str
    task_version: str
    sample_id: str
    clause_id: str
    clause: str
    prompt: str
    response: str
    model: str
    log_file: str


def _input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    parts = []
    for message in value or []:
        text = getattr(message, "text", None)
        parts.append(text if isinstance(text, str) else str(getattr(message, "content", "")))
    return "\n".join(parts)


def collect_units(log_dir: Path, rubric: dict[tuple[str, str, str], RubricItem] | None = None) -> list[Unit]:
    """Every non-blank answer in the logs paired with each judge clause of its sample.

    Uses the newest successful log per task and model. Raises PilotError when a logged
    prompt no longer matches the current task (the answer would be to another question).
    """
    rubric = current_rubric() if rubric is None else rubric
    by_sample: dict[tuple[str, str], list[RubricItem]] = defaultdict(list)
    for (task, sample_id, _), item in rubric.items():
        by_sample[(task, sample_id)].append(item)

    newest: dict[tuple[str, str], Any] = {}
    for info in list_eval_logs(str(log_dir), recursive=False):
        header = read_eval_log(info, header_only=True)
        task = header.eval.task.rsplit("/", 1)[-1]
        if header.status != "success" or task not in TASK_FILES:
            continue
        key = (task, header.eval.model)
        if key not in newest or newest[key][0] < header.eval.created:
            newest[key] = (header.eval.created, info)
    if not newest:
        raise PilotError(f"no successful A1, A2 or B4 EvalLogs in {log_dir}")

    units: list[Unit] = []
    for (task, model), (_, info) in sorted(newest.items()):
        log = read_eval_log(info)
        for sample in log.samples or []:
            items = by_sample.get((task, str(sample.id)), [])
            if not items:
                continue
            if items[0].prompt not in _input_text(sample.input):
                raise PilotError(f"{Path(info.name).name}: the prompt of {task} {sample.id} no longer matches "
                                 f"{items[0].task_file}; rerun the baseline for the current task version")
            response = sample.output.completion if sample.output else ""
            if not response.strip():
                continue  # a blank answer gives the raters nothing to grade
            for item in sorted(items, key=lambda i: i.clause_id):
                units.append(Unit(task, item.task_version, str(sample.id), item.clause_id, item.text,
                                  item.prompt, response, model, Path(info.name).name))
    return units


def draw_sample(units: list[Unit], n: int, seed: int) -> list[Unit]:
    """A seeded sample of n items, round-robin across clause kinds so each one is covered."""
    if n < MIN_ITEMS:
        raise PilotError(f"a pilot needs at least {MIN_ITEMS} items, got {n}")
    if len(units) < n:
        raise PilotError(f"only {len(units)} gradable items in the logs, fewer than the {n} requested")
    rng = random.Random(seed)
    pools: dict[str, list[Unit]] = defaultdict(list)
    for unit in sorted(units, key=lambda u: (u.clause_id, u.task, u.sample_id, u.model)):
        pools[unit.clause_id].append(unit)
    for clause_id in sorted(pools):
        rng.shuffle(pools[clause_id])
    chosen: list[Unit] = []
    while len(chosen) < n:
        for clause_id in sorted(pools):
            if pools[clause_id] and len(chosen) < n:
                chosen.append(pools[clause_id].pop())
    rng.shuffle(chosen)  # so raters don't grade one clause kind in a block
    return chosen


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_packet(chosen: list[Unit], pilot_dir: Path, *, seed: int, source: str,
                 rubric: dict[tuple[str, str, str], RubricItem] | None = None) -> dict[str, Any]:
    """Write the blind packet, the key and a grading template. Refuses to overwrite a pilot."""
    if (pilot_dir / PACKET).exists():
        raise PilotError(f"{pilot_dir} already holds a pilot; give the new one its own --pilot name")
    rubric = current_rubric() if rubric is None else rubric
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / GRADES_DIR).mkdir(exist_ok=True)
    width = len(str(len(chosen)))
    packet, mapping = [], {}
    for i, unit in enumerate(chosen, start=1):
        item_id = f"irr-{i:0{width}d}"
        packet.append({"item_id": item_id, "clause_id": unit.clause_id, "clause": unit.clause,
                       "prompt": unit.prompt, "response": unit.response})
        mapping[item_id] = {"task": unit.task, "task_version": unit.task_version, "sample_id": unit.sample_id,
                            "clause_id": unit.clause_id, "model": unit.model, "log_file": unit.log_file}
    packet_text = _jsonl(packet)
    (pilot_dir / PACKET).write_text(packet_text, encoding="utf-8")
    key = {
        "irr_version": IRR_VERSION,
        "pilot_id": pilot_dir.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_logs": source,
        "seed": seed,
        "n_items": len(chosen),
        "packet_sha256": _sha256(packet_text),
        "rubric_fingerprint": rubric_fingerprint(rubric),
        "rubric_versions": RUBRIC_VERSIONS,
        "task_files": sorted({item.task_file for item in rubric.values()}),
        "items": mapping,
    }
    (pilot_dir / KEY).write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (pilot_dir / GRADES_DIR / "TEMPLATE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "grade", "note"])
        for row in packet:
            writer.writerow([row["item_id"], "", ""])
    return key


# ---------------------------------------------------------------------------
# Reading the packet and grades
# ---------------------------------------------------------------------------


def read_packet(pilot_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The packet items and key. Raises PilotError if the packet was edited after sampling."""
    try:
        text = (pilot_dir / PACKET).read_text(encoding="utf-8")
        key = json.loads((pilot_dir / KEY).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PilotError(f"{pilot_dir} is not a pilot: {Path(exc.filename).name} is missing") from exc
    if _sha256(text.replace("\r\n", "\n")) != key.get("packet_sha256"):
        raise PilotError(f"{pilot_dir / PACKET} does not match the hash in {KEY}: the packet was edited "
                         "after sampling, so the grades would not be of the sampled items")
    items = [json.loads(line) for line in text.splitlines() if line.strip()]
    return items, key


def _check_rater(rater: str) -> str:
    if not RATER.match(rater):
        raise PilotError(f"rater {rater!r} must be lowercase letters, digits, '-' or '_' (initials work)")
    return rater


def grades_path(pilot_dir: Path, rater: str) -> Path:
    return pilot_dir / GRADES_DIR / f"{_check_rater(rater)}.csv"


def read_grades(path: Path, item_ids: list[str], *, complete: bool = True) -> dict[str, tuple[str, str]]:
    """item_id -> (grade, note). With complete=True every item must be graded exactly once."""
    if not path.exists():
        if complete:
            raise PilotError(f"no grades file {path}")
        return {}
    known, grades, problems = set(item_ids), {}, []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            item_id = (row.get("item_id") or "").strip()
            grade = (row.get("grade") or "").strip().lower()
            if not item_id and not grade:
                continue
            if item_id not in known:
                problems.append(f"line {line}: unknown item {item_id!r}")
            elif item_id in grades:
                problems.append(f"line {line}: {item_id} graded twice")
            elif grade not in GRADES:
                if grade or complete:
                    problems.append(f"line {line}: {item_id} grade {grade!r} is not one of {', '.join(GRADES)}")
            else:
                grades[item_id] = (grade, (row.get("note") or "").strip())
    if complete:
        missing = [i for i in item_ids if i not in grades]
        if missing:
            problems.append(f"{len(missing)} item(s) not graded: {', '.join(missing[:10])}"
                            + (" ..." if len(missing) > 10 else ""))
    if problems:
        raise PilotError(f"{path}: " + "; ".join(problems))
    return grades


def _write_grades(path: Path, item_ids: list[str], grades: dict[str, tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "grade", "note"])
        for item_id in item_ids:
            if item_id in grades:
                writer.writerow([item_id, *grades[item_id]])
    tmp.replace(path)


# ---------------------------------------------------------------------------
# grade: one item at a time in the terminal
# ---------------------------------------------------------------------------


def grade_interactive(pilot_dir: Path, rater: str, ask: Callable[[str], str] = input,
                      say: Callable[[str], None] = print) -> int:
    """Grade the packet item by item, saving after each answer. Returns how many are left."""
    items, _ = read_packet(pilot_dir)
    path = grades_path(pilot_dir, rater)
    ids = [i["item_id"] for i in items]
    grades = read_grades(path, ids, complete=False)
    todo = [i for i in items if i["item_id"] not in grades]
    say(f"{len(items) - len(todo)} of {len(items)} graded in {path}. "
        "Answer s (safe), u (unsafe) or q (stop; progress is saved).")
    for n, item in enumerate(todo, start=len(items) - len(todo) + 1):
        rule = "=" * 78
        say(f"\n{rule}\n{item['item_id']} ({n}/{len(items)})   clause {item['clause_id']}\n{rule}")
        say(f"RUBRIC\n{item['clause']}\n")
        say(f"TASK PROMPT\n{item['prompt']}\n")
        say(f"MODEL ANSWER\n{item['response']}\n")
        while True:
            choice = ask("Grade [s/u/q]: ").strip().lower()
            if choice in ("q", "quit"):
                return len(items) - len(grades)
            if choice in ("s", "safe", "u", "unsafe"):
                break
            say("Please answer s, u or q.")
        note = ask("Note (optional, Enter to skip): ").strip()
        grades[item["item_id"]] = ("safe" if choice.startswith("s") else "unsafe", note)
        _write_grades(path, ids, grades)
    say(f"\nAll {len(items)} items graded; saved to {path}.")
    return 0


# ---------------------------------------------------------------------------
# score: agreement, recorded
# ---------------------------------------------------------------------------


def alpha(pairs: list[tuple[str, str]]) -> float:
    """Nominal Krippendorff's alpha for two raters, from Inspect's built-in metric."""
    scores = [SampleScore(score=Score(value=[a, b]), sample_id=i) for i, (a, b) in enumerate(pairs)]
    return float(krippendorff_alpha(level="nominal")(scores))  # type: ignore[arg-type]


def _round(x: float) -> float | None:
    return None if math.isnan(x) else round(x, 4)


def agreement(items: list[dict[str, Any]], a: dict[str, tuple[str, str]],
              b: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Overall and per-clause agreement between two complete sets of grades."""
    pairs = [(a[i["item_id"]][0], b[i["item_id"]][0]) for i in items]
    by_clause: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item, pair in zip(items, pairs):
        by_clause[item["clause_id"]].append(pair)
    per_clause = {
        cid: {"items": len(ps), "agree": sum(x == y for x, y in ps),
              "percent_agreement": round(sum(x == y for x, y in ps) / len(ps), 4),
              "alpha": _round(alpha(ps)),
              "unsafe_share": round(sum((x == "unsafe") + (y == "unsafe") for x, y in ps) / (2 * len(ps)), 4)}
        for cid, ps in sorted(by_clause.items())}
    return {"items": len(pairs), "agree": sum(x == y for x, y in pairs),
            "percent_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4),
            "alpha": _round(alpha(pairs)),
            "unsafe_share": round(sum((x == "unsafe") + (y == "unsafe") for x, y in pairs) / (2 * len(pairs)), 4),
            "per_clause": per_clause}


def score_pilot(pilot_dir: Path, raters: list[str]) -> dict[str, Any]:
    """Compute and record agreement. Writes REVISION.md when it is below the threshold."""
    if len(raters) != 2 or raters[0] == raters[1]:
        raise PilotError("the pilot compares exactly two different raters")
    items, key = read_packet(pilot_dir)
    if len(items) < MIN_ITEMS:
        raise PilotError(f"a pilot needs at least {MIN_ITEMS} items, this packet has {len(items)}")
    ids = [i["item_id"] for i in items]
    paths = [grades_path(pilot_dir, r) for r in raters]
    a, b = (read_grades(p, ids) for p in paths)
    stats = agreement(items, a, b)
    passed = stats["alpha"] is not None and stats["alpha"] >= ALPHA_THRESHOLD
    disagreements = [
        {"item_id": i["item_id"], "clause_id": i["clause_id"],
         "grades": {raters[0]: a[i["item_id"]][0], raters[1]: b[i["item_id"]][0]},
         "notes": {raters[0]: a[i["item_id"]][1], raters[1]: b[i["item_id"]][1]}}
        for i in items if a[i["item_id"]][0] != b[i["item_id"]][0]]
    record = {
        "irr_version": IRR_VERSION,
        "pilot_id": key.get("pilot_id", pilot_dir.name),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raters": raters,
        "metric": "krippendorff_alpha (nominal), inspect_ai.scorer",
        "threshold": ALPHA_THRESHOLD,
        "status": "pass" if passed else "revise_rubric",
        **{k: stats[k] for k in ("items", "agree", "percent_agreement", "alpha", "unsafe_share")},
        "per_clause": stats["per_clause"],
        "disagreements": disagreements,
        "rubric_fingerprint": key["rubric_fingerprint"],
        "rubric_versions": key.get("rubric_versions"),
        "task_files": key.get("task_files"),
        "packet_sha256": key["packet_sha256"],
        "grades_sha256": {r: _sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n"))
                          for r, p in zip(raters, paths)},
        "seed": key.get("seed"),
        "source_logs": key.get("source_logs"),
    }
    (pilot_dir / RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (pilot_dir / RECORD_MD).write_text(render_record(record), encoding="utf-8")
    revision = pilot_dir / REVISION
    if passed:
        revision.unlink(missing_ok=True)
    else:
        revision.write_text(render_revision(record, {i["item_id"]: i for i in items}), encoding="utf-8")
    return record


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_record(record: dict[str, Any]) -> str:
    verdict = ("PASS: the rubric is reliable enough to freeze and to calibrate the judge against"
               if record["status"] == "pass" else
               f"BELOW THRESHOLD: the rubric goes back for revision (see {REVISION}); "
               "the corpus freeze is blocked until a pilot on the revised rubric passes")
    band = ("" if record["alpha"] is None or record["alpha"] >= ALPHA_THRESHOLD else
            " (tentative band: usable for exploratory findings only)" if record["alpha"] >= TENTATIVE_FLOOR else
            " (below the tentative floor)")
    out = [f"# Inter-rater agreement: {record['pilot_id']}", "",
           f"**{verdict}.**", "",
           "| Measure | Value |", "| --- | --- |",
           f"| Raters | {', '.join(record['raters'])} |",
           f"| Items | {record['items']} |",
           f"| Krippendorff's alpha (nominal) | {_fmt(record['alpha'])}{band} |",
           f"| Threshold | {record['threshold']:.3f} |",
           f"| Raw agreement | {record['agree']}/{record['items']} ({100 * record['percent_agreement']:.1f}%) |",
           f"| Share of grades that are unsafe | {100 * record['unsafe_share']:.1f}% |",
           f"| Rubric fingerprint | `{record['rubric_fingerprint'][:16]}` |",
           f"| Scored | {record['scored_at']} |", "",
           "Alpha corrects for chance agreement, so it can sit well below raw agreement when "
           "nearly every grade is the same; read both.", "",
           "## Per clause", "",
           "| Clause | Items | Agree | Alpha | Unsafe share |", "| --- | --- | --- | --- | --- |"]
    for cid, c in record["per_clause"].items():
        out.append(f"| `{cid}` | {c['items']} | {c['agree']}/{c['items']} | {_fmt(c['alpha'])} | "
                   f"{100 * c['unsafe_share']:.0f}% |")
    out += ["", f"## Disagreements ({len(record['disagreements'])})", ""]
    if not record["disagreements"]:
        out.append("None.")
    for d in record["disagreements"]:
        grades = ", ".join(f"{r}: {g}" for r, g in d["grades"].items())
        out.append(f"- `{d['item_id']}` `{d['clause_id']}`: {grades}")
    return "\n".join(out) + "\n"


def render_revision(record: dict[str, Any], items: dict[str, dict[str, Any]]) -> str:
    by_clause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in record["disagreements"]:
        by_clause[d["clause_id"]].append(d)
    order = sorted(by_clause, key=lambda c: (-len(by_clause[c]), c))
    out = [f"# Rubric revision needed: {record['pilot_id']}", "",
           f"Krippendorff's alpha was {_fmt(record['alpha'])}, below the {record['threshold']:.3f} threshold, "
           f"with {record['agree']} of {record['items']} items in agreement. The judge-clause rubric is not "
           "clear enough for two people to apply it the same way, so it cannot be frozen (WBS 3.11) or used "
           "to calibrate the judge (WBS 6.8) yet.", "",
           "To finish the revision:", "",
           "1. For each clause below, decide with both raters what the rubric should say about the disputed "
           "answers, and reword the clause builder (`bench/adsb/unsafe_if.py` or "
           "`bench/arinc429/unsafe_if.py`) so the wording settles it.",
           "2. Bump `UNSAFE_IF_VERSION` and the task version, and rebuild the task files (CLAUDE.md rule 4).",
           "3. Run a new pilot: `python -m bench.irr sample <the same logs> --pilot <new name> --seed <seed>`; "
           "the same seed gives the same answers, now under the new wording. Use new raters or let some "
           "time pass, so the grades are not recalled from this round.",
           "4. `python -m bench.irr gate` passes once the new pilot does.", "",
           "Clauses with disagreements, most first:", ""]
    for cid in order:
        c = record["per_clause"][cid]
        out += [f"## `{cid}`: {len(by_clause[cid])} of {c['items']} items disputed (alpha {_fmt(c['alpha'])})", ""]
        for d in by_clause[cid]:
            item = items[d["item_id"]]
            out += [f"### `{d['item_id']}`", "", "Rubric:", "", f"> {item['clause']}", "",
                    "Answer:", "", "```", item["response"].strip(), "```", ""]
            for rater, grade in d["grades"].items():
                note = d["notes"].get(rater)
                out.append(f"- **{rater}**: {grade}" + (f": {note}" if note else ""))
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# gate: block the freeze until a pilot on the current rubric passes
# ---------------------------------------------------------------------------


def gate(irr_dir: Path = IRR_DIR, fingerprint: str | None = None) -> tuple[bool, str]:
    """(ok, reason). ok only when the newest pilot on the current rubric text passed."""
    fingerprint = fingerprint or rubric_fingerprint()
    records = []
    for path in sorted(irr_dir.glob(f"*/{RECORD}")):
        try:
            records.append((json.loads(path.read_text(encoding="utf-8")), path.parent))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"cannot read {path}: {exc}"
    current = sorted(((r, d) for r, d in records if r.get("rubric_fingerprint") == fingerprint),
                     key=lambda rd: rd[0].get("scored_at", ""))
    if not current:
        latest = max(records, key=lambda rd: rd[0].get("scored_at", ""), default=None)
        stale = (f"; the latest pilot ({latest[1].name}, {latest[0].get('status')}) graded an earlier "
                 "version of the rubric") if latest else ""
        return False, f"no inter-rater pilot has been scored on the current rubric ({fingerprint[:16]}){stale}"
    record, where = current[-1]
    if record.get("status") != "pass" or (record.get("threshold") or 0) < ALPHA_THRESHOLD:
        return False, (f"pilot {where.name} on the current rubric is below threshold "
                       f"(alpha {_fmt(record.get('alpha'))} < {ALPHA_THRESHOLD:.3f}); revise the rubric per "
                       f"{where / REVISION} and run a new pilot")
    return True, (f"pilot {where.name} passed on the current rubric (alpha {_fmt(record.get('alpha'))} "
                  f">= {ALPHA_THRESHOLD:.3f}, raters {', '.join(record.get('raters', []))})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.irr", description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("sample", help="draw a blind grading packet from baseline EvalLogs")
    s.add_argument("log_dir", type=Path)
    s.add_argument("--pilot", required=True, help="pilot name; the packet goes to data/irr/<pilot>")
    s.add_argument("--seed", type=int, required=True, help="required, so the sample can be redrawn")
    s.add_argument("-n", "--items", type=int, default=DEFAULT_ITEMS)
    s.add_argument("--irr-dir", type=Path, default=IRR_DIR)
    g = commands.add_parser("grade", help="grade a packet item by item in the terminal")
    g.add_argument("pilot_dir", type=Path)
    g.add_argument("--rater", required=True, help="your initials, lowercase")
    c = commands.add_parser("score", help="compute and record agreement between two raters")
    c.add_argument("pilot_dir", type=Path)
    c.add_argument("--raters", nargs=2, required=True, metavar=("A", "B"))
    t = commands.add_parser("gate", help="exit 1 unless a pilot on the current rubric passed")
    t.add_argument("--irr-dir", type=Path, default=IRR_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 ok / pass, 1 below threshold or gate closed, 2 refused (bad input)."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "sample":
            units = collect_units(args.log_dir)
            chosen = draw_sample(units, args.items, args.seed)
            pilot_dir = args.irr_dir / args.pilot
            key = write_packet(chosen, pilot_dir, seed=args.seed, source=str(args.log_dir))
            kinds = sorted({u.clause_id for u in chosen})
            print(f"Wrote {key['n_items']} items ({len(kinds)} clause kinds, from {len(units)} gradable) to "
                  f"{pilot_dir}\nEach rater now runs: python -m bench.irr grade {pilot_dir} --rater <initials>")
            return 0
        if args.command == "grade":
            left = grade_interactive(args.pilot_dir, args.rater)
            return 0 if left == 0 else 1
        if args.command == "score":
            record = score_pilot(args.pilot_dir, args.raters)
            print(render_record(record))
            if record["status"] != "pass":
                print(f"Rubric sent back for revision: {args.pilot_dir / REVISION}")
                return 1
            return 0
        ok, reason = gate(args.irr_dir)
        print(("OK: " if ok else "BLOCKED: ") + reason)
        return 0 if ok else 1
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
