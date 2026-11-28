"""Per-task report and ground-truth review list for a baseline run (WBS 3.8, issue #63).

    python -m bench.baseline_report logs/baseline/v1.0.0

Reads every EvalLog in a baseline log directory (one per task and model, written by
``bench baseline``) and writes ``report/`` beside them:

``report.md``       summary per task and model, every sample's score per model, and the
                    review list
``report.json``     the same data with every model's full raw answer, for rescoring
                    (CLAUDE.md rule 8)
``per_sample.csv``  one row per sample and model, for a spreadsheet

Capability and safety are kept in separate columns and never combined (CLAUDE.md rule
2). Capability is the task's own scorer; safety is the unsafe-if rule scorer, counted
only on samples it actually checked.

**Ground-truth review.** A sample that scores 0 on capability for every model in the
run is flagged for review. When no model gets an item right, the item may be wrong:
an ambiguous prompt, a missing premise, or a bad target. Every target is already
re-derived from its reference decoder before each run, so the review is about the item
as a whole, not the arithmetic. The flag says only that a human should look; a hard
item can be correct and still beat every model. A sample is flagged only when every
model has a score for it, and a run with one model flags every miss, which the report
says. A sample where any model returned no text at all (a blank answer, usually cut off
by max_tokens while reasoning) is not flagged either: it is listed separately as not
reviewable, with each model's stop reason, because a blank says nothing about the item.
The same holds for an answer cut off by max_tokens partway through, which the report
counts as cut off.

When a task has more than one log for the same model, as after a retry, the newest
successful one is used. Task-model pairs with no successful log are listed as missing.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

# 1.1.0: a blank answer is reported as "no answer" and keeps its sample out of the review;
# max_tokens and each answer's stop reason are recorded.
# 1.2.0: an answer cut off by max_tokens partway through counts like a blank one for the
# review, and each run's model_args (e.g. stream) are recorded.
# 1.2.1: the report files are written as UTF-8 whatever the OS default, so an answer with
# characters such as "\u2082" no longer crashes the report on Windows (cp1252). The CSV
# carries a byte-order mark so spreadsheets open it as UTF-8. Contents are unchanged.
REPORT_VERSION = "1.2.1"
SAFETY_SCORERS = ("unsafe_if_rule_scorer",)
REPORT_DIR = "report"


@dataclass
class Cell:
    """One sample's result for one model."""

    capability: float | None
    safety: float | None  # None when the safety scorer did not check this sample
    answer: str
    stop_reason: str | None = None

    @property
    def no_answer(self) -> bool:
        """The model returned no text at all, e.g. cut off by max_tokens while reasoning."""
        return not self.answer.strip()

    @property
    def cut_off(self) -> bool:
        """The model hit max_tokens: blank, or an answer stopped partway through."""
        return self.no_answer or self.stop_reason == "max_tokens"


@dataclass
class Row:
    task: str
    sample_id: str
    target: str
    case: str | None
    task_text: str
    cells: dict[str, Cell] = field(default_factory=dict)


@dataclass
class Run:
    """One task-model pair: the log it came from and what it records."""

    task: str
    model: str
    status: str
    log: str
    task_version: str | None = None
    capability_scorer: str | None = None
    samples: int = 0
    capability_mean: float | None = None
    checked: int = 0
    unsafe: int = 0
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    no_answer: int = 0
    cut_off: int = 0  # answers that hit max_tokens but are not blank
    stop_reasons: dict[str, int] = field(default_factory=dict)
    commit: str | None = None
    created: str = ""


def _score_value(score: Any) -> float | None:
    try:
        return float(score.value)
    except (TypeError, ValueError):
        return None


def _capability_scorer(names: list[str]) -> str | None:
    return next((n for n in names if n not in SAFETY_SCORERS), None)


def _target_text(target: Any) -> str:
    return target if isinstance(target, str) else json.dumps(target)


def load_runs(log_dir: Path) -> tuple[list[Run], dict[tuple[str, str], EvalLog]]:
    """Every log's header, and the newest successful full log per task and model."""
    runs: list[Run] = []
    best: dict[tuple[str, str], EvalLog] = {}
    for info in list_eval_logs(str(log_dir), recursive=False):
        header = read_eval_log(info, header_only=True)
        key = (header.eval.task, header.eval.model)
        runs.append(Run(header.eval.task, header.eval.model, header.status, Path(info.name).name,
                        created=header.eval.created))
        if header.status != "success":
            continue
        if key in best and best[key].eval.created >= header.eval.created:
            continue
        best[key] = read_eval_log(info)
    return runs, best


def build_report(log_dir: Path) -> dict[str, Any]:
    all_runs, logs = load_runs(log_dir)
    if not all_runs:
        raise FileNotFoundError(f"no EvalLogs in {log_dir}")
    tasks = sorted({r.task for r in all_runs})
    models = sorted({r.model for r in all_runs})
    rows: dict[tuple[str, str], Row] = {}
    runs: list[Run] = []
    for (task, model), log in sorted(logs.items()):
        samples = log.samples or []
        names = list(samples[0].scores or {}) if samples else []
        cap_name = _capability_scorer(names)
        config = log.plan.config  # the generate config the run used (temperature, seed, ...)
        run = Run(task, model, log.status, Path(log.location).name, task_version=str(log.eval.task_version),
                  capability_scorer=cap_name, samples=len(samples),
                  temperature=config.temperature if config else None, seed=config.seed if config else None,
                  max_tokens=config.max_tokens if config else None,
                  model_args=dict(log.eval.model_args or {}),
                  commit=log.eval.revision.commit if log.eval.revision else None, created=log.eval.created)
        values = []
        for s in samples:
            scores = s.scores or {}
            cap = _score_value(scores[cap_name]) if cap_name in scores else None
            safety_score = next((scores[n] for n in SAFETY_SCORERS if n in scores), None)
            checked = bool(safety_score is not None and (safety_score.metadata or {}).get("checked"))
            safety = _score_value(safety_score) if checked else None
            meta = s.metadata or {}
            row = rows.setdefault((task, str(s.id)), Row(task, str(s.id), _target_text(s.target),
                                                         meta.get("case"), str(meta.get("task_text", ""))))
            stop = str(s.output.stop_reason) if s.output and s.output.stop_reason else None
            cell = Cell(cap, safety, s.output.completion if s.output else "", stop)
            row.cells[model] = cell
            run.stop_reasons[stop or "none"] = run.stop_reasons.get(stop or "none", 0) + 1
            run.no_answer += int(cell.no_answer)
            run.cut_off += int(cell.cut_off and not cell.no_answer)
            if cap is not None:
                values.append(cap)
            if checked:
                run.checked += 1
                run.unsafe += int(safety == 0.0)
        run.capability_mean = sum(values) / len(values) if values else None
        runs.append(run)

    missing = [{"task": t, "model": m} for t in tasks for m in models if (t, m) not in logs]
    failed = [{"task": r.task, "model": r.model, "status": r.status, "log": r.log}
              for r in all_runs if r.status != "success" and (r.task, r.model) not in logs]
    review, unreviewable = [], []
    for row in sorted(rows.values(), key=lambda r: (r.task, r.sample_id)):
        cells = [row.cells.get(m) for m in models]
        if not cells or not all(c is not None and c.capability == 0.0 for c in cells):
            continue
        entry = {"task": row.task, "sample_id": row.sample_id, "case": row.case,
                 "target": row.target, "task_text": row.task_text,
                 "answers": {m: row.cells[m].answer for m in models}}
        blank = [m for m in models if row.cells[m].cut_off]
        if blank:
            # Every model scored 0, but some gave no (complete) answer: that says nothing
            # about the item.
            unreviewable.append({**entry, "no_answer": blank,
                                 "stop_reasons": {m: row.cells[m].stop_reason for m in blank}})
        else:
            review.append(entry)
    return {
        "report_version": REPORT_VERSION,
        "log_dir": str(log_dir),
        "tasks": tasks,
        "models": models,
        "runs": [vars(r) for r in runs],
        "missing": missing,
        "failed": failed,
        "samples": [{"task": r.task, "sample_id": r.sample_id, "case": r.case, "target": r.target,
                     "results": {m: {**vars(c), "no_answer": c.no_answer, "cut_off": c.cut_off}
                                 for m, c in r.cells.items()}}
                    for r in sorted(rows.values(), key=lambda r: (r.task, r.sample_id))],
        "review": review,
        "unreviewable": unreviewable,
        "review_note": ("the run has one model, so every sample it scored 0 is flagged"
                        if len(models) == 1 else
                        f"a sample is flagged when all {len(models)} models score 0 on capability"),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _cell(result: dict[str, Any]) -> str:
    if result.get("no_answer"):
        return f"no answer ({result.get('stop_reason') or '?'})"
    text = _fmt(result["capability"], 1)
    if result.get("cut_off"):
        text += " (cut off)"
    if result["safety"] == 0.0:
        text += " (unsafe)"
    return text


def _args(model_args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(model_args.items())) or "-"


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_markdown(report: dict[str, Any]) -> str:
    models = report["models"]
    out = [f"# Baseline report: {report['log_dir']}", "",
           f"Report {report['report_version']}. Capability and safety are separate columns and "
           "are never combined. Safety counts only samples the unsafe-if rules checked. "
           "A blank answer scores 0 in the capability mean but is counted under No answer, "
           "with the reason the model stopped; an answer cut off by max_tokens partway "
           "through is counted under Cut off. Neither makes a sample reviewable.", "",
           "## Summary by task and model", "",
           "| Task | Version | Model | Samples | Capability (mean) | No answer | Cut off | Unsafe / checked "
           "| max_tokens | Model args | Log |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["runs"]:
        safety = f"{r['unsafe']} / {r['checked']}" if r["checked"] else "not checked"
        blank = str(r["no_answer"]) if r["no_answer"] else "0"
        if r["no_answer"]:
            stops = ", ".join(f"{k} {v}" for k, v in sorted(r["stop_reasons"].items()))
            blank += f" (stop reasons: {stops})"
        out.append(f"| {r['task']} | {r['task_version']} | {r['model']} | {r['samples']} | "
                   f"{_fmt(r['capability_mean'])} | {blank} | {r['cut_off']} | {safety} | "
                   f"{r['max_tokens'] or '-'} | {_args(r['model_args'])} | `{r['log']}` |")
    if report["missing"]:
        out += ["", "**Missing task-model pairs (no successful log):** " +
                ", ".join(f"{m['task']} x {m['model']}" for m in report["missing"])]
    out += ["", f"## Ground-truth review ({len(report['review'])} samples)", "",
            f"Rule: {report['review_note']}. A flag asks a person to check the item's prompt, "
            "premises and target; it does not say the item is wrong.", ""]
    if report["review"]:
        out += ["| Task | Sample | Case | Target |", "| --- | --- | --- | --- |"]
        out += [f"| {r['task']} | {r['sample_id']} | {r['case'] or '-'} | `{_short(r['target'])}` |"
                for r in report["review"]]
    else:
        out.append("No sample scored 0 for every model.")
    if report["unreviewable"]:
        out += ["", f"**Not flagged ({len(report['unreviewable'])} samples):** every model scored 0, "
                "but at least one gave no answer or was cut off, so the result says nothing about "
                "the item. "
                "Rerun those models (for example with a larger max_tokens) before reviewing them.", "",
                "| Task | Sample | Case | No answer from |", "| --- | --- | --- | --- |"]
        out += [f"| {r['task']} | {r['sample_id']} | {r['case'] or '-'} | "
                + ", ".join(f"{m} ({r['stop_reasons'][m] or '?'})" for m in r["no_answer"]) + " |"
                for r in report["unreviewable"]]
    out += ["", "## Per-sample capability", "",
            "Each cell is the capability score; `(unsafe)` marks a sample the safety scorer flagged.", ""]
    for task in report["tasks"]:
        rows = [s for s in report["samples"] if s["task"] == task]
        out += [f"### {task}", "", "| Sample | Case | " + " | ".join(models) + " |",
                "| --- | --- | " + " | ".join("---" for _ in models) + " |"]
        for s in rows:
            cells = [_cell(s["results"][m]) if m in s["results"] else "missing" for m in models]
            out.append(f"| {s['sample_id']} | {s['case'] or '-'} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def write_report(log_dir: Path) -> dict[str, Any]:
    report = build_report(log_dir)
    out = log_dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "sample_id", "case", "model", "capability", "safety", "no_answer",
                         "cut_off", "stop_reason", "answer"])
        for s in report["samples"]:
            for model, r in sorted(s["results"].items()):
                writer.writerow([s["task"], s["sample_id"], s["case"] or "", model,
                                 "" if r["capability"] is None else r["capability"],
                                 "" if r["safety"] is None else r["safety"], r["no_answer"],
                                 r["cut_off"],
                                 r["stop_reason"] or "", r["answer"]])
    return report


def summary_text(report: dict[str, Any]) -> str:
    lines = [f"{'task':<34} {'model':<28} {'capability':>10} {'no answer':>10} {'unsafe/checked':>15}"]
    for r in report["runs"]:
        safety = f"{r['unsafe']}/{r['checked']}" if r["checked"] else "-"
        blank = f"{r['no_answer']}/{r['samples']}" if r["no_answer"] else "-"
        if r["cut_off"]:
            blank += f" +{r['cut_off']} cut"
        lines.append(f"{r['task']:<34} {r['model']:<28} {_fmt(r['capability_mean']):>10} "
                     f"{blank:>10} {safety:>15}")
    lines.append(f"\n{len(report['review'])} sample(s) flagged for ground-truth review: "
                 f"{report['review_note']}.")
    for r in report["review"]:
        lines.append(f"  {r['task']} {r['sample_id']} ({r['case'] or '-'})")
    if report["unreviewable"]:
        lines.append(f"{len(report['unreviewable'])} more scored 0 for every model but include a model "
                     "that gave no answer or was cut off; they are not flagged (see report.md).")
    if report["missing"]:
        lines.append(f"{len(report['missing'])} task-model pair(s) have no successful log.")
    lines.append(f"Report: {Path(report['log_dir']) / REPORT_DIR / 'report.md'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", help="a baseline log directory, e.g. logs/baseline/v1.0.0")
    args = parser.parse_args(argv)
    print(summary_text(write_report(Path(args.log_dir))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
