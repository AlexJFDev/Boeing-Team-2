"""Run resume and per-task timeout reporting (WBS 1.20, issue #28).

Inspect already does the heavy lifting; this module is only the domain-specific glue.

* Resume: ``bench run --resume`` runs the config's task(s) with ``inspect_ai.eval_set``
  into a fixed log directory. ``eval_set`` skips a task whose log there has status
  ``success`` and runs the rest, so a relaunch after an interruption never re-runs a
  completed task. ``prune_superseded`` then removes the leftover log header an
  interrupted attempt wrote (status ``started``) once a newer log for the same task
  exists, so the directory holds one EvalLog per task.
* Timeout: the config's ``time_limit`` is Inspect's per-sample ``time_limit``. A sample
  that exceeds it ends with ``EvalSample.limit.type == "time"`` and the run carries on
  with the next sample and task. ``task_status`` marks a task whose log has such a
  sample ``timed-out``. The mark is derived from the log, never stored separately, so
  it cannot disagree with it.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

Status = Literal["completed", "timed-out", "failed", "incomplete"]
TIME_LIMITS = ("time", "working")


@dataclass(frozen=True)
class TaskStatus:
    task: str
    status: Status
    location: str
    timed_out_samples: int = 0


def timed_out_samples(log: EvalLog) -> int:
    """Samples in a full log that ended on a time limit (``time`` or ``working``)."""
    return sum(
        1 for s in log.samples or [] if s.limit is not None and s.limit.type in TIME_LIMITS
    )


def task_status(log: EvalLog) -> TaskStatus:
    """A task is ``completed`` when its log has status ``success`` (the test
    ``eval_set`` uses to skip it), and ``timed-out`` when it finished but at least one
    sample hit the time limit. ``failed`` is an error or cancelled log; ``incomplete``
    is a log still marked ``started`` (an interrupted attempt)."""
    if log.status == "success":
        n = timed_out_samples(log)
        return TaskStatus(log.eval.task, "timed-out" if n else "completed", log.location, n)
    status: Status = "incomplete" if log.status == "started" else "failed"
    return TaskStatus(log.eval.task, status, log.location)


def prune_superseded(log_dir: str) -> list[str]:
    """Delete ``started`` logs that a later log for the same task has replaced.

    An interrupted run leaves a header-only log behind; the relaunch writes a new log
    for the task. Only those empty headers are removed, never a log with results.
    """
    by_task = defaultdict(list)
    for info in list_eval_logs(log_dir, recursive=False):
        header = read_eval_log(info, header_only=True)
        by_task[header.eval.task_id].append((header.eval.created, header.status, info.name))
    removed = []
    for entries in by_task.values():
        entries.sort()
        for _, status, name in entries[:-1]:
            if status == "started":
                Path(name.removeprefix("file://")).unlink()
                removed.append(name)
    return removed


def summarize(log_dir: str) -> list[TaskStatus]:
    """One status per task in ``log_dir``, newest log for each task."""
    latest: dict[str, EvalLog] = {}
    for info in list_eval_logs(log_dir, recursive=False):
        log = read_eval_log(info)
        key = log.eval.task_id
        if key not in latest or log.eval.created > latest[key].eval.created:
            latest[key] = log
    return sorted((task_status(log) for log in latest.values()), key=lambda s: s.task)


def format_summary(statuses: list[TaskStatus]) -> str:
    lines = ["task status:"]
    for s in statuses:
        extra = f" ({s.timed_out_samples} sample(s) hit the time limit)" if s.timed_out_samples else ""
        lines.append(f"  {s.task}: {s.status}{extra}")
    return "\n".join(lines)
