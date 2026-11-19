"""Run resume and per-task timeout handling (WBS 1.20, issue #28).

Acceptance criteria:
- When an interrupted run is relaunched with ``--resume``, completed tasks are skipped
  and no EvalLog is duplicated. Negative: without ``--resume`` a relaunch writes a
  second log, and an unfinished task is still run.
- When a task exceeds its time limit, it is marked timed-out and the run continues with
  the next task. Negative: a task inside its limit is ``completed``, and a config with
  no ``time_limit`` records none.

Each run is a real ``python -m runner.cli`` subprocess against a small task file in a
temp directory (mockllm/model, no server), so an interruption is a genuine process
death rather than a simulated one.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml
from inspect_ai.log import list_eval_logs, read_eval_log

from runner.resume import task_status

REPO = Path(__file__).resolve().parent.parent

TASKS = '''
import asyncio, os
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, solver


@solver
def work(name):
    async def solve(state, gen):
        with open("runs.txt", "a") as f:
            f.write(name + "\\n")
        if os.environ.get("CRASH_IN") == name:
            await asyncio.sleep(3)  # let finished tasks write their logs first
            os._exit(9)  # the process dies mid-task, as on Ctrl-C or power loss
        await asyncio.sleep(float(os.environ.get("SLEEP_" + name, "0")))
        return state
    return solve


def make(name):
    return Task(
        dataset=[Sample(input="Reply: ready", target="ready")],
        solver=[work(name), generate()],
        scorer=includes(),
    )


@task
def first():
    return make("first")


@task
def second():
    return make("second")
'''


def setup(tmp_path, **extra):
    (tmp_path / "tasks.py").write_text(TASKS)
    cfg = {
        "task": "tasks.py",
        "model": "mockllm/model",
        "generate_config": {"temperature": 0, "seed": 42},
        "metadata": {"corpus_version": "synthetic-0.1.0"},
        **extra,
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(cfg))


def bench(tmp_path, *args, **env):
    return subprocess.run(
        [sys.executable, "-m", "runner.cli", "run", "run.yaml", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), **env},
    )


def headers(tmp_path):
    return [read_eval_log(i, header_only=True) for i in list_eval_logs(str(tmp_path / "logs"))]


def ran(tmp_path):
    path = tmp_path / "runs.txt"
    return path.read_text().split() if path.exists() else []


def logs_by_task(tmp_path):
    out: dict[str, list] = {}
    for h in headers(tmp_path):
        out.setdefault(h.eval.task, []).append(h)
    return out


# --- resume -------------------------------------------------------------------------


def test_resume_skips_completed_tasks_and_writes_no_duplicate_log(tmp_path):
    setup(tmp_path)
    interrupted = bench(tmp_path, "--resume", CRASH_IN="second")
    assert interrupted.returncode != 0
    before = logs_by_task(tmp_path)
    assert before["first"][0].status == "success"
    assert before["second"][0].status == "started"  # the interrupted attempt
    first_log = before["first"][0].location
    first_bytes = Path(first_log.removeprefix("file://")).read_bytes()

    resumed = bench(tmp_path, "--resume")
    assert resumed.returncode == 0, resumed.stderr

    assert ran(tmp_path) == ["first", "second", "second"]  # first was not re-run
    after = logs_by_task(tmp_path)
    assert {t: len(v) for t, v in after.items()} == {"first": 1, "second": 1}
    assert after["first"][0].location == first_log
    assert Path(first_log.removeprefix("file://")).read_bytes() == first_bytes
    assert after["second"][0].status == "success"
    assert "first: completed" in resumed.stdout and "second: completed" in resumed.stdout


def test_resume_of_a_finished_run_runs_nothing(tmp_path):
    setup(tmp_path)
    assert bench(tmp_path, "--resume").returncode == 0
    names = sorted(h.location for h in headers(tmp_path))
    assert bench(tmp_path, "--resume").returncode == 0
    assert ran(tmp_path) == ["first", "second"]
    assert sorted(h.location for h in headers(tmp_path)) == names


def test_without_resume_a_relaunch_duplicates_the_logs(tmp_path):
    """Negative case: the flag is what prevents duplicates."""
    setup(tmp_path)
    assert bench(tmp_path).returncode == 0
    assert bench(tmp_path).returncode == 0
    assert ran(tmp_path) == ["first", "second"] * 2
    assert {t: len(v) for t, v in logs_by_task(tmp_path).items()} == {"first": 2, "second": 2}


def test_resume_does_not_skip_a_task_that_never_finished(tmp_path):
    """Negative case: only a successful log counts as done."""
    setup(tmp_path)
    assert bench(tmp_path, "--resume", CRASH_IN="first").returncode != 0
    assert logs_by_task(tmp_path)["first"][0].status == "started"
    assert bench(tmp_path, "--resume").returncode == 0
    assert ran(tmp_path).count("first") == 2
    assert all(h.status == "success" for h in headers(tmp_path))


def test_resume_uses_the_log_dir_option(tmp_path):
    setup(tmp_path)
    assert bench(tmp_path, "--resume", "--log-dir", "mine").returncode == 0
    assert len(list_eval_logs(str(tmp_path / "mine"))) == 2
    assert not (tmp_path / "logs").exists()


def test_log_dir_without_resume_is_a_usage_error(tmp_path):
    setup(tmp_path)
    result = bench(tmp_path, "--log-dir", "mine")
    assert result.returncode == 2
    assert "--resume" in result.stderr
    assert not (tmp_path / "runs.txt").exists()


# --- per-task timeout ---------------------------------------------------------------


def test_task_over_its_time_limit_is_marked_timed_out_and_the_run_continues(tmp_path):
    setup(tmp_path, time_limit=1)
    result = bench(tmp_path, SLEEP_first="30")
    assert result.returncode == 0, result.stderr
    assert "first: timed-out" in result.stdout
    assert "second: completed" in result.stdout

    by_task = logs_by_task(tmp_path)
    first = read_eval_log(by_task["first"][0].location)
    assert task_status(first).status == "timed-out"
    [sample] = first.samples
    assert sample.limit is not None and sample.limit.type == "time"
    assert first.eval.metadata["time_limit_seconds"] == 1
    # the next task ran and finished normally
    assert ran(tmp_path) == ["first", "second"] or ran(tmp_path) == ["second", "first"]
    second = read_eval_log(by_task["second"][0].location)
    assert task_status(second).status == "completed"


def test_timed_out_task_is_not_re_run_on_resume(tmp_path):
    setup(tmp_path, time_limit=1)
    assert bench(tmp_path, "--resume", SLEEP_first="30").returncode == 0
    resumed = bench(tmp_path, "--resume", SLEEP_first="30")
    assert resumed.returncode == 0
    assert ran(tmp_path) == ["first", "second"] or ran(tmp_path) == ["second", "first"]
    assert "first: timed-out" in resumed.stdout
    assert {t: len(v) for t, v in logs_by_task(tmp_path).items()} == {"first": 1, "second": 1}


def test_task_inside_its_limit_is_completed_not_timed_out(tmp_path):
    """Negative case."""
    setup(tmp_path, time_limit=30)
    result = bench(tmp_path, SLEEP_first="0.2")
    assert result.returncode == 0
    assert "first: completed" in result.stdout
    assert "timed-out" not in result.stdout


def test_no_time_limit_means_none_is_recorded_or_applied(tmp_path):
    setup(tmp_path)
    assert bench(tmp_path, SLEEP_first="1.5").returncode == 0
    first = read_eval_log(logs_by_task(tmp_path)["first"][0].location)
    assert "time_limit_seconds" not in first.eval.metadata
    assert task_status(first).status == "completed"


def test_bad_time_limit_is_refused_before_any_model_call(tmp_path):
    for bad in (0, -5, 1.5, "10", True):
        setup(tmp_path, time_limit=bad)
        result = bench(tmp_path)
        assert result.returncode == 2, bad
        assert "time_limit" in result.stderr
        assert not (tmp_path / "logs").exists() and not (tmp_path / "runs.txt").exists()
