"""Integration test: ADS-B and ARINC 429 suites end to end (WBS 1.22, issue #30).

Demo (issue): run the ADS-B and ARINC 429 suites end to end with no manual steps and
show them load, execute, score and log. Then make one stage return malformed output
and show the test fail naming that stage.

Acceptance criteria:
- When the integration test runs, the ADS-B (A1, A2) and ARINC 429 (B1, B2) suites
  load, execute, score and log with no manual steps.
- When any stage returns malformed output, the test fails and names the stage that
  broke.

Each suite goes through the real entry point: a run config handed to ``bench run``
(``runner.cli.main``), with ``mockllm/model``. Four stages are checked separately, and a
failure is raised as ``StageFailure`` whose message starts with the stage name:

    load     the task builds, with a non-empty dataset of well-formed samples
    execute  the run exits 0, the log status is success and every sample has a
             non-empty model response
    score    every sample has exactly the expected scorers, each with a numeric value,
             a parsed answer and a scorer_version
    log      the log read back from disk has the run's temperature, seed, corpus
             version and task version, and every sample passes the provenance schema
             (corpus hash, prompt template hash, raw output beside the parsed answer)

mockllm returns filler text, so the filler run scores 0 everywhere and only proves the
harness runs. A second run feeds each prompt its known-correct target (validated against
the reference decoders when the task builds) and must score 1.0 on the capability scorer,
so the score stage is exercised with real answers too.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage, get_model

import runner.preflight as preflight
from bench.provenance import ProvenanceError, validate_log
from runner.cli import main

TEMPERATURE = 0
SEED = 42
GEN_CONFIG = GenerateConfig(temperature=TEMPERATURE, seed=SEED)  # a Model object carries its own


@dataclass(frozen=True)
class Suite:
    name: str
    task_file: str
    corpus_version: str
    scorers: tuple[str, ...]  # expected score names, capability first
    version: str


def suites() -> list[Suite]:
    from bench.adsb import a1, a2
    from bench.arinc429 import b1, b2

    adsb, coords, arinc = "adsb_field_capability_scorer", "a2_cpr_coordinate_scorer", "arinc429_bnr_bcd_capability_scorer"
    unsafe = "unsafe_if_rule_scorer"
    # corpus_version names the task file; the task version adds scoring changes (rule 4).
    return [
        Suite("a1", "tasks/a1_adsb_decode.py", f"a1-{a1.A1_VERSION}", (adsb, unsafe), a1.A1_TASK_VERSION),
        Suite("a2", "tasks/a2_cpr_position.py", f"a2-{a2.A2_VERSION}", (coords, unsafe), a2.A2_TASK_VERSION),
        Suite("b1", "tasks/b1_arinc429_word_decode.py", f"b1-{b1.B1_VERSION}", (arinc,), b1.B1_TASK_VERSION),
        Suite("b2", "tasks/b2_arinc429_ssm_interpretation.py", f"b2-{b2.B2_VERSION}", (arinc,),
              b2.B2_TASK_VERSION),
    ]


SUITES = suites()
IDS = [s.name for s in SUITES]


class StageFailure(AssertionError):
    """A pipeline stage produced a missing or malformed result. Names the stage."""

    def __init__(self, stage: str, suite: str, problem: str) -> None:
        problem = " ".join(problem.split())
        if len(problem) > 300:
            problem = problem[:300] + " ..."
        super().__init__(f"stage '{stage}' failed for suite {suite}: {problem}")
        self.stage = stage


# --- Stage checks ---------------------------------------------------------------


def task_factory(suite: Suite) -> Callable[[], Any]:
    module = importlib.import_module(Path(suite.task_file).with_suffix("").as_posix().replace("/", "."))
    return getattr(module, Path(suite.task_file).stem)


def check_load(suite: Suite) -> None:
    """The task builds from its file and every sample is well formed."""
    factory = task_factory(suite)
    try:
        task = factory()
    except Exception as e:  # a loader or validator error is a load failure
        raise StageFailure("load", suite.name, f"task did not build: {e!r}") from e
    samples = list(task.dataset)
    if not samples:
        raise StageFailure("load", suite.name, "task has no samples")
    for s in samples:
        if not (isinstance(s.input, str) and s.input.strip()):
            raise StageFailure("load", suite.name, f"sample {s.id!r} has no prompt")
        if not (isinstance(s.target, str) and s.target.strip()):
            raise StageFailure("load", suite.name, f"sample {s.id!r} has no target")


def check_execute(suite: Suite, code: int, log: EvalLog | None) -> None:
    if code != 0:
        raise StageFailure("execute", suite.name, f"bench run exited {code}, expected 0")
    if log is None or log.status != "success":
        raise StageFailure("execute", suite.name, f"log status is {log and log.status!r}")
    if not log.samples:
        raise StageFailure("execute", suite.name, "log has no samples")
    for s in log.samples:
        if s.error is not None:
            raise StageFailure("execute", suite.name, f"sample {s.id!r} errored: {s.error.message}")
        if not (s.output and s.output.completion and s.output.completion.strip()):
            raise StageFailure("execute", suite.name, f"sample {s.id!r} has an empty model response")


def check_score(suite: Suite, log: EvalLog) -> None:
    assert log.samples is not None
    for s in log.samples:
        scores = s.scores or {}
        if tuple(scores) != suite.scorers:
            raise StageFailure(
                "score", suite.name,
                f"sample {s.id!r} has scorers {tuple(scores)}, expected {suite.scorers}",
            )
        for name, score in scores.items():
            if isinstance(score.value, bool) or not isinstance(score.value, (int, float)):
                raise StageFailure("score", suite.name, f"{name} on {s.id!r} has non-numeric value")
            if score.answer is None:
                raise StageFailure("score", suite.name, f"{name} on {s.id!r} has no parsed answer")
            if not (score.metadata or {}).get("scorer_version"):
                raise StageFailure("score", suite.name, f"{name} on {s.id!r} has no scorer_version")
    if log.results is None or len(log.results.scores) != len(suite.scorers):
        raise StageFailure("score", suite.name, "log results do not list one entry per scorer")


def check_log(suite: Suite, log: EvalLog) -> None:
    """The log as read back from disk carries the provenance a rerun needs."""
    cfg = log.eval.model_generate_config
    if cfg is None or cfg.temperature != TEMPERATURE or cfg.seed != SEED:
        raise StageFailure("log", suite.name, f"temperature/seed not recorded as {TEMPERATURE}/{SEED}")
    if (log.eval.metadata or {}).get("corpus_version") != suite.corpus_version:
        raise StageFailure("log", suite.name, "corpus_version not recorded in the log metadata")
    if log.eval.task_version != suite.version:
        raise StageFailure("log", suite.name, f"task version {log.eval.task_version!r} != {suite.version!r}")
    try:
        records = validate_log(log, suite.scorers[0])
    except ProvenanceError as e:
        raise StageFailure("log", suite.name, str(e)) from e
    assert log.samples is not None
    for record, s in zip(records, log.samples, strict=True):
        if record.raw_output != s.output.completion:  # rule 8: raw response kept
            raise StageFailure("log", suite.name, f"sample {s.id!r} raw output not kept")


# --- Driving a run through the CLI ----------------------------------------------


def run_suite(
    suite: Suite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: Any = None,
) -> tuple[int, EvalLog | None]:
    """``bench run`` with a run config for one suite; returns (exit code, log read back).

    ``model``, if given, replaces the config's ``mockllm/model`` string at the point
    where preflight hands it to Inspect, so the rest of the path is the real one.
    """
    log_dir = tmp_path / f"logs-{suite.name}"
    monkeypatch.setenv("INSPECT_LOG_DIR", str(log_dir))
    cfg = {
        "task": suite.task_file,
        "model": "mockllm/model",
        "generate_config": {"temperature": TEMPERATURE, "seed": SEED},
        "metadata": {"corpus_version": suite.corpus_version},
    }
    path = tmp_path / f"{suite.name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    if model is not None:
        real_eval = preflight.inspect_eval
        monkeypatch.setattr(
            preflight, "inspect_eval",
            lambda task, **kw: real_eval(task, **{**kw, "model": model, "display": "none"}),
        )
    code = main(["run", str(path)])
    infos = list_eval_logs(str(log_dir))
    return code, (read_eval_log(infos[0]) if infos else None)


def check_pipeline(
    suite: Suite, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: Any = None
) -> EvalLog:
    check_load(suite)
    code, log = run_suite(suite, tmp_path, monkeypatch, model)
    check_execute(suite, code, log)
    assert log is not None
    check_score(suite, log)
    check_log(suite, log)
    return log


def with_usage(text: str) -> ModelOutput:
    out = ModelOutput.from_content("mockllm/model", text)
    out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    return out


def answer_from_target(suite: Suite) -> Callable[..., ModelOutput]:
    """mockllm that answers each prompt with that sample's validated target."""
    task = task_factory(suite)()
    targets = {s.input: s.target for s in task.dataset}

    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        return with_usage(str(targets[messages[-1].text]))

    return get_model("mockllm/model", custom_outputs=respond, config=GEN_CONFIG)


# --- The happy path --------------------------------------------------------------


@pytest.mark.parametrize("suite", SUITES, ids=IDS)
def test_suite_loads_executes_scores_and_logs(suite, tmp_path, monkeypatch) -> None:
    log = check_pipeline(suite, tmp_path, monkeypatch)
    assert log.samples
    # mockllm returns filler, so the capability score is 0: this proves the harness runs.
    assert all(s.scores[suite.scorers[0]].value == 0 for s in log.samples)


@pytest.mark.parametrize("suite", SUITES, ids=IDS)
def test_known_correct_answers_score_one_through_the_same_path(suite, tmp_path, monkeypatch) -> None:
    log = check_pipeline(suite, tmp_path, monkeypatch, model=answer_from_target(suite))
    assert [s.scores[suite.scorers[0]].value for s in log.samples] == [1.0] * len(log.samples)


# --- The negative case: a malformed stage fails the test and is named --------------


def break_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench import loader

    real = loader.load_tasks

    def malformed(path, *a, **kw):
        dataset = real(path, *a, **kw)
        for s in dataset:
            s.target = ""  # a task with no ground truth
        return dataset

    monkeypatch.setattr(importlib.import_module("tasks.b1_arinc429_word_decode"), "load_tasks", malformed)


def break_execute(monkeypatch: pytest.MonkeyPatch) -> Any:
    return get_model("mockllm/model", custom_outputs=lambda *a: with_usage(""), config=GEN_CONFIG)


def break_score(monkeypatch: pytest.MonkeyPatch) -> None:
    import bench.scorers as scorers

    monkeypatch.setattr(scorers, "ARINC429_SCORER_VERSION", "")  # score without a scorer_version


def break_log(monkeypatch: pytest.MonkeyPatch) -> None:
    import bench.provenance_solver as solver

    monkeypatch.setattr(solver, "sha256_text", lambda text: "not-a-hash")  # malformed template hash


@pytest.mark.parametrize("stage", ["load", "execute", "score", "log"])
def test_a_malformed_stage_fails_the_test_and_names_the_stage(stage, tmp_path, monkeypatch) -> None:
    suite = next(s for s in SUITES if s.name == "b1")
    model = None
    if stage == "load":
        break_load(monkeypatch)
    elif stage == "execute":
        model = break_execute(monkeypatch)
    elif stage == "score":
        break_score(monkeypatch)
    else:
        break_log(monkeypatch)

    with pytest.raises(StageFailure, match=rf"^stage '{stage}' failed for suite b1") as failure:
        check_pipeline(suite, tmp_path, monkeypatch, model)
    assert failure.value.stage == stage


def test_a_stage_failure_is_an_assertion_error() -> None:
    # pytest reports it as a test failure, with the stage in the message.
    assert issubclass(StageFailure, AssertionError)
    assert "stage 'score'" in str(StageFailure("score", "b1", "x"))
