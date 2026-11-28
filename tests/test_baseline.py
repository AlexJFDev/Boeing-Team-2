"""Checks for the 3.8 baseline runner and report (issue #63).

Demo (issue #63): run the baseline evaluation and show per-task scores and EvalLogs for
every task in the ADS-B and ARINC 429 suites. Then show a task that scores 0 for every
model flagged for ground-truth review.

Acceptance criteria checked here:
- When the baseline config runs, every task runs against every model through
  ``eval_set``, one EvalLog per pair, and the report gives each pair's capability and
  safety separately, with each sample's score per model and its log file.
- When every model scores 0 on a sample, it is flagged for ground-truth review; a sample
  that any model gets right, or that a model has no score for, is not (negative case).
- When the config leaves out the model list, temperature, seed or corpus version, or
  names an unknown key or a missing task, it is refused before any model call, with exit
  code 2 and no log (negative case).

The runs here use scripted stand-in models (scripts/rehearse_baseline.py), because a
real model needs a server; the real baseline is ``bench baseline`` on a machine with one.
"""

from __future__ import annotations

import codecs
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import baseline_report
from runner import baseline, cli

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import rehearse_baseline as rehearsal  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "baseline_v1.0.0.yaml"
SMALL_TASKS = ["tasks/b2_arinc429_ssm_interpretation.py", "tasks/b3_arinc429_bus_conformance.py",
               "tasks/b4_arinc429_fault_vs_injection.py"]


def config(**changes: Any) -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(changes)
    return cfg


def write(tmp_path: Path, cfg: dict[str, Any]) -> Path:
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# --- The config --------------------------------------------------------------------


def test_shipped_config_covers_every_adsb_and_arinc429_suite() -> None:
    cfg = baseline.load_and_check(CONFIG)
    assert [Path(t).stem.split("_")[0] for t in cfg["tasks"]] == ["a1", "a2", "b1", "b2", "b3", "b4"]
    assert len(cfg["models"]) >= 2
    assert cfg["generate_config"]["temperature"] == 0 and cfg["generate_config"]["seed"] == 42


def test_arc_config_fits_the_arc_limits() -> None:
    cfg = baseline.load_and_check(REPO_ROOT / "configs" / "baseline_v1.2.0.yaml")
    assert cfg["tasks"] == baseline.load_and_check(CONFIG)["tasks"]  # same six suites as v1.0.0
    assert all(m.startswith("vllm/") for m in cfg["models"]) and len(cfg["models"]) == 3
    gen = cfg["generate_config"]
    assert gen["temperature"] == 0 and gen["seed"] == 42
    assert gen["max_tokens"] <= 8000  # ARC's non-streaming cap
    assert gen["max_connections"] * len(cfg["models"]) <= 20  # ARC: 20 in flight per user
    assert gen["max_connections"] <= 10  # ARC: 10 in flight per model


@pytest.mark.parametrize(("change", "message"), [
    ({"models": None}, "missing required field models"),
    ({"generate_config": {"seed": 42}}, "generate_config.temperature"),
    ({"generate_config": {"temperature": 0}}, "generate_config.seed"),
    ({"metadata": {}}, "metadata.corpus_version"),
    ({"model": "ollama/llama3.1"}, "unknown field model"),
    ({"generate_config": {"temperature": 0, "seed": 42, "temp": 1}}, "unknown field generate_config.temp"),
    ({"tasks": ["tasks/no_such_task.py"]}, "does not exist"),
    ({"models": ["ollama/llama3.1", "ollama/llama3.1"]}, "same entry twice"),
    ({"models": []}, "non-empty list"),
])
def test_incomplete_config_is_refused(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(baseline.BaselineConfigError, match=message):
        baseline.load_and_check(write(tmp_path, config(**change)))


def test_refused_config_exits_2_before_any_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(baseline, "LOG_ROOT", tmp_path / "logs")
    def no_run(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("the run must not start")
    monkeypatch.setattr(baseline, "run", no_run)
    assert cli.main(["baseline", str(write(tmp_path, config(metadata={})))]) == 2
    assert "metadata.corpus_version" in capsys.readouterr().err
    assert not (tmp_path / "logs").exists()


def test_unreachable_vllm_endpoint_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setattr(baseline, "load_dotenv", lambda: None)  # never read a developer's .env
    monkeypatch.setattr(baseline, "run", lambda *a, **k: pytest.fail("the run must not start"))
    path = write(tmp_path, config(models=["vllm/gpt-oss-120b"]))
    assert baseline.main([str(path)]) == 2


def test_non_vllm_models_record_no_endpoint() -> None:
    assert baseline.endpoint_metadata(["ollama/llama3.1", "mockllm/x"]) == {}


def test_log_dir_carries_the_baseline_version() -> None:
    assert baseline.log_dir_for(config()).name == "v1.0.0"


# --- A rehearsal run ------------------------------------------------------------------


@pytest.fixture(scope="module")
def rehearsal_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = config(tasks=SMALL_TASKS)
    log_dir = tmp_path_factory.mktemp("baseline") / "v1.0.0"
    prompts = rehearsal.samples_by_prompt()
    models = [rehearsal.standin(n, prompts) for n in rehearsal.TRAPS]
    assert baseline.run(cfg, log_dir, models=models, display="none")
    return log_dir


@pytest.fixture(scope="module")
def report(rehearsal_dir: Path) -> dict[str, Any]:
    return baseline_report.write_report(rehearsal_dir)


def test_one_evallog_per_task_and_model(report: dict[str, Any], rehearsal_dir: Path) -> None:
    pairs = {(r["task"], r["model"]) for r in report["runs"]}
    assert len(pairs) == len(report["runs"]) == 3 * 2
    assert report["missing"] == [] and report["failed"] == []
    for r in report["runs"]:
        assert (rehearsal_dir / r["log"]).is_file()
        assert r["temperature"] == 0 and r["seed"] == 42 and r["task_version"]


def test_capability_and_safety_are_separate(report: dict[str, Any]) -> None:
    b4 = {r["model"]: r for r in report["runs"] if r["task"] == "b4_arinc429_fault_vs_injection"}
    a, b = b4["mockllm/standin-trap-a"], b4["mockllm/standin-trap-b"]
    assert a["capability_scorer"] == "b4_triage_scorer"
    assert a["capability_mean"] == b["capability_mean"] == pytest.approx(0.6)  # same capability...
    assert (a["unsafe"], a["checked"]) == (6, 6) and (b["unsafe"], b["checked"]) == (0, 6)  # ...different safety
    for r in report["runs"]:
        assert not any("combined" in k for k in r)


def test_per_sample_scores_by_model(report: dict[str, Any]) -> None:
    by_id = {(s["task"], s["sample_id"]): s for s in report["samples"]}
    assert len(by_id) == 14 + 16 + 15
    b2_01 = by_id[("b2_arinc429_ssm_interpretation", "b2-01")]["results"]
    # Field-level scoring (B2 task 1.0.1): swapped tables get both SSM bits right and both
    # meanings wrong (0.5); one table for both words gets one meaning wrong (0.75).
    assert {m: r["capability"] for m, r in b2_01.items()} == {
        "mockllm/standin-trap-a": 0.5, "mockllm/standin-trap-b": 0.75}
    for result in b2_01.values():
        assert result["answer"]  # raw answer kept for rescoring


def test_review_flags_exactly_the_samples_every_model_scored_0(report: dict[str, Any]) -> None:
    flagged = {(r["task"], r["sample_id"]) for r in report["review"]}
    expected = {(s["task"], s["sample_id"]) for s in report["samples"]
                if all(r["capability"] == 0.0 for r in s["results"].values())}
    assert flagged == expected
    # B2's traps earn partial credit under the field-level scorer, so only B3, scored on
    # the whole violation list, has items every stand-in scored 0 on.
    assert {t for t, _ in flagged} == {"b3_arinc429_bus_conformance"}
    assert not any(t == "b4_arinc429_fault_vs_injection" for t, _ in flagged)  # each B4 item: one stand-in right
    controls = [s for s in report["samples"] if s["case"] in ("ssm_01", "ssm_10")]
    assert controls and not any((s["task"], s["sample_id"]) in flagged for s in controls)
    entry = report["review"][0]
    assert set(entry["answers"]) == set(report["models"]) and entry["target"]


def test_report_files(report: dict[str, Any], rehearsal_dir: Path) -> None:
    out = rehearsal_dir / "report"
    md = (out / "report.md").read_text()
    assert "## Ground-truth review (" in md and "| b3-07 |" in md and "(unsafe)" in md
    assert json.loads((out / "report.json").read_text())["review"] == report["review"]
    rows = (out / "per_sample.csv").read_text().splitlines()
    assert len(rows) == 1 + 45 * 2
    assert rows[0].endswith("no_answer,cut_off,stop_reason,answer")


def test_report_files_are_utf8_whatever_the_os_default(rehearsal_dir: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    # On Windows the default encoding is cp1252, which can't write e.g. "CO\u2082" from a
    # model's answer: the v1.4.0 baseline crashed writing per_sample.csv.
    copy = tmp_path / "v1.0.0"
    shutil.copytree(rehearsal_dir, copy)
    shutil.rmtree(copy / "report")
    text = "CO\u2082 \u00b15% \u2014 \u2713"
    real = baseline_report.build_report

    def with_unicode(log_dir: Path) -> dict[str, Any]:
        report = real(log_dir)
        first = report["samples"][0]["results"]
        first[next(iter(first))]["answer"] = text
        return report

    monkeypatch.setattr(baseline_report, "build_report", with_unicode)
    baseline_report.write_report(copy)
    csv_bytes = (copy / "report" / "per_sample.csv").read_bytes()
    assert csv_bytes.startswith(codecs.BOM_UTF8)  # so spreadsheets read it as UTF-8
    assert text in csv_bytes.decode("utf-8-sig")
    saved = json.loads((copy / "report" / "report.json").read_text(encoding="utf-8"))
    assert text in [r["answer"] for r in saved["samples"][0]["results"].values()]

    # No file in the report writer may fall back to the platform default encoding.
    script = (f"import pathlib; from bench import baseline_report; "
              f"baseline_report.write_report(pathlib.Path({str(copy)!r}))")
    subprocess.run([sys.executable, "-X", "warn_default_encoding",
                    "-W", "error::EncodingWarning:bench.baseline_report", "-c", script],
                   cwd=Path(__file__).resolve().parents[1], check=True)


def test_a_missing_pair_is_reported_and_never_flagged(rehearsal_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "v1.0.0"
    shutil.copytree(rehearsal_dir, copy)
    shutil.rmtree(copy / "report")
    report = baseline_report.build_report(copy)
    victim = next(r for r in report["runs"] if r["task"] == "b3_arinc429_bus_conformance")
    (copy / victim["log"]).unlink()
    report = baseline_report.build_report(copy)
    assert {"task": victim["task"], "model": victim["model"]} in report["missing"]
    assert not any(r["task"] == victim["task"] for r in report["review"])


def test_one_model_flags_every_miss(rehearsal_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "one"
    copy.mkdir()
    for r in baseline_report.build_report(rehearsal_dir)["runs"]:
        if r["model"] == "mockllm/standin-trap-b":
            shutil.copy(rehearsal_dir / r["log"], copy / r["log"])
    report = baseline_report.build_report(copy)
    assert "one model" in report["review_note"]
    misses = {(s["task"], s["sample_id"]) for s in report["samples"]
              if s["results"]["mockllm/standin-trap-b"]["capability"] == 0.0}
    assert {(r["task"], r["sample_id"]) for r in report["review"]} == misses


def test_report_only_rebuilds_from_logs(rehearsal_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["baseline", "--report-only", str(rehearsal_dir)]) == 0
    out = capsys.readouterr().out
    assert "flagged for ground-truth review" in out and "b3_arinc429_bus_conformance" in out


@pytest.fixture(scope="module")
def blank_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """trap-a plus a stand-in that returns no text on B3, as a model cut off by max_tokens does."""
    from inspect_ai.model import ModelOutput, ModelUsage, get_model

    prompts = rehearsal.samples_by_prompt()

    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        sample = prompts[messages[-1].text]
        sub = (sample.metadata or {}).get("subfamily")
        text = "" if sub == "B3" else rehearsal.answer(sample, "standin-trap-b")
        output = ModelOutput.from_content("mockllm/standin-blank", text)
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output

    log_dir = tmp_path_factory.mktemp("blank") / "v1.0.0"
    models = [rehearsal.standin("standin-trap-a", prompts),
              get_model("mockllm/standin-blank", custom_outputs=respond)]
    assert baseline.run(config(tasks=SMALL_TASKS), log_dir, models=models, display="none")
    return baseline_report.write_report(log_dir)


def test_blank_answers_are_reported_as_no_answer(blank_report: dict[str, Any]) -> None:
    b3 = next(r for r in blank_report["runs"]
              if r["task"] == "b3_arinc429_bus_conformance" and r["model"] == "mockllm/standin-blank")
    assert b3["no_answer"] == 16 and b3["stop_reasons"] == {"stop": 16}
    assert b3["capability_mean"] == 0.0  # the mean is still the scorer's
    sample = next(s for s in blank_report["samples"] if s["sample_id"] == "b3-03")
    assert sample["results"]["mockllm/standin-blank"]["no_answer"] is True
    assert sample["results"]["mockllm/standin-trap-a"]["no_answer"] is False


def test_blank_answers_keep_samples_out_of_the_review(blank_report: dict[str, Any]) -> None:
    flagged = {(r["task"], r["sample_id"]) for r in blank_report["review"]}
    assert not any(t == "b3_arinc429_bus_conformance" for t, _ in flagged)
    unreviewable = {r["sample_id"] for r in blank_report["unreviewable"]}
    assert unreviewable == {f"b3-{n:02d}" for n in range(1, 17)}
    assert blank_report["unreviewable"][0]["no_answer"] == ["mockllm/standin-blank"]
    assert flagged == set()  # B2's traps earn partial credit, and B3 is unreviewable


def test_report_records_max_tokens_and_shows_no_answer(blank_report: dict[str, Any]) -> None:
    assert {r["max_tokens"] for r in blank_report["runs"]} == {1024}
    md = render = baseline_report.render_markdown(blank_report)
    assert "no answer (stop)" in md and "**Not flagged (16 samples):**" in render
    assert "max_tokens" in md


def test_empty_log_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        baseline_report.build_report(tmp_path)
