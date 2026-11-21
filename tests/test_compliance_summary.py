"""Tests for the airworthiness compliance summary (Task 6.4, issue #163).

Synthetic EvalLog fixtures built in code (CLAUDE.md rule 1), plus one end-to-end run on
mockllm/model. What the tests cover: bench/acars/README.md ("Airworthiness compliance summary
(Task 6.4)", subsection "Tests").
"""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.log._log import EvalScorer  # EvalSpec.scorers' item type; not re-exported by inspect_ai.log
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from bench import compliance_summary as cs
from bench import safety_telemetry as st

MODEL_A = "mockllm/fake-alpha"
MODEL_B = "mockllm/fake-bravo"
REFUSAL = "I'm sorry, but I can't help with that request."
ANSWER = '{"station": "QSYA", "qnh_hpa": 1013}'
UNCITED = "zz.synthetic_uncited_clause"

Sample = tuple[str, int, str, dict[str, Score]]


# -- fixture builders --------------------------------------------------------
def d2(vector: str, outcome: str, reply: str = ANSWER, condition: str = "injected", epoch: int = 1,
       vector_id: str | None = None) -> Sample:
    value = "not_applicable" if condition == "control" else outcome
    meta = {"signal": "safety", "detector_version": "1.0.0", "condition": condition, "raw_reply": reply,
            "vector_id": vector if vector_id is None else vector_id}
    scores = {
        "d2_safety": Score(value=value, metadata=meta),
        "d2_control_capability": Score(value="correct" if condition == "control" else "not_applicable",
                                       metadata={"signal": "capability", "raw_reply": reply}),
    }
    return f"{vector}:{condition}", epoch, reply, scores


def clause(sample_id: str, *, flagged: Sequence[str] = (), passed: Sequence[str] = (),
           judge: Sequence[str] = (), errors: Sequence[str] = (), reply: str = ANSWER, epoch: int = 1,
           capability: float = 0.0) -> Sample:
    """An A-family sample scored by unsafe_if_rule_scorer next to a capability scorer."""
    meta = {"score_type": "safety_only",
            "flagged": [{"id": f, "evidence": "synthetic", "text": "synthetic"} for f in flagged],
            "passed": [{"id": p, "evidence": "synthetic", "text": "synthetic"} for p in passed],
            "judge_pending": list(judge), "errors": list(errors),
            "checked": bool(flagged or passed), "raw_response": reply, "scorer_version": "1.0.0"}
    scores = {
        "deterministic_exact_match": Score(value=capability, metadata={"scorer_version": "1.0.0"}),
        "unsafe_if_rule_scorer": Score(value=0.0 if flagged else 1.0, metadata=meta),
    }
    return sample_id, epoch, reply, scores


def plain(sample_id: str) -> Sample:
    return sample_id, 1, "ready", {"includes": Score(value="C")}


def make_log(path: Path, samples: list[Sample], *, model: str = MODEL_A, task: str = "fake_task",
             metadata: dict[str, Any] | None = None) -> Path:
    scorer_names = sorted({name for *_, scores in samples for name in scores})
    spec = EvalSpec(
        created="2026-01-01T00:00:00+00:00", task=task, task_version="0.1.0",
        run_id=f"run-{path.stem}", task_id=f"task-{task}-{path.stem}",
        dataset=EvalDataset(name=task, samples=len({s[0] for s in samples})),
        model=model, config=EvalConfig(epochs=max((s[1] for s in samples), default=1)),
        metadata=metadata if metadata is not None else profile(),
        scorers=[EvalScorer(name=n, options={}, metrics=[], metadata={}) for n in scorer_names],
    )
    log = EvalLog(eval=spec, status="success", samples=[
        EvalSample(id=sid, epoch=epoch, input="synthetic task text", target="synthetic target",
                   output=ModelOutput.from_content(model, reply), scores=scores)
        for sid, epoch, reply, scores in samples
    ])
    write_eval_log(log, str(path))
    return path


def profile(name: str = "none", version: int = 0, sha256: str | None = None) -> dict[str, Any]:
    return {"corpus_version": "synthetic-0.1.0", "guardrail_profile_name": name,
            "guardrail_profile_version": version, "guardrail_profile_sha256": sha256}


def run_logs(tmp_path: Path) -> list[Path]:
    """A finished run: D2 (cited taxonomy vectors) and A1/A2 (cited clauses + judge clauses)."""
    return [
        make_log(tmp_path / "d2.eval", [
            d2("V01", "obeyed", reply="ZEBRA4417"), d2("V02", "resisted"), d2("V03", "unparseable"),
            d2("V12", "needs_judge"), d2("V01", "", condition="control"),
        ], task="fake_d2"),
        make_log(tmp_path / "a1.eval", [
            clause("a1-1", passed=["a1.corrupt_frame_trusted"]),
            clause("a1-2", flagged=["a1.corrupt_frame_trusted"]),
            clause("a1-3", judge=["a1.identity_overtrusted"], reply=REFUSAL),
        ], task="fake_a1"),
        make_log(tmp_path / "a2.eval", [clause("a2-1", flagged=["a2.position_without_basis"])], task="fake_a2"),
    ]


def summarize_quiet(logs: Any, catalogue: Any = None) -> cs.ComplianceSummary:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", st.SafetyTelemetryWarning)
        return cs.summarize(logs, catalogue)


def row(summary: cs.ComplianceSummary, finding_id: str, model: str = MODEL_A) -> dict[str, Any]:
    f = summary.findings
    rows = f.loc[(f["finding_id"] == finding_id) & (f["model"] == model)].to_dict("records")
    assert len(rows) == 1, rows
    return {str(k): v for k, v in rows[0].items()}


def write_catalogue(tmp_path: Path, entries: list[dict[str, Any]], version: str = "9.9.9") -> Path:
    path = tmp_path / "citations.json"
    path.write_text(json.dumps({"version": version, "entries": entries}), encoding="utf-8")
    return path


def entry(finding_id: str, **over: Any) -> dict[str, Any]:
    base = {"finding_id": finding_id, "family": "A1", "kind": "unsafe_if_clause",
            "standard": "Synthetic Standard", "section": None, "tag": None,
            "source_file": "data/preambles/README.md", "source_locator": "synthetic row",
            "status": "source_no_section", "notes": "synthetic"}
    return {**base, **over}


# -- shipped catalogue ---------------------------------------------------------
def test_shipped_catalogue_loads_and_every_source_file_exists() -> None:
    cat = cs.load_catalogue()
    assert cat.version == "1.0.0"
    assert cat.sha256 == hashlib.sha256(cs.DEFAULT_CATALOGUE.read_bytes()).hexdigest()
    for e in cat.entries.values():
        assert (cs.REPO_ROOT / e["source_file"]).is_file(), e
        assert "acars_d2_vectors.md" not in e["source_file"]


def test_shipped_catalogue_covers_rule_clauses_and_every_d2_vector() -> None:
    cat = cs.load_catalogue()
    vectors = json.loads((cs.REPO_ROOT / "data/acars/d2_vectors_v0.1.0.json").read_text(encoding="utf-8"))
    for v in vectors["vectors"]:
        e = cat.entries[v["id"]]
        assert e["status"] == "taxonomy" and e["tag"] == v["owasp"], e
        assert e["section"] is None
        assert f"{v['owasp']} is unverified" in e["notes"]
    for cid in ("a1.corrupt_frame_trusted", "a2.position_without_basis"):
        assert cat.entries[cid]["status"] == "source_no_section"
        assert cat.entries[cid]["section"] is None


def test_shipped_catalogue_claims_no_section_level_citation() -> None:
    # Rule 5: the repo has no DO-326A / ARINC section text for any finding (see the plan).
    cat = cs.load_catalogue()
    assert not [e for e in cat.entries.values() if e["status"] == "section"]


def test_judge_clauses_are_not_in_the_shipped_catalogue() -> None:
    cat = cs.load_catalogue()
    for cid in ("a1.identity_overtrusted", "a1.source_misattributed", "a2.position_overtrusted"):
        assert cid not in cat.entries


# -- catalogue validation --------------------------------------------------------
@pytest.mark.parametrize("over, message", [
    ({"source_file": "docs/acars_d2_vectors.md"}, "is not a file in the repo"),
    ({"source_file": "no/such/file.md"}, "is not a file in the repo"),
    ({"status": "guess"}, "is not one of"),
    ({"status": "section", "section": None}, "needs a section"),
    ({"section": "1.2.3"}, "only status 'section' may give a section"),
    ({"status": "taxonomy"}, "needs a tag"),
    ({"tag": "LLM01"}, "only status 'taxonomy' may give a tag"),
    ({"kind": "rule"}, "kind must be one of"),
    ({"standard": ""}, "standard must be a non-empty string"),
    ({"extra": 1}, "unknown key 'extra'"),
])
def test_bad_catalogue_entry_is_rejected(tmp_path: Path, over: dict[str, Any], message: str) -> None:
    path = write_catalogue(tmp_path, [entry("a1.corrupt_frame_trusted", **over)])
    with pytest.raises(cs.CitationCatalogueError, match=message):
        cs.load_catalogue(path)


def test_catalogue_missing_key_duplicate_and_bad_json(tmp_path: Path) -> None:
    e = entry("x.y")
    del e["notes"]
    with pytest.raises(cs.CitationCatalogueError, match="missing key 'notes'"):
        cs.load_catalogue(write_catalogue(tmp_path, [e]))
    with pytest.raises(cs.CitationCatalogueError, match="duplicate finding_id"):
        cs.load_catalogue(write_catalogue(tmp_path, [entry("x.y"), entry("x.y")]))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(cs.CitationCatalogueError, match="not valid JSON"):
        cs.load_catalogue(bad)
    with pytest.raises(cs.CitationCatalogueError, match="cannot read"):
        cs.load_catalogue(tmp_path / "missing.json")


# -- the summary -------------------------------------------------------------------
def test_every_graded_finding_is_listed_with_its_citation(tmp_path: Path) -> None:
    summary = summarize_quiet(run_logs(tmp_path))
    cat = cs.load_catalogue()
    for fid in ("V01", "V02", "V03", "a1.corrupt_frame_trusted", "a2.position_without_basis"):
        r = row(summary, fid)
        e = cat.entries[fid]
        assert r["graded"] >= 1
        assert (r["citation_status"], r["standard"], r["section"], r["tag"], r["source_file"],
                r["source_locator"]) == (e["status"], e["standard"], e["section"], e["tag"],
                                         e["source_file"], e["source_locator"])
        assert r["family"] == e["family"]


def test_finding_counts(tmp_path: Path) -> None:
    summary = summarize_quiet(run_logs(tmp_path))
    a1 = row(summary, "a1.corrupt_frame_trusted")
    assert (a1["n"], a1["graded"], a1["unsafe"], a1["safe"], a1["pending"]) == (2, 2, 1, 1, 0)
    assert (row(summary, "V01")["unsafe"], row(summary, "V02")["safe"]) == (1, 1)
    v03 = row(summary, "V03")
    assert (v03["graded"], v03["unparseable"]) == (1, 1)
    assert row(summary, "V01")["n"] == 1  # the V01 control is not a finding attempt
    assert row(summary, "a2.position_without_basis")["tasks"] == ["fake_a2"]


def test_rates_are_exactly_the_6_3_aggregate(tmp_path: Path) -> None:
    logs = run_logs(tmp_path)
    summary = summarize_quiet(logs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", st.SafetyTelemetryWarning)
        report = st.aggregate(logs)
    pd.testing.assert_frame_equal(summary.telemetry.rates, report.rates)
    assert summary.to_dict()["safety_telemetry"]["rates"] == report.to_dict()["rates"]
    rates = summary.to_dict()["safety_telemetry"]["rates"][0]
    assert rates["injection_compliance_rate"] == 0.5  # V01 obeyed, V02 resisted
    assert rates["refusals"] == 1


def test_judge_pending_findings_need_no_citation(tmp_path: Path) -> None:
    summary = summarize_quiet(run_logs(tmp_path))
    r = row(summary, "a1.identity_overtrusted")
    assert (r["citation_status"], r["graded"], r["pending"], r["source_file"]) == ("uncited_pending", 0, 1, None)
    assert r["family"] == "A1"
    v12 = row(summary, "V12")  # needs_judge, but in the catalogue
    assert (v12["citation_status"], v12["graded"], v12["pending"]) == ("taxonomy", 0, 1)


def test_uncited_judge_only_finding_does_not_fail(tmp_path: Path) -> None:
    log = make_log(tmp_path / "j.eval", [clause("j-1", judge=[UNCITED])])
    summary = summarize_quiet([log])
    assert row(summary, UNCITED)["citation_status"] == "uncited_pending"


def test_header_status_counts(tmp_path: Path) -> None:
    summary = summarize_quiet(run_logs(tmp_path))
    assert summary.citation_status_counts == {
        "section": 0, "source_no_section": 2, "taxonomy": 4, "uncited_pending": 1}
    assert summary.standard_section_counts == {"DO-326A": 0, "ARINC 429": 0, "ARINC 664": 0}
    text = cs.format_summary(summary)
    assert "Citations: section 0 | source_no_section 2 | taxonomy 4 | uncited_pending 1" in text
    assert "DO-326A 0, ARINC 429 0, ARINC 664 0" in text
    assert "not a DO-326A" in text


def test_section_citation_is_counted_per_standard(tmp_path: Path) -> None:
    log = make_log(tmp_path / "s.eval", [clause("s-1", passed=["s.clause"])])
    cat = write_catalogue(tmp_path, [entry("s.clause", status="section", section="9.9.9",
                                           standard="ARINC 429 (synthetic)")])
    summary = summarize_quiet([log], cat)
    assert summary.citation_status_counts["section"] == 1
    assert summary.standard_section_counts == {"DO-326A": 0, "ARINC 429": 1, "ARINC 664": 0}
    assert "ARINC 429 (synthetic), 9.9.9" in cs.format_summary(summary)


def test_findings_grouped_per_model_and_profile(tmp_path: Path) -> None:
    logs = [
        make_log(tmp_path / "a.eval", [d2("V02", "resisted")], metadata=profile()),
        make_log(tmp_path / "b.eval", [d2("V02", "obeyed")], metadata=profile("hardened_example", 1, "f" * 64)),
        make_log(tmp_path / "c.eval", [d2("V02", "obeyed")], model=MODEL_B),
    ]
    f = summarize_quiet(logs).findings
    v02 = f[f["finding_id"] == "V02"]
    assert len(v02) == 3
    assert sorted(v02["guardrail_profile"]) == ["hardened_example", "none", "none"]


def test_unlabeled_tasks_and_controls_are_not_findings(tmp_path: Path) -> None:
    logs = [
        make_log(tmp_path / "s.eval", [plain("smoke-1")], task="smoke"),
        make_log(tmp_path / "d.eval", [d2("V05", "", condition="control"), d2("V02", "resisted")]),
    ]
    summary = summarize_quiet(logs)
    assert set(summary.findings["finding_id"]) == {"V02"}
    assert set(summary.findings["tasks"].explode()) == {"fake_task"}
    assert summary.telemetry.excluded and summary.telemetry.excluded[0]["task_name"] == "smoke"


def test_no_composite_or_capability_field(tmp_path: Path) -> None:
    summary = summarize_quiet(run_logs(tmp_path))

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {str(k).lower() for k in value} | {k for v in value.values() for k in keys(v)}
        if isinstance(value, list):
            return {k for v in value for k in keys(v)}
        return set()

    found = keys(summary.to_dict()) | {c.lower() for c in summary.findings.columns}
    for word in ("composite", "overall", "combined", "capability", "deterministic_exact_match"):
        assert not [k for k in found if word in k], word


def test_provenance_records_versions_and_inputs(tmp_path: Path) -> None:
    logs = run_logs(tmp_path)
    d = summarize_quiet(logs).to_dict()
    assert d["summary_version"] == cs.SUMMARY_VERSION
    assert d["citation_catalogue"] == {"path": "data/safety/citations_v1.0.0.json", "version": "1.0.0",
                                       "sha256": hashlib.sha256(cs.DEFAULT_CATALOGUE.read_bytes()).hexdigest()}
    tel = d["safety_telemetry"]
    assert tel["telemetry_version"] == st.TELEMETRY_VERSION
    assert {Path(i["log"]).name for i in tel["inputs"]} == {"d2.eval", "a1.eval", "a2.eval"}
    assert all(i["log_sha256"] for i in tel["inputs"])


# -- uncited finding: the negative case ----------------------------------------------
def test_uncited_graded_finding_fails_and_names_it(tmp_path: Path) -> None:
    logs = [*run_logs(tmp_path),
            make_log(tmp_path / "u.eval", [clause("u-1", flagged=[UNCITED]), clause("u-2", passed=[UNCITED])],
                     task="fake_uncited")]
    with pytest.raises(cs.UncitedFindingError) as exc:
        summarize_quiet(logs)
    message = str(exc.value)
    assert UNCITED in message
    assert "fake_uncited" in message and "u.eval" in message and "2 graded sample(s)" in message
    assert "citations_v1.0.0.json" in message
    assert [f["finding_id"] for f in exc.value.findings] == [UNCITED]
    for cited in ("V01", "a1.corrupt_frame_trusted", "a1.identity_overtrusted"):
        assert cited not in message


def test_uncited_d2_vector_and_missing_id_fail(tmp_path: Path) -> None:
    log = make_log(tmp_path / "v.eval", [d2("V99", "resisted"), d2("V02", "obeyed", vector_id="")])
    with pytest.raises(cs.UncitedFindingError) as exc:
        summarize_quiet([log])
    assert [f["finding_id"] for f in exc.value.findings] == [cs.MISSING_ID, "V99"]


def test_clause_error_is_listed_but_not_graded(tmp_path: Path) -> None:
    log = make_log(tmp_path / "e.eval", [clause("e-1", passed=["a1.corrupt_frame_trusted"],
                                                errors=[f"{UNCITED}: unknown rule 'nope'"])])
    r = row(summarize_quiet([log]), UNCITED)
    assert (r["clause_errors"], r["graded"], r["citation_status"]) == (1, 0, "uncited_pending")


def test_cli_uncited_exits_2_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = make_log(tmp_path / "u.eval", [clause("u-1", flagged=[UNCITED])], task="fake_uncited")
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", st.SafetyTelemetryWarning)
        code = cs.main([str(log), "--out", str(out)])
    assert code == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "summary not generated" in err and UNCITED in err


# -- CLI and files -----------------------------------------------------------------
def test_cli_writes_json_and_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_logs(tmp_path)
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", st.SafetyTelemetryWarning)
        assert cs.main([str(tmp_path), "--out", str(out)]) == 0
    json_path = next(out.glob("compliance_summary_*.json"))
    csv_path = next(out.glob("compliance_summary_*.csv"))
    d = json.loads(json_path.read_text(encoding="utf-8"))  # strict JSON: no NaN
    assert {f["finding_id"] for f in d["findings"]} >= {"V01", "a2.position_without_basis"}
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == cs.FINDING_COLUMNS
    assert "Safety findings and the source each is graded against" in capsys.readouterr().out


def test_cli_exit_codes_for_bad_input(tmp_path: Path) -> None:
    assert cs.main([str(tmp_path / "missing")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cs.main([str(empty)]) == 2
    log = make_log(tmp_path / "d.eval", [d2("V02", "resisted")])
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert cs.main([str(log), "--citations", str(bad), "--no-write"]) == 2


def test_cli_nothing_labeled_exits_1(tmp_path: Path) -> None:
    log = make_log(tmp_path / "s.eval", [plain("smoke-1")], task="smoke")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", st.SafetyTelemetryWarning)
        assert cs.main([str(log), "--no-write"]) == 1


def test_logs_are_not_modified(tmp_path: Path) -> None:
    logs = run_logs(tmp_path)
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in logs}
    summarize_quiet(logs)
    assert before == {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in logs}


# -- end to end ----------------------------------------------------------------------
def test_end_to_end_real_d2_run_is_fully_cited(tmp_path: Path) -> None:
    from bench.acars.d2_run import run as d2_run

    log = d2_run("mockllm/model", script="obedient", log_dir=str(tmp_path))
    summary = summarize_quiet([log.location])
    f = summary.findings
    assert set(f["finding_id"]) == {f"V{i:02d}" for i in range(1, 17)}
    assert set(f["citation_status"]) == {"taxonomy"}
    assert int(f["graded"].sum()) == 14 and int(f["pending"].sum()) == 2  # V12, V14 need the judge
