"""Synthetic data boundary checker (WBS 1.27, issue #35).

Acceptance criteria (product backlog):
- When a task's source field lists an approved source (spec-synthesized, OpenSky,
  airplanes.live, adsb.lol, Airframes.io), it passes.
- When a task has no source or an unapproved source, the checker fails it.

Sprint 3 backlog demo, also covered:
- The checker passes a clean synthetic corpus, then fails when a file with a
  real-looking registration or key is added, naming that file.

Fixtures must not be real identifiers either (CLAUDE.md rule 1). The U.S. marks used
as positives (N98QZ, N987QZ, N979QX, N6429X) were each shown as "Not Assigned/Reserved"
in the FAA registry N-number inquiry when this was written. There is no 5-digit or
2-symbol positive: every one tried was assigned, reserved or deregistered. Those
boundaries are covered by the negatives (N123456, N1). The hyphenated positives use
Q-first suffixes (G-QXZQ) and were not checked against each national register.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from bench import boundary as bd
from bench.boundary import (
    APPROVED_SOURCES,
    REPO_ROOT,
    Report,
    check_paths,
    check_record,
    main,
    scan_line,
)
from bench.loader import REQUIRED_FIELDS


def task(**overrides: Any) -> dict[str, Any]:
    record = {"id": "t-001", "family": "A", "preamble": "adsb_v1.0.0", "source": "spec-synthesized",
              "input": "Decode 8DF00A112369A6B1C31820ABE595.", "target": "{}"}
    record.update(overrides)
    return {k: v for k, v in record.items() if v is not ...}


def codes(text: str) -> list[str]:
    return [code for code, _ in scan_line(text)]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


# ---------------------------------------------------------------------------
# Acceptance criterion 1: approved sources pass
# ---------------------------------------------------------------------------


def test_approved_sources_are_the_backlog_list() -> None:
    assert APPROVED_SOURCES == ("spec-synthesized", "opensky", "airplanes.live", "adsb.lol", "airframes.io")


@pytest.mark.parametrize("source", APPROVED_SOURCES)
def test_task_with_an_approved_source_passes(source: str) -> None:
    assert check_record(task(source=source), "t.jsonl", 1) == ([], True)


def test_approved_external_sources_in_a_source_object_pass() -> None:
    doc = {"source": {"kind": "sanitized", "external_sources_used": ["opensky", "adsb.lol"]}}
    assert check_record(doc, "m.json", None) == ([], False)


def test_the_repository_corpus_passes() -> None:
    """The committed data/ and prompts/ must stay inside the boundary."""
    report = check_paths()
    assert report.passed, "\n".join(map(str, report.violations))
    assert report.task_records >= 60   # A1 + A2 + loader demo at the time of writing


# ---------------------------------------------------------------------------
# Acceptance criterion 2: no source or an unapproved source fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", [..., None, ""])
def test_task_with_no_source_fails(source: Any) -> None:
    found, is_task = check_record(task(source=source), "t.jsonl", 3)
    assert is_task
    assert [(v.code, v.line) for v in found] == [("S1", 3)]
    assert "task 't-001': no source" in found[0].message


@pytest.mark.parametrize("source", [
    "OpenSky",               # tags are exact; case is not ignored
    "opensky-network",
    "flightradar24",
    "real capture",
    "airframes",
    ["spec-synthesized"],
    42,
])
def test_task_with_an_unapproved_source_fails(source: Any) -> None:
    found, _ = check_record(task(source=source), "t.jsonl", 1)
    assert [v.code for v in found] == ["S2"]
    assert "unapproved source" in found[0].message and "approved: spec-synthesized" in found[0].message


def test_unapproved_external_source_in_a_source_object_fails() -> None:
    doc = {"source": {"kind": "sanitized", "external_sources_used": ["opensky", "flightaware"]}}
    found, _ = check_record(doc, "m.json", None)
    assert [v.code for v in found] == ["S2"] and "flightaware" in found[0].message


def test_non_task_records_need_no_source_but_a_declared_one_must_be_approved() -> None:
    frame = {"frame_id": "adsb-0001", "hex": "8DF00A112369A6B1C31820ABE595"}
    assert check_record(frame, "f.jsonl", 1) == ([], False)
    assert [v.code for v in check_record({**frame, "source": "opensky"}, "f.jsonl", 1)[0]] == []
    assert [v.code for v in check_record({**frame, "source": "scraped"}, "f.jsonl", 1)[0]] == ["S2"]


def test_task_records_are_found_by_their_keys_wherever_they_live(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "acars" / "extra_tasks.jsonl", [task(), task(id="t-002", source=...)])
    report = check_paths([tmp_path])
    assert report.task_records == 2
    assert [(v.path.endswith("acars/extra_tasks.jsonl"), v.line, v.code) for v in report.violations] == [
        (True, 2, "S1")]


def test_task_keys_match_the_loader() -> None:
    assert bd.TASK_KEYS <= set(REQUIRED_FIELDS) and "source" in REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# Content: real-looking registrations (R1, R2) and keys (K1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "N98QZ", "N987QZ", "N979QX", "N6429X",
    "tail N987QZ, gate 4", "(N979QX)", "REG N6429X.",
])
def test_us_registrations_are_flagged(text: str) -> None:
    assert codes(text) == ["R1"]


@pytest.mark.parametrize("text", [
    "N1 at 92 percent", "N2", "N3",          # engine spool speeds; one-symbol marks skipped
    "N123456",                               # six symbols: not a valid N-number
    "N1234AB",                               # four digits and two letters: six symbols
    "N0123", "N01Z",                         # cannot start with zero
    "N12I", "N12O", "N1OA",                  # no I or O
    "n987qz", "SN987QZ", "N987QZ9",          # lowercase, or part of a longer token
    "8DF00A112369A6B1C31820ABE595", "F00A11", "ZZZ101", "SYN1053", "SYN-A008",
])
def test_non_registrations_are_not_flagged_as_us(text: str) -> None:
    assert "R1" not in codes(text)


@pytest.mark.parametrize("text", [
    "G-QXZQ", "VH-QXZ", "D-QXZQ", "EI-QXZ", "C-QXZQ", "9V-QXZ", "A6-QXZ", "reg: JA-QXZ.",
])
def test_hyphenated_registrations_are_flagged(text: str) -> None:
    assert codes(text) == ["R2"]


@pytest.mark.parametrize("text", [
    "DO-260B", "ED-102A", "ARINC-429",       # standards: digits after the hyphen
    "SYN-A008", "SYN-ABCD",                  # reserved synthetic format
    "ZZ06-ZZ09", "X-RAY", "E-MAIL", "A-1",
])
def test_non_registrations_are_not_flagged_as_hyphenated(text: str) -> None:
    assert "R2" not in codes(text)


FAKE = "Q7wErTy9UiOpAsDfGhJkLzXcVbNm1234567890ab"  # not a real credential


def PEM(kind: str) -> str:
    """A private-key header, assembled so the source never contains one (gitleaks scans it)."""
    return "-----BEGIN " + kind + "PRIVATE" + " KEY-----"



@pytest.mark.parametrize("text", [
    PEM(""),
    PEM("RSA "),
    PEM("OPENSSH "),
    "AKIA" + "Q7WERTY9UIOPASDF",
    "token ghp_" + FAKE[:36],
    "sk-proj-" + FAKE,
    "sk-ant-api03-" + FAKE,
    "AIza" + FAKE[:35],
    "xoxb-" + FAKE[:20],
    "hf_" + FAKE[:34],
    "api_key = '" + FAKE[:20] + "'",
    '"password": "' + FAKE[:16] + '"',
])
def test_key_like_strings_are_flagged(text: str) -> None:
    assert "K1" in codes(text)


@pytest.mark.parametrize("text", [
    "AKIA" + "SYNTHETIC0000000",                  # planted fake key, marked
    "sk-synthetic-" + FAKE,
    "sha256 3f9a1c0e7b2d4f6a8c0e2b4d6f8a0c2e4b6d8f0a2c4e6b8d0f2a4c6e8b0d2f4a",
    "password: hunter2",                          # too short to look like a secret
    "the api key is stored in .env",
    "-----BEGIN PUBLIC KEY-----",
])
def test_non_secrets_and_marked_synthetic_keys_pass(text: str) -> None:
    assert "K1" not in codes(text)


def test_reported_keys_are_redacted() -> None:
    [(code, message)] = list(scan_line("sk-proj-" + FAKE))
    assert code == "K1" and FAKE not in message and "chars" in message


# ---------------------------------------------------------------------------
# Sprint demo: clean corpus passes, an added file fails and is named
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A copy of the committed corpus to add files to."""
    root = tmp_path / "data"
    shutil.copytree(REPO_ROOT / "data", root, ignore=shutil.ignore_patterns("afdx"))
    return root


def test_clean_copy_passes(corpus: Path) -> None:
    report = check_paths([corpus])
    assert report.passed and report.summary().startswith("PASS")


@pytest.mark.parametrize("name, content, code", [
    ("tasks/leak_registration.jsonl",
     json.dumps(task(input="Decode the frame from N987QZ.")) + "\n", "R1"),
    ("adsb/notes.txt", "Captured from G-QXZQ at the gate.\n", "R2"),
    ("acars/keys.json", json.dumps({"config": {"api_key": "Z" * 4 + FAKE[:24]}}), "K1"),
    ("tasks/unsourced.jsonl", json.dumps(task(source=...)) + "\n", "S1"),
    ("tasks/scraped.jsonl", json.dumps(task(source="flightradar24")) + "\n", "S2"),
])
def test_added_file_fails_and_is_named(corpus: Path, capsys: pytest.CaptureFixture[str],
                                       name: str, content: str, code: str) -> None:
    (corpus / name).write_text(content)
    report = check_paths([corpus])
    assert not report.passed
    assert [Path(p).as_posix().endswith(name) for p in report.failed_files()] == [True]
    assert [v.code for v in report.violations] == [code]

    assert main([str(corpus)]) == 1
    out = capsys.readouterr().out
    assert name in out and f"[{code}]" in out and out.strip().splitlines()[-1].startswith("FAIL: 1 problem")


def test_violations_report_the_line(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("line one\nline two has N979QX\n")
    [v] = check_paths([tmp_path]).violations
    assert (v.line, v.code) == (2, "R1") and str(v).endswith(":2: [R1] real-looking U.S. registration 'N979QX'")


# ---------------------------------------------------------------------------
# Files the checker cannot read as text, and the command line
# ---------------------------------------------------------------------------


def test_binary_files_are_skipped_and_counted(tmp_path: Path) -> None:
    (tmp_path / "trace.pcap").write_bytes(b"\xd4\xc3\xb2\xa1" + b"\xff" * 40)
    report = check_paths([tmp_path])
    assert report.passed and report.files_scanned == 0
    assert [Path(p).name for p in report.binary_skipped] == ["trace.pcap"]
    assert "1 binary files skipped" in report.summary()


def test_hidden_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x.txt").write_text("N987QZ")
    assert check_paths([tmp_path]).files_scanned == 0


def test_malformed_json_fails(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text('{"id": 1}\n{not json\n')
    (tmp_path / "bad.json").write_text("{")
    assert sorted((Path(v.path).name, v.line, v.code) for v in check_paths([tmp_path]).violations) == [
        ("bad.json", None, "J1"), ("bad.jsonl", 2, "J1")]


def test_command_line_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("PASS: data boundary check")
    assert main([str(tmp_path / "missing")]) == 2
    assert "no such file or directory" in capsys.readouterr().err


def test_report_summary_counts() -> None:
    report = Report(files_scanned=3, task_records=2)
    assert report.summary() == "PASS: data boundary check (3 files, 2 task records)"
