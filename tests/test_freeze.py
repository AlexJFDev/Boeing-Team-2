"""Freeze protocol corpus v1.0 (WBS 3.11, issue #66).

Acceptance criteria:

1. When v1.0 is frozen, its corpus hash is recorded and tagged in Git.
2. (negative) When a task still has an open review issue, the freeze is blocked.

The inter-rater pilot (3.9) has not been graded yet, so these tests give the freeze a
stand-in passing pilot record in a temporary folder; the real repository's freeze is
blocked on that check until two team members grade it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from bench import corpus, freeze, irr

REVIEW = freeze.REVIEW_LABEL


def passing_irr(root: Path) -> Path:
    """A stand-in pilot record that passed on the current rubric."""
    pilot = root / "irr" / "pilot-standin"
    pilot.mkdir(parents=True)
    (pilot / irr.RECORD).write_text(json.dumps({
        "rubric_fingerprint": irr.rubric_fingerprint(), "status": "pass", "threshold": irr.ALPHA_THRESHOLD,
        "alpha": 0.9, "raters": ["standin-a", "standin-b"], "scored_at": "2026-11-27T00:00:00+00:00"}))
    return root / "irr"


def issue(number: int, title: str, body: str = "") -> dict[str, Any]:
    return {"number": number, "title": title, "body": body}


@pytest.fixture
def irr_dir(tmp_path: Path) -> Path:
    return passing_irr(tmp_path)


@pytest.fixture
def frozen(tmp_path: Path, irr_dir: Path) -> tuple[Path, dict[str, Any]]:
    out = tmp_path / "frozen"
    checks, record = freeze.freeze(out_dir=out, issues=[], issues_source="test", irr_dir=irr_dir)
    assert record is not None, [c for c in checks if not c.ok]
    return out, record


# ---------------------------------------------------------------------------
# Acceptance 1: frozen, hash recorded, tagged
# ---------------------------------------------------------------------------


def test_freeze_records_the_corpus_hash(frozen: tuple[Path, dict[str, Any]]) -> None:
    out, record = frozen
    saved = json.loads(freeze.record_path(out).read_text(encoding="utf-8"))
    assert saved == record
    assert record["corpus"] == "protocol" and record["release"] == "1.0.0"
    assert record["corpus_hash"] == corpus.corpus_hash(record["files"])
    assert set(record["families"]) == {"A1", "A2", "B1", "B2", "B3", "B4"}
    assert record["tasks"] == 129
    assert set(record["files"]) == {
        *(e["task_file"] for e in record["families"].values()),
        "data/preambles/adsb_v1.1.0.txt", "data/preambles/arinc429_v1.0.0.txt"}
    assert record["tag"] == "corpus-protocol-v1.0.0"
    assert all(c["ok"] for c in record["checks"])
    assert [c["name"] for c in record["checks"]] == ["manifests", "schema", "boundary", "docs", "irr", "reviews"]
    assert record["documentation"] == {"docs/corpus.md": corpus.sha256_file(freeze.DOC)}


def test_each_family_hash_is_the_one_its_evallogs_record(frozen: tuple[Path, dict[str, Any]]) -> None:
    _, record = frozen
    for fam in freeze.protocol_families():
        entry = record["families"][fam.id]
        assert entry["run_corpus_hash"] == corpus.task_corpus(fam.task_file)["corpus_hash"]
        assert entry["task_version"] == fam.task_version
    assert record["families"]["A2"]["capability_scorer"] == "a2_cpr_coordinate_scorer==1.0.0"
    assert record["families"]["B1"]["capability_scorer"] == "arinc429_bnr_bcd_capability_scorer==1.1.0"
    assert record["families"]["A1"]["oracles"] == ["bench.adsb.oracle==1.0.0", "pyModeS==3.6.0"]
    assert record["families"]["A1"]["sources"] == ["opensky", "spec-synthesized"]


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, frozen: tuple[Path, dict[str, Any]]) -> Path:
    """A throwaway git repository holding the committed record."""
    out, _ = frozen
    root = tmp_path / "repo"
    (root / "data" / "frozen").mkdir(parents=True)
    shutil.copy(freeze.record_path(out), root / "data" / "frozen")
    git(root, "init", "-q")
    git(root, "config", "user.name", "t")
    git(root, "config", "user.email", "t@example.com")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "freeze protocol v1.0.0")
    return root


def test_tag_marks_the_commit_with_the_corpus_hash(repo: Path, frozen: tuple[Path, dict[str, Any]]) -> None:
    _, record = frozen
    name = freeze.tag(repo / "data" / "frozen", git_root=repo)
    assert name == "corpus-protocol-v1.0.0"
    assert git(repo, "rev-list", "-n1", name) == git(repo, "rev-parse", "HEAD")
    assert git(repo, "cat-file", "-t", name) == "tag"  # annotated
    message = git(repo, "tag", "-l", "--format=%(contents)", name)
    assert f"corpus_hash: {record['corpus_hash']}" in message and "data/frozen/protocol_v1.0.0.json" in message

    with pytest.raises(freeze.FreezeError, match="never moved"):
        freeze.tag(repo / "data" / "frozen", git_root=repo)


def test_tag_refuses_an_uncommitted_or_edited_record(repo: Path) -> None:
    path = repo / "data" / "frozen" / "protocol_v1.0.0.json"
    record = json.loads(path.read_text())
    record["tasks"] = 130
    path.write_text(json.dumps(record))
    with pytest.raises(freeze.FreezeError, match="differs from the committed copy"):
        freeze.tag(repo / "data" / "frozen", git_root=repo)
    git(repo, "rm", "-q", "--cached", "data/frozen/protocol_v1.0.0.json")
    git(repo, "commit", "-qm", "untrack")
    with pytest.raises(freeze.FreezeError, match="not committed at HEAD"):
        freeze.tag(repo / "data" / "frozen", git_root=repo)


def test_a_frozen_record_is_never_overwritten(frozen: tuple[Path, dict[str, Any]], irr_dir: Path) -> None:
    out, record = frozen
    with pytest.raises(freeze.FreezeError, match="never rewritten"):
        freeze.freeze(out_dir=out, issues=[], irr_dir=irr_dir)
    assert json.loads(freeze.record_path(out).read_text()) == record
    assert freeze.main(["freeze", "--out-dir", str(out), "--irr-dir", str(irr_dir)]) == 2


def test_verify_catches_a_frozen_file_changed_after_the_freeze(frozen: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    out, record = frozen
    assert freeze.verify(out) == []
    files_root = tmp_path / "copy"
    for rel in record["files"]:
        (files_root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(freeze.REPO_ROOT / rel, files_root / rel)
    assert freeze.verify(out, files_root) == []
    edited = files_root / record["families"]["B3"]["task_file"]
    edited.write_bytes(edited.read_bytes().replace(b"b3-07", b"b3-7x", 1))
    (files_root / "data/preambles/adsb_v1.1.0.txt").unlink()
    problems = freeze.verify(out, files_root)
    assert any("b3_arinc429_bus_conformance_v1.0.0.jsonl has changed since the freeze" in p for p in problems)
    assert any("adsb_v1.1.0.txt is missing" in p for p in problems)
    assert freeze.main(["verify", "--out-dir", str(out)]) == 0


# ---------------------------------------------------------------------------
# Acceptance 2 (negative): an open review issue blocks the freeze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("found, names", [
    (issue(201, "Review b3-07: is the gap rule right?"), ["b3-07"]),
    (issue(202, "Ground truth", "Check b1-a429-0003 and B1-A429-0004"), ["b1-a429-0003", "b1-a429-0004"]),
    (issue(203, "a2_cpr_position_v1.2.0.jsonl tolerance"), ["a2_cpr_position_v1.2.0.jsonl"]),
    (issue(204, "A2 targets look off by a frame"), ["A2"]),
    (issue(205, "Something is wrong with a task"), ["(names no task)"]),
])
def test_an_open_review_issue_on_a_task_blocks_the_freeze(found: dict[str, Any], names: list[str],
                                                         tmp_path: Path, irr_dir: Path) -> None:
    out = tmp_path / "frozen"
    checks, record = freeze.freeze(out_dir=out, issues=[found], issues_source="test", irr_dir=irr_dir)
    assert record is None and not freeze.record_path(out).exists()
    reviews = next(c for c in checks if c.name == "reviews")
    assert not reviews.ok and f"#{found['number']}" in reviews.detail
    assert all(n in reviews.detail for n in names)
    assert all(c.ok for c in checks if c.name != "reviews")


def test_review_issues_on_other_corpora_do_not_block(tmp_path: Path, irr_dir: Path) -> None:
    others = [issue(301, "Review d1-03 ACARS label"), issue(302, "C1 virtual link IDs")]
    checks, record = freeze.freeze(out_dir=tmp_path / "f", issues=others, issues_source="test", irr_dir=irr_dir)
    assert record is not None
    assert "2 open review issue(s) checked" in next(c for c in checks if c.name == "reviews").detail


def test_unreadable_review_issues_block_the_freeze(tmp_path: Path, irr_dir: Path) -> None:
    def offline(repo: str) -> list[dict[str, Any]]:
        raise urllib.error.URLError("no network")
    checks, record = freeze.freeze(out_dir=tmp_path / "f", irr_dir=irr_dir, fetch=offline)
    assert record is None
    assert "could not read" in next(c for c in checks if c.name == "reviews").detail


def test_the_cli_reports_blocked_and_ready(tmp_path: Path, irr_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([issue(7, "Review b4-11")]))
    assert freeze.main(["check", "--review-issues", str(issues), "--irr-dir", str(irr_dir)]) == 1
    assert "BLOCKED" in capsys.readouterr().out
    issues.write_text("[]")
    assert freeze.main(["check", "--review-issues", str(issues), "--irr-dir", str(irr_dir)]) == 0
    assert "READY to freeze" in capsys.readouterr().out


def test_no_passing_inter_rater_pilot_blocks_the_freeze(tmp_path: Path) -> None:
    empty = tmp_path / "irr"
    empty.mkdir()
    checks, record = freeze.freeze(out_dir=tmp_path / "f", issues=[], issues_source="test", irr_dir=empty)
    assert record is None and not next(c for c in checks if c.name == "irr").ok


def test_pull_requests_are_not_review_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, body: Any) -> None:
            self.body = json.dumps(body).encode()

        def read(self) -> bytes:
            return self.body

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    seen: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> Response:
        seen.append(request.full_url)
        return Response([{"number": 1, "title": "Review b3-07", "body": None, "html_url": "u"},
                         {"number": 2, "title": "PR touching b3-07", "pull_request": {}}])

    monkeypatch.setattr(freeze.urllib.request, "urlopen", fake_urlopen)
    issues = freeze.fetch_review_issues("owner/repo")
    assert [i["number"] for i in issues] == [1]
    assert "labels=task-review" in seen[0] and "state=open" in seen[0]
