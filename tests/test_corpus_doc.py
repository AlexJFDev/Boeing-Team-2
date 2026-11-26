"""Protocol corpus documentation (WBS 3.10, issue #65).

Acceptance criteria:

1. When a reader looks up any task family, the doc lists its source spec, oracle and
   scoring tolerance.
2. (negative) When a task cites a source not in the documentation, review fails: the
   check exits 1 and names the task and the source.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bench import corpus_doc as cd

FAMILIES = ["A1", "A2", "B1", "B2", "B3", "B4"]


@pytest.fixture
def tasks(tmp_path: Path) -> Path:
    """A copy of data/tasks to break."""
    out = tmp_path / "tasks"
    shutil.copytree(cd.TASKS_DIR, out)
    return out


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    """A copy of docs/corpus.md to break."""
    out = tmp_path / "corpus.md"
    shutil.copy(cd.DOC, out)
    return out


def _codes(problems: list[cd.Problem]) -> list[str]:
    return [p.code for p in problems]


def _rewrite(path: Path, change) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        change(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_the_committed_doc_covers_every_task_file() -> None:
    assert cd.check() == []
    assert cd.main([]) == 0
    assert sorted(cd.task_files()) == FAMILIES


# ---------------------------------------------------------------------------
# Acceptance 1: look up any family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_lists_source_spec_oracle_and_tolerance(family: str, capsys: pytest.CaptureFixture[str]) -> None:
    entry = cd.parse_doc().entries[family]
    for name in ("Source spec", "Oracle", "Tolerance", "Data source", "Scorer", "Task files"):
        assert entry.fields[name].text.strip(), name

    assert cd.main(["--family", family.lower()]) == 0
    out = capsys.readouterr().out
    for name in ("Source spec", "Oracle", "Tolerance"):
        assert f"\n{name}" in out
    assert "Sources cited:" in out and "NOT IN THE SOURCES TABLE" not in out


def test_oracle_rows_name_the_oracle_and_version_the_tasks_use() -> None:
    doc = cd.parse_doc()
    assert "pyModeS==3.6.0" in doc.entries["A1"].fields["Oracle"].tokens
    assert "bench.adsb.cpr==1.1.0" in doc.entries["A2"].fields["Oracle"].tokens
    assert "bench.arinc429.conformance==1.0.0" in doc.entries["B3"].fields["Oracle"].tokens
    assert "bench.arinc429.triage==1.0.0" in doc.entries["B4"].fields["Oracle"].tokens


def test_looking_up_an_undocumented_family_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cd.main(["--family", "C1"]) == 2
    assert "documented: A1, A2, B1, B2, B3, B4" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Acceptance 2 (negative): a source missing from the doc fails review
# ---------------------------------------------------------------------------


def test_a_task_citing_a_source_missing_from_the_doc_fails(tasks: Path, capsys: pytest.CaptureFixture[str]) -> None:
    b4 = tasks / "b4_arinc429_fault_vs_injection_v1.0.0.jsonl"
    # adsb.lol is an approved source (bench/boundary.py), so only the doc check catches it.
    _rewrite(b4, lambda row: row.update(source="adsb.lol") if row["id"] == "b4-03" else None)

    problems = cd.check(tasks_dir=tasks)
    assert sorted(_codes(problems)) == ["D4", "D5"]
    assert all("b4-03" in p.message and "'adsb.lol'" in p.message for p in problems)
    assert problems[0].path.endswith(b4.name) and problems[0].line == 3

    assert cd.main(["--tasks", str(tasks), "--annotations"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: corpus documentation check, 2 problem(s)" in out
    assert "::error file=" in out and "[D4] task b4-03 cites source 'adsb.lol'" in out


def test_a_source_in_the_table_but_not_the_family_entry_fails(tasks: Path) -> None:
    _rewrite(tasks / "b1_arinc429_word_decode_v1.0.0.jsonl", lambda row: row.update(source="opensky"))
    assert _codes(cd.check(tasks_dir=tasks)) == ["D5"]


def test_an_oracle_version_the_doc_does_not_name_fails(tasks: Path) -> None:
    path = tasks / "b3_arinc429_bus_conformance_v1.0.0.jsonl"
    _rewrite(path, lambda row: row["metadata"].update(checker="bench.arinc429.conformance==1.1.0"))
    problems = cd.check(tasks_dir=tasks)
    assert _codes(problems) == ["D6"] and "bench.arinc429.conformance==1.1.0" in problems[0].message


def test_a_new_task_version_or_family_must_be_documented(tasks: Path) -> None:
    shutil.copy(tasks / "a2_cpr_position_v1.2.0.jsonl", tasks / "a2_cpr_position_v1.3.0.jsonl")
    shutil.copy(tasks / "b1_arinc429_word_decode_v1.0.0.jsonl", tasks / "c1_afdx_decode_v1.0.0.jsonl")
    problems = cd.check(tasks_dir=tasks)
    assert sorted(_codes(problems)) == ["D1", "D3"]
    assert any("a2_cpr_position_v1.3.0.jsonl is not listed" in p.message for p in problems)
    assert any("task family C1 has no" in p.message for p in problems)


def test_the_loader_demo_is_not_part_of_the_corpus() -> None:
    assert not any(p.name.startswith("loader_demo") for ps in cd.task_files().values() for p in ps)


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def test_doc_problems_are_reported(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    text = text.replace("`wikipedia-arinc429` (BNR and BCD SSM tables)", "`arinc-429-spec` (BNR and BCD SSM tables)")
    text = text.replace("`b3_arinc429_bus_conformance_v1.0.0.jsonl` (current)",
                        "`b3_arinc429_bus_conformance_v1.0.0.jsonl` (current), `b3_arinc429_bus_conformance_v0.9.0.jsonl`")
    text = text.replace("| Scorer | `b4_triage_scorer`", "| Scorer | `b4_field_scorer`")
    text = text.replace("| Tolerance | None. Both words'", "| Tolerances | None. Both words'")
    doc.write_text(text, encoding="utf-8")
    problems = cd.check(doc_path=doc)
    found = {p.code: p.message for p in problems}
    assert set(found) == {"D2", "D7", "D8", "D9"}
    assert "B2 entry has no 'Tolerance'" in found["D2"]
    assert "'arinc-429-spec', which is not in the Sources table" in found["D7"]
    assert "b3_arinc429_bus_conformance_v0.9.0.jsonl, which does not exist" in found["D8"]
    assert "b4_field_scorer" in found["D9"]


def test_a_family_with_no_entry_fails(doc: Path) -> None:
    doc.write_text(doc.read_text(encoding="utf-8").replace("## B2 · ARINC 429 SSM", "## SSM"), encoding="utf-8")
    problems = cd.check(doc_path=doc)
    assert "D1" in _codes(problems) and any("task family B2" in p.message for p in problems)


def test_missing_doc_or_task_dir_exits_2(tmp_path: Path) -> None:
    assert cd.main(["--doc", str(tmp_path / "nope.md")]) == 2
    assert cd.main(["--tasks", str(tmp_path / "nope")]) == 2
