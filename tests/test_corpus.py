"""Corpus versioning and corpus hash (WBS 1.14, issue #22).

Acceptance criteria (issue):
- When a corpus file matches its manifest hash, the task loads and the hash is
  recorded in the EvalLog.
- When a corpus file differs from its manifest hash, loading fails; frozen corpora are
  never edited in place.

From the product backlog, also covered:
- When the corpus is unchanged, its hash is identical on every machine and is written
  to every EvalLog.
- When any task file is edited, the corpus hash changes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from bench import corpus
from bench.corpus import (
    CorpusError,
    add_to_manifest,
    corpus_hash,
    manifest_entries,
    task_corpus,
    verify_file,
)
from bench.loader import load_tasks
from bench.provenance import corpus_hash_from_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
FRAMES = DATA / "adsb" / "frames_v1.0.0.jsonl"
DEMO_TASKS = DATA / "tasks" / "loader_demo_v0.1.0.jsonl"
ADSB_PREAMBLE = DATA / "preambles" / "adsb_v1.0.0.txt"
MANIFESTED_DIRS = [DATA / "adsb", DATA / "tasks", DATA / "preambles"]


@pytest.fixture
def repo_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of data/ in a temporary repo root, so files can be edited safely."""
    shutil.copytree(DATA, tmp_path / "data")
    monkeypatch.setattr(corpus, "REPO_ROOT", tmp_path)
    return tmp_path


def flip_one_byte(path: Path, offset: int = 10) -> None:
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 0x01
    path.write_bytes(bytes(raw))


# --- Acceptance criterion 1: matching files load and the hash is recorded ----


@pytest.mark.parametrize("directory", MANIFESTED_DIRS, ids=lambda d: d.name)
def test_every_committed_corpus_file_matches_its_manifest(directory: Path) -> None:
    listed = manifest_entries(directory / "MANIFEST.json")
    on_disk = {p.name for p in directory.iterdir()
               if p.is_file() and p.name not in ("MANIFEST.json", "README.md")}
    assert on_disk == set(listed), "every corpus file must be listed, and nothing else"
    for name in listed:
        verify_file(directory / name)


def test_frame_set_hash_is_the_manifest_hash() -> None:
    """The demo: frames_v1.0.0.jsonl verifies, and its hash is the one in MANIFEST.json."""
    assert verify_file(FRAMES) == json.loads((DATA / "adsb" / "MANIFEST.json").read_text())["sha256"]
    assert verify_file(FRAMES) == corpus_hash_from_manifest(DATA / "adsb" / "MANIFEST.json")


def test_task_corpus_covers_the_task_file_and_its_preambles() -> None:
    described = task_corpus(DEMO_TASKS)
    assert set(described["corpus_files"]) == {
        "data/tasks/loader_demo_v0.1.0.jsonl", "data/preambles/adsb_v1.0.0.txt"}
    assert described["corpus_hash"] == corpus_hash(described["corpus_files"])
    assert described["corpus_hash_scheme"] == corpus.CORPUS_VERSION


def test_extra_data_files_join_the_corpus() -> None:
    with_frames = task_corpus(DEMO_TASKS, data_files=[FRAMES])
    assert "data/adsb/frames_v1.0.0.jsonl" in with_frames["corpus_files"]
    assert with_frames["corpus_hash"] != task_corpus(DEMO_TASKS)["corpus_hash"]


def test_corpus_hash_is_recorded_in_the_evallog(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_ai import eval as inspect_eval

    monkeypatch.chdir(REPO_ROOT)
    [log] = inspect_eval("tasks/loader_demo.py", model="mockllm/model",
                         log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    assert log.eval.metadata is not None
    assert log.eval.metadata["corpus_hash"] == task_corpus(DEMO_TASKS)["corpus_hash"]
    assert log.eval.metadata["corpus_files"] == task_corpus(DEMO_TASKS)["corpus_files"]


# --- Acceptance criterion 2: a file that differs stops the load ---------------


def test_one_flipped_byte_fails_verification(repo_copy: Path) -> None:
    frames = repo_copy / "data" / "adsb" / "frames_v1.0.0.jsonl"
    verify_file(frames)
    flip_one_byte(frames)
    with pytest.raises(CorpusError, match="does not match .*never edited in place"):
        verify_file(frames)


def test_edited_task_file_fails_to_load(repo_copy: Path) -> None:
    tasks = repo_copy / "data" / "tasks" / "loader_demo_v0.1.0.jsonl"
    load_tasks(tasks, repo_copy / "data" / "preambles")
    tasks.write_text(tasks.read_text().replace("ZZZ101", "ZZZ102"))  # still valid JSON
    with pytest.raises(CorpusError, match="loader_demo_v0.1.0.jsonl does not match"):
        load_tasks(tasks, repo_copy / "data" / "preambles")


def test_edited_preamble_fails_to_load(repo_copy: Path) -> None:
    preambles = repo_copy / "data" / "preambles"
    flip_one_byte(preambles / "adsb_v1.0.0.txt", offset=40)
    with pytest.raises(CorpusError, match="adsb_v1.0.0.txt does not match"):
        load_tasks(repo_copy / "data" / "tasks" / "loader_demo_v0.1.0.jsonl", preambles)


def test_unlisted_task_file_fails_to_load(repo_copy: Path) -> None:
    tasks_dir = repo_copy / "data" / "tasks"
    copy = tasks_dir / "loader_demo_v0.2.0.jsonl"
    shutil.copy(tasks_dir / "loader_demo_v0.1.0.jsonl", copy)
    with pytest.raises(CorpusError, match="not listed in tasks/MANIFEST.json"):
        load_tasks(copy, repo_copy / "data" / "preambles")


def test_cli_prints_the_hash_and_fails_on_a_flipped_byte(
        repo_copy: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frames = repo_copy / "data" / "adsb" / "frames_v1.0.0.jsonl"
    assert corpus.main([str(frames)]) == 0
    assert verify_file(frames) in capsys.readouterr().out
    flip_one_byte(frames)
    assert corpus.main([str(frames)]) == 2
    assert "does not match" in capsys.readouterr().err


# --- Frozen corpora are never edited in place ---------------------------------


def test_manifest_entry_cannot_be_overwritten(repo_copy: Path) -> None:
    """Re-hashing an edited file is refused, so an in-place edit cannot be made to pass."""
    tasks = repo_copy / "data" / "tasks" / "loader_demo_v0.1.0.jsonl"
    before = (repo_copy / "data" / "tasks" / "MANIFEST.json").read_text()
    flip_one_byte(tasks)
    with pytest.raises(CorpusError, match="already listed with a different sha256"):
        add_to_manifest(tasks)
    assert (repo_copy / "data" / "tasks" / "MANIFEST.json").read_text() == before


def test_a_change_is_a_new_version(repo_copy: Path) -> None:
    tasks_dir = repo_copy / "data" / "tasks"
    new = tasks_dir / "loader_demo_v0.2.0.jsonl"
    new.write_text((tasks_dir / "loader_demo_v0.1.0.jsonl").read_text().replace("ZZZ101", "ZZZ102"))
    sha = add_to_manifest(new)
    assert verify_file(new) == sha
    verify_file(tasks_dir / "loader_demo_v0.1.0.jsonl")  # the old version is untouched
    assert add_to_manifest(new) == sha  # adding the same bytes again is harmless


def test_unversioned_file_name_is_refused(repo_copy: Path) -> None:
    unversioned = repo_copy / "data" / "tasks" / "new_tasks.jsonl"
    unversioned.write_text("{}\n")
    with pytest.raises(CorpusError, match="no version in its name"):
        add_to_manifest(unversioned)


def test_single_file_manifest_is_left_to_its_generator(repo_copy: Path) -> None:
    extra = repo_copy / "data" / "adsb" / "frames_v1.1.0.jsonl"
    extra.write_text("{}\n")
    with pytest.raises(CorpusError, match="single-file manifest"):
        add_to_manifest(extra)


# --- Backlog: same hash on every machine; any edit changes it ----------------


def test_corpus_hash_matches_sha256sum_format() -> None:
    """Reproducible without our code: `sha256sum <files> | sha256sum` from the repo root."""
    files = task_corpus(DEMO_TASKS)["corpus_files"]
    listing = "".join(
        f"{hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()}  {p}\n" for p in sorted(files))
    assert task_corpus(DEMO_TASKS)["corpus_hash"] == hashlib.sha256(listing.encode()).hexdigest()


def test_corpus_hash_does_not_depend_on_where_the_repo_is(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    here = task_corpus(DEMO_TASKS)["corpus_hash"]
    for name in ("checkout-a", "somewhere/else/checkout-b"):
        root = tmp_path / name
        shutil.copytree(DATA, root / "data")
        monkeypatch.setattr(corpus, "REPO_ROOT", root)
        moved = task_corpus(root / "data" / "tasks" / "loader_demo_v0.1.0.jsonl",
                            preamble_dir=root / "data" / "preambles")
        assert moved["corpus_hash"] == here


def test_line_endings_cannot_vary_between_machines() -> None:
    """git must not convert line endings under data/, or Windows checkouts would hash differently."""
    rules = (REPO_ROOT / ".gitattributes").read_text().split("\n")
    assert "data/** -text" in rules


def test_corpus_hash_ignores_listing_order_but_not_content() -> None:
    files = {"data/a_v1.0.0.txt": "1" * 64, "data/b_v1.0.0.txt": "2" * 64}
    assert corpus_hash(files) == corpus_hash(dict(reversed(list(files.items()))))
    assert corpus_hash(files) != corpus_hash({**files, "data/a_v1.0.0.txt": "3" * 64})
    assert corpus_hash(files) != corpus_hash({**files, "data/c_v1.0.0.txt": "1" * 64})


def test_any_task_file_change_changes_the_corpus_hash(repo_copy: Path) -> None:
    """An edit saved as a new version gives a new corpus hash."""
    tasks_dir = repo_copy / "data" / "tasks"
    preambles = repo_copy / "data" / "preambles"
    old = task_corpus(tasks_dir / "loader_demo_v0.1.0.jsonl", preamble_dir=preambles)
    new = tasks_dir / "loader_demo_v0.2.0.jsonl"
    new.write_text((tasks_dir / "loader_demo_v0.1.0.jsonl").read_text().replace(" ", "  ", 1))
    add_to_manifest(new)
    assert task_corpus(new, preamble_dir=preambles)["corpus_hash"] != old["corpus_hash"]


def test_corpus_hash_fits_the_provenance_schema() -> None:
    """7.1's ProvenanceRecord requires corpus_hash to be 64 lowercase hex characters."""
    from bench.provenance import ProvenanceRecord

    record = ProvenanceRecord.model_validate({
        "task_id": "t", "corpus_hash": task_corpus(DEMO_TASKS)["corpus_hash"],
        "model_hash": "0" * 64, "prompt_template_hash": "0" * 64, "raw_output": "",
        "parsed_answer": "", "token_usage": {"input_tokens": 0, "output_tokens": 0,
                                             "total_tokens": 0},
    })
    assert record.corpus_hash == task_corpus(DEMO_TASKS)["corpus_hash"]
