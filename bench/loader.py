"""Task loader with operational preamble injection (WBS 1.13, issue #21).

Reads a JSON Lines task file with Inspect's ``json_dataset`` and turns each record
into a ``Sample`` whose input is the family preamble followed by the task text:

    <family preamble>

    Premises for this task:          (only when the task has its own premises)
    - <premise>

    <task text>

The preamble goes in the user message, not the system message. It states the
premises that make the answer determinate, so it is part of the task and must be
the same under every configuration we compare. The system message belongs to the
configuration (prompt templates, guardrail profiles) and could otherwise drop it.
See data/preambles/README.md.

Task record fields (all required):

    id        unique task ID
    family    A (ADS-B / Mode S) | B (ARINC 429) | C (ARINC 664 / AFDX) | D (ACARS / VDL2)
    preamble  preamble name, resolved to data/preambles/<name>.txt
    source    where the data came from (checked against the approved list in WBS 1.27)
    input     task text shown to the model after the preamble
    target    ground-truth answer

Optional fields:

    premises  list of strings: premises that hold for this task only, such as the
              resync rule (catalogue C1) or a label's transmission rate (B3)
    metadata  object carried into ``Sample.metadata``

Loading fails with ``TaskLoadError`` naming the task (or its line number when it
has no usable ID) if a record is malformed or names a preamble that does not exist.
After the records pass, the task file and every preamble it uses are checked against
their manifests (WBS 1.14, bench/corpus.py); a file that does not match raises
``CorpusError``.

Command-line demo, prints the final prompt of every task:

    python -m bench.loader data/tasks/loader_demo_v0.1.0.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, Sample, json_dataset

from bench.corpus import CorpusError, task_corpus

LOADER_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
PREAMBLE_DIR = REPO_ROOT / "data" / "preambles"

FAMILIES = frozenset({"A", "B", "C", "D"})
REQUIRED_FIELDS = ("id", "family", "preamble", "source", "input", "target")
PREAMBLE_SEPARATOR = "\n\n"


class TaskLoadError(ValueError):
    """A task record is malformed or refers to a preamble that does not exist."""


def load_preamble(name: str, preamble_dir: Path = PREAMBLE_DIR) -> str:
    """The text of preamble ``name``. Raises FileNotFoundError if there is none."""
    if not name or Path(name).name != name:
        raise FileNotFoundError(f"invalid preamble name {name!r}")
    path = preamble_dir / f"{name}.txt"
    return path.read_bytes().decode("utf-8").rstrip("\n")


def preamble_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TASK_PREMISES_HEADING = "Premises for this task:"


def build_prompt(preamble: str, task_text: str, premises: list[str] | None = None) -> str:
    parts = [preamble]
    if premises:
        parts.append("\n".join([TASK_PREMISES_HEADING, *(f"- {p}" for p in premises)]))
    parts.append(task_text)
    return PREAMBLE_SEPARATOR.join(parts)


def _problems(record: dict[str, Any]) -> list[str]:
    problems = []
    for name in REQUIRED_FIELDS:
        value = record.get(name)
        if name not in record or value is None:
            problems.append(f"missing field '{name}'")
        elif name != "target" and not (isinstance(value, str) and value.strip()):
            problems.append(f"field '{name}' must be a non-empty string")
    target = record.get("target")
    if target is not None and not (
        (isinstance(target, str) and target.strip())
        or (isinstance(target, list) and target and all(isinstance(t, str) for t in target))
    ):
        problems.append("field 'target' must be a non-empty string or list of strings")
    family = record.get("family")
    if isinstance(family, str) and family.strip() and family not in FAMILIES:
        problems.append(f"family {family!r} is not one of {sorted(FAMILIES)}")
    premises = record.get("premises")
    if premises is not None and not (
        isinstance(premises, list) and premises
        and all(isinstance(p, str) and p.strip() for p in premises)
    ):
        problems.append("field 'premises' must be a non-empty list of non-empty strings")
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        problems.append("field 'metadata' must be an object")
    return problems


def make_record_to_sample(preamble_dir: Path = PREAMBLE_DIR):
    """A ``record_to_sample`` function for ``json_dataset`` that validates and injects."""
    seen: set[str] = set()
    cache: dict[str, str] = {}
    line = 0

    def record_to_sample(record: dict[str, Any]) -> Sample:
        nonlocal line
        line += 1
        task_id = record.get("id")
        where = f"task {task_id!r}" if isinstance(task_id, str) and task_id else f"record {line}"

        problems = _problems(record)
        if problems:
            raise TaskLoadError(f"{where}: " + "; ".join(problems))
        if task_id in seen:
            raise TaskLoadError(f"{where}: duplicate task id")
        seen.add(task_id)

        name = record["preamble"]
        if name not in cache:
            try:
                cache[name] = load_preamble(name, preamble_dir)
            except FileNotFoundError:
                raise TaskLoadError(
                    f"{where}: preamble {name!r} does not exist in {preamble_dir}"
                ) from None
        preamble = cache[name]

        return Sample(
            id=task_id,
            input=build_prompt(preamble, record["input"], record.get("premises")),
            target=record["target"],
            metadata={
                **record.get("metadata", {}),
                "family": record["family"],
                "source": record["source"],
                "preamble": name,
                "preamble_sha256": preamble_sha256(preamble),
                "premises": record.get("premises", []),
                "task_text": record["input"],
                "loader_version": LOADER_VERSION,
            },
        )

    return record_to_sample


def load_tasks(path: str | Path, preamble_dir: Path = PREAMBLE_DIR, *,
               verify: bool = True) -> Dataset:
    """Load a task file.

    Raises TaskLoadError on the first bad record, then CorpusError if the task file
    or a preamble it uses does not match its manifest. ``verify=False`` skips the
    manifest check; it is for tests that build throwaway task files.
    """
    dataset = json_dataset(
        str(path),
        sample_fields=make_record_to_sample(preamble_dir),
        shuffle=False,
    )
    if verify:
        task_corpus(path, preamble_dir=preamble_dir)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Print each task's final prompt.")
    parser.add_argument("tasks", help="JSON Lines task file")
    args = parser.parse_args()
    try:
        dataset = load_tasks(args.tasks)
    except (TaskLoadError, CorpusError) as exc:
        raise SystemExit(f"error: {exc}") from None
    for sample in dataset:
        print(f"===== {sample.id} (preamble {sample.metadata['preamble']})")
        print(sample.input)
        print(f"----- target: {sample.target}\n")


if __name__ == "__main__":
    main()
