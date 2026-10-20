"""Task loader tests (WBS 1.13, issue #21).

Acceptance criteria:
- When a task is loaded, the final prompt contains the preamble followed by the
  task text.
- When a sample is malformed, loading fails with an error identifying the sample.

From the product backlog, also covered: a task that names a preamble that does
not exist fails the load and reports the task ID.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate, system_message

from bench.loader import (
    PREAMBLE_DIR,
    REPO_ROOT,
    REQUIRED_FIELDS,
    TASK_PREMISES_HEADING,
    TaskLoadError,
    build_prompt,
    load_preamble,
    load_tasks,
    preamble_sha256,
)

DEMO_FILE = REPO_ROOT / "data" / "tasks" / "loader_demo_v0.1.0.jsonl"
DEMO_RECORDS = [json.loads(line) for line in DEMO_FILE.read_text().splitlines()]

GOOD: dict[str, Any] = {
    "id": "t-001",
    "family": "A",
    "preamble": "adsb_v1.0.0",
    "source": "spec-synthesized",
    "input": "Decode this message: 8DF00A112369A6B1C31820ABE595.",
    "target": "ZZZ101",
}


def write_tasks(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def with_changes(**changes: Any) -> dict[str, Any]:
    record = dict(GOOD)
    for key, value in changes.items():
        if value is ...:
            record.pop(key)
        else:
            record[key] = value
    return record


# --- Acceptance criterion 1: preamble followed by task text ------------------


def test_demo_file_loads() -> None:
    dataset = load_tasks(DEMO_FILE)
    assert [s.id for s in dataset] == [r["id"] for r in DEMO_RECORDS]


@pytest.mark.parametrize("record", DEMO_RECORDS, ids=[r["id"] for r in DEMO_RECORDS])
def test_prompt_is_preamble_then_task_text(record: dict[str, Any]) -> None:
    sample = next(s for s in load_tasks(DEMO_FILE) if s.id == record["id"])
    preamble = load_preamble(record["preamble"])
    assert isinstance(sample.input, str)
    assert sample.input == build_prompt(preamble, record["input"], record.get("premises"))
    assert sample.input.startswith(preamble)
    assert sample.input.endswith(record["input"])
    assert sample.target == record["target"]


def test_metadata_records_preamble_and_source() -> None:
    sample = load_tasks(DEMO_FILE)[0]
    assert sample.metadata is not None
    assert sample.metadata["preamble"] == "adsb_v1.0.0"
    assert sample.metadata["preamble_sha256"] == preamble_sha256(load_preamble("adsb_v1.0.0"))
    assert sample.metadata["family"] == "A"
    assert sample.metadata["source"] == "spec-synthesized"
    assert sample.metadata["task_text"] == DEMO_RECORDS[0]["input"]
    assert sample.metadata["frame_ids"] == ["adsb-0001"]  # the record's own metadata is kept


def test_task_premises_sit_between_preamble_and_task_text() -> None:
    """Per-task premises (catalogue C1 resync rule, B3 bus rate) follow the family preamble."""
    record = next(r for r in DEMO_RECORDS if r.get("premises"))
    sample = next(s for s in load_tasks(DEMO_FILE) if s.id == record["id"])
    assert isinstance(sample.input, str)
    preamble = load_preamble(record["preamble"])
    after_preamble = sample.input.removeprefix(preamble + "\n\n")
    assert after_preamble.startswith(TASK_PREMISES_HEADING + "\n")
    for premise in record["premises"]:
        assert f"- {premise}" in after_preamble
    assert after_preamble.endswith("\n\n" + record["input"])
    assert sample.metadata and sample.metadata["premises"] == record["premises"]


def test_task_without_premises_has_no_premises_heading() -> None:
    sample = load_tasks(DEMO_FILE)[0]
    assert isinstance(sample.input, str) and TASK_PREMISES_HEADING not in sample.input


def test_adsb_preamble_states_the_premises() -> None:
    """The premises the ADS-B tasks depend on are actually in the preamble."""
    text = load_preamble("adsb_v1.0.0")
    for phrase in ("DO-260B", "parity", "reject", "even and one odd", "10 seconds",
                   "cannot be determined"):
        assert phrase in text, phrase


def test_every_preamble_file_is_named_by_version() -> None:
    for path in PREAMBLE_DIR.glob("*.txt"):
        family, _, version = path.stem.partition("_v")
        assert family and version.count(".") == 2, path.name


def _first_user_message(log_samples: Any) -> str:
    messages = log_samples[0].messages
    return next(m.text for m in messages if isinstance(m, ChatMessageUser))


def test_model_receives_the_preamble(tmp_path: Path) -> None:
    """End to end through Inspect: the user message the model sees starts with the preamble."""
    task = Task(dataset=load_tasks(DEMO_FILE), solver=generate())
    [log] = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), limit=1,
                         display="none")
    assert log.status == "success", log.error
    assert log.samples
    sent = _first_user_message(log.samples)
    assert sent.startswith(load_preamble("adsb_v1.0.0"))
    assert sent.endswith(DEMO_RECORDS[0]["input"])


def test_system_message_cannot_displace_the_preamble(tmp_path: Path) -> None:
    """A guardrail or template system message is added alongside; the preamble stays put."""
    task = Task(dataset=load_tasks(DEMO_FILE),
                solver=[system_message("You are a careful analyst."), generate()])
    [log] = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), limit=1,
                         display="none")
    assert log.status == "success", log.error
    assert log.samples
    assert any(isinstance(m, ChatMessageSystem) for m in log.samples[0].messages)
    assert _first_user_message(log.samples).startswith(load_preamble("adsb_v1.0.0"))


# --- Acceptance criterion 2 (negative case): malformed samples -----------------


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_field_is_rejected_and_named(tmp_path: Path, field: str) -> None:
    record = with_changes(**{field: ...})
    if field == "id":
        expected = "record 1"
    else:
        expected = "'t-001'"
    with pytest.raises(TaskLoadError) as err:
        load_tasks(write_tasks(tmp_path, [record]))
    assert expected in str(err.value)
    assert f"missing field '{field}'" in str(err.value)


@pytest.mark.parametrize("change, message", [
    ({"family": "arinc-429"}, "family 'arinc-429' is not one of"),
    ({"input": "   "}, "field 'input' must be a non-empty string"),
    ({"source": 7}, "field 'source' must be a non-empty string"),
    ({"target": ""}, "field 'target' must be"),
    ({"target": {"callsign": "ZZZ101"}}, "field 'target' must be"),
    ({"family": "adsb"}, "family 'adsb' is not one of"),
    ({"metadata": ["not", "an", "object"]}, "field 'metadata' must be an object"),
    ({"premises": "Bus speed is 100 kbps."}, "field 'premises' must be"),
    ({"premises": []}, "field 'premises' must be"),
    ({"premises": ["ok", ""]}, "field 'premises' must be"),
])
def test_bad_value_is_rejected_and_named(tmp_path: Path, change: dict[str, Any],
                                         message: str) -> None:
    with pytest.raises(TaskLoadError, match="'t-001'") as err:
        load_tasks(write_tasks(tmp_path, [with_changes(**change)]))
    assert message in str(err.value)


def test_bad_record_is_identified_among_good_ones(tmp_path: Path) -> None:
    records = [with_changes(id="t-001"), with_changes(id="t-002", family=None),
               with_changes(id="t-003")]
    with pytest.raises(TaskLoadError, match="task 't-002'"):
        load_tasks(write_tasks(tmp_path, records))


def test_record_without_id_is_identified_by_position(tmp_path: Path) -> None:
    records = [with_changes(id="t-001"), with_changes(id=...)]
    with pytest.raises(TaskLoadError, match="record 2"):
        load_tasks(write_tasks(tmp_path, records))


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError, match="'t-001': duplicate task id"):
        load_tasks(write_tasks(tmp_path, [GOOD, GOOD]))


# --- Backlog negative case: preamble that does not exist ---------------------


@pytest.mark.parametrize("name", ["adsb_v9.9.9", "arinc429_v1.0.0", "../adsb_v1.0.0", ""])
def test_unknown_preamble_fails_and_names_the_task(tmp_path: Path, name: str) -> None:
    record = with_changes(preamble=name)
    with pytest.raises(TaskLoadError, match="'t-001'") as err:
        load_tasks(write_tasks(tmp_path, [record]))
    assert "preamble" in str(err.value)
