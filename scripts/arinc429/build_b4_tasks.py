"""Build the B4 ARINC 429 fault-versus-injection task file (WBS 3.6, issue #61).

    python scripts/arinc429/build_b4_tasks.py

Writes data/tasks/b4_arinc429_fault_vs_injection_v<B4_VERSION>.jsonl: seeded word
streams from one transmitter, some with injected or faulted words, with targets from
the triage reference (bench/arinc429/triage.py) and per-task unsafe-if clauses, then
validates it. The build fails unless the reference reaches exactly the verdict the
builder intended. The output is deterministic, and tests/test_b4_tasks.py checks the
committed file matches it. B4 uses the ARINC 429 family preamble (arinc429_v1.0.0);
each task's transmitter and limits are in its own premises.

Once a run has used a version, treat the file as frozen: change bench/arinc429/b4.py,
bump B4_VERSION, and build a new file instead of overwriting the old one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arinc429.b4 import TASK_FILE, validate_samples, write_task_file  # noqa: E402
from bench.loader import load_tasks  # noqa: E402


def main() -> None:
    path = write_task_file(TASK_FILE)
    dataset = load_tasks(path, verify=False)
    validate_samples(dataset)
    print(f"wrote {len(dataset)} B4 tasks to {path}")


if __name__ == "__main__":
    main()
