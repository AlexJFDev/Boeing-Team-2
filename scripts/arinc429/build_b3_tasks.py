"""Build the B3 ARINC 429 bus conformance task file (WBS 3.5, issue #60).

    python scripts/arinc429/build_b3_tasks.py

Writes data/tasks/b3_arinc429_bus_conformance_v<B3_VERSION>.jsonl: timed word streams
built from the seeded word generator, with targets from the conformance checker
(bench/arinc429/conformance.py), then validates it. The build fails unless the checker
finds exactly the violations the builder placed. The output is deterministic, and
tests/test_b3_tasks.py checks the committed file matches it. B3 uses the ARINC 429
family preamble (arinc429_v1.0.0); each task's bus speed, gap, word timing and label
rates are in its own premises.

Once a run has used a version, treat the file as frozen: change bench/arinc429/b3.py,
bump B3_VERSION, and build a new file instead of overwriting the old one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arinc429.b3 import TASK_FILE, validate_samples, write_task_file  # noqa: E402
from bench.loader import load_tasks  # noqa: E402


def main() -> None:
    path = write_task_file(TASK_FILE)
    dataset = load_tasks(path, verify=False)
    validate_samples(dataset)
    print(f"wrote {len(dataset)} B3 tasks to {path}")


if __name__ == "__main__":
    main()
