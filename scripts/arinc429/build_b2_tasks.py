"""Build the B2 ARINC 429 SSM interpretation task file (WBS 3.4, issue #59).

    python scripts/arinc429/build_b2_tasks.py

Writes data/tasks/b2_arinc429_ssm_interpretation_v<B2_VERSION>.jsonl from the seeded
word generator, with targets from the reference decoder, then validates it. The
output is deterministic, and tests/test_b2_tasks.py checks the committed file matches
it. B2 uses the ARINC 429 family preamble (arinc429_v1.0.0) that B1 introduced.

Once a run has used a version, treat the file as frozen: change bench/arinc429/b2.py,
bump B2_VERSION, and build a new file instead of overwriting the old one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arinc429.b2 import TASK_FILE, validate_samples, write_task_file  # noqa: E402
from bench.loader import load_tasks  # noqa: E402


def main() -> None:
    path = write_task_file(TASK_FILE)
    dataset = load_tasks(path)
    validate_samples(dataset)
    print(f"wrote {len(dataset)} B2 tasks to {path}")


if __name__ == "__main__":
    main()
