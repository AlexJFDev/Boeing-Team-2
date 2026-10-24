"""Build the A1 ADS-B decode task file (WBS 2.5, issue #48).

    python scripts/adsb/build_a1_tasks.py

Writes data/tasks/a1_adsb_decode_v<A1_VERSION>.jsonl from the frame set in
data/adsb, with targets from the pyModeS oracle, then validates it. The output is
deterministic, and tests/test_a1_tasks.py checks the committed file matches it.

Once a run has used a version, treat the file as frozen: change bench/adsb/a1.py,
bump A1_VERSION, and build a new file instead of overwriting the old one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.adsb.a1 import TASK_FILE, validate_samples, write_task_file  # noqa: E402
from bench.loader import load_tasks  # noqa: E402


def main() -> None:
    path = write_task_file(TASK_FILE)
    dataset = load_tasks(path)
    validate_samples(dataset)
    print(f"wrote {len(dataset)} A1 tasks to {path}")


if __name__ == "__main__":
    main()
