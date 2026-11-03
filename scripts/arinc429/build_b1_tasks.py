"""Build the B1 ARINC 429 word-decode task file (WBS 3.3, issue #58).

    python scripts/arinc429/build_b1_tasks.py

Writes data/tasks/b1_arinc429_word_decode_v<B1_VERSION>.jsonl from the seeded word
generator, with targets from the reference decoder, then validates it. The output is
deterministic, and tests/test_b1_tasks.py checks the committed file matches it.

The family preamble data/preambles/arinc429_v1.0.0.txt is rendered from the label
profiles by ``bench.arinc429.b1.render_preamble``. Pass --preamble to write it; it is
only needed when the preamble is first created or moved to a new version.

Once a run has used a version, treat the file as frozen: change bench/arinc429/b1.py,
bump B1_VERSION, and build a new file instead of overwriting the old one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.arinc429.b1 import TASK_FILE, validate_samples, write_preamble, write_task_file  # noqa: E402
from bench.loader import load_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preamble", action="store_true", help="also write the family preamble")
    args = parser.parse_args()
    if args.preamble:
        print(f"wrote {write_preamble()}")
    path = write_task_file(TASK_FILE)
    dataset = load_tasks(path)
    validate_samples(dataset)
    print(f"wrote {len(dataset)} B1 tasks to {path}")


if __name__ == "__main__":
    main()
