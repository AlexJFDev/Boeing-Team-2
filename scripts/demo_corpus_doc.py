"""Demo for the protocol corpus documentation (WBS 3.10, issue #65).

    python scripts/demo_corpus_doc.py

1. Looks up every task family in docs/corpus.md and prints its source spec, oracle and
   scoring tolerance, then one whole entry (B3) with the sources it cites.
2. Checks the committed doc against data/tasks: PASS.
3. Copies data/tasks to a temporary folder and makes one B4 task cite "adsb.lol", an
   approved data source (bench/boundary.py) that the doc does not list. The data-
   boundary check still passes; the documentation check fails and names the task, which
   is how review fails in CI.

Nothing in the repository is changed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench import boundary  # noqa: E402
from bench import corpus_doc as cd  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'#' * 78}\n# {text}\n{'#' * 78}")


def main() -> int:
    banner("1. Look up each family: source spec, oracle, scoring tolerance")
    doc = cd.parse_doc()
    for family, entry in sorted(doc.entries.items()):
        print(f"\n{entry.title}")
        for name in ("Source spec", "Oracle", "Tolerance"):
            print(f"  {name:<12} {entry.fields[name].text}")
    print("\nOne whole entry, as `python -m bench.corpus_doc --family B3` prints it:\n")
    cd.main(["--family", "B3"])

    banner("2. Check the committed doc against every task file")
    rc = cd.main([])
    if rc:
        return rc

    banner("3. A task cites a source the doc does not list -> review fails")
    with tempfile.TemporaryDirectory() as tmp:
        tasks = Path(tmp) / "tasks"
        shutil.copytree(cd.TASKS_DIR, tasks)
        path = tasks / "b4_arinc429_fault_vs_injection_v1.0.0.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows[2]["source"] = "adsb.lol"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"Changed {rows[2]['id']} in a temporary copy of {path.name}: source 'spec-synthesized' -> 'adsb.lol'\n")

        report = boundary.check_paths([path])
        print(f"Data boundary check (adsb.lol is an approved source): {report.summary()}\n")
        rc = cd.main(["--tasks", str(tasks)])
        print(f"\nCorpus documentation check exit code: {rc} (1 = fail, so the PR check fails)")
    return 0 if rc == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
