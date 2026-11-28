"""Demo for the inter-rater agreement pilot (WBS 3.9, issue #64). No server needed.

    python scripts/demo_irr_pilot.py

1. Runs the real A1, A2 and B4 tasks through Inspect on two stand-in models whose
   answers are scripted (bench/irr_scripted.py), writing ordinary EvalLogs.
2. Draws a blind 30-item pilot packet from those logs with ``bench.irr sample``.
3. Round 1: two raters grade the same packet, agreeing on all but one item.
   Agreement is computed and recorded (agreement.json / agreement.md), and the
   freeze gate opens.
4. Round 2: the same packet graded by two raters who resolve the borderline answers
   in opposite ways. Alpha falls below 0.800, REVISION.md sends the rubric back, and
   the freeze gate is blocked.

The answers and the grades are SCRIPTED stand-ins, not models or team members. In the
real pilot, two team members grade answers from the baseline logs (docs/irr_pilot.md).
Everything is written to logs/irr_demo/<UTC>/ (git-ignored); nothing goes in data/irr/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval as inspect_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from bench import irr  # noqa: E402
from bench.irr_scripted import standin_grades, standin_model  # noqa: E402

TASKS = ["tasks/a1_adsb_decode.py", "tasks/a2_cpr_position.py", "tasks/b4_arinc429_fault_vs_injection.py"]
MODELS = ["standin-x", "standin-y"]
SEED = 3909


def banner(text: str) -> None:
    print(f"\n{'#' * 78}\n# {text}\n{'#' * 78}")


def grade_round(pilot_dir: Path, raters: dict[str, tuple[str, frozenset[str]]]) -> dict:
    _, key = irr.read_packet(pilot_dir)
    ids = list(key["items"])
    for rater, (habit, flip) in raters.items():
        irr._write_grades(irr.grades_path(pilot_dir, rater), ids, standin_grades(key, habit, flip))
        print(f"  {rater}: grades/{rater}.csv ({habit} on borderline answers"
              + (f", one slip on {', '.join(sorted(flip))}" if flip else "") + ")")
    return key


def main() -> int:
    run_dir = REPO_ROOT / "logs" / "irr_demo" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir, irr_dir = run_dir / "logs", run_dir / "irr"

    banner("1. Run A1, A2 and B4 on two scripted stand-in models")
    logs = inspect_eval(TASKS, model=[standin_model(m) for m in MODELS], log_dir=str(log_dir),
                        display="none", seed=42, temperature=0.0)
    print(f"  {len(logs)} EvalLogs in {log_dir.relative_to(REPO_ROOT)}: "
          + ", ".join(f"{lg.eval.task}/{lg.eval.model.split('/')[-1]} ({lg.status})" for lg in logs))
    if any(lg.status != "success" for lg in logs):
        return 1

    banner("2. Draw a blind pilot packet")
    rc = irr.main(["sample", str(log_dir), "--pilot", "pilot-demo-1", "--seed", str(SEED), "--irr-dir", str(irr_dir)])
    if rc:
        return rc
    pilot1 = irr_dir / "pilot-demo-1"
    first = irr.read_packet(pilot1)[0][0]
    print(f"\n  The packet shows the rubric, the prompt and the answer; no model name, no score. {first['item_id']}:")
    print(f"    clause: {first['clause_id']}\n    rubric: {first['clause'][:110]}...")
    print("    answer: " + first["response"].replace("\n", "\n            "))

    banner("3. Round 1: two raters grade the same answers -> agreement computed and recorded")
    key = grade_round(pilot1, {"rater-a": ("strict", frozenset()),
                               "rater-b": ("strict", frozenset({next(iter(sorted(irr.read_packet(pilot1)[1]["items"])))}))})
    print()
    rc = irr.main(["score", str(pilot1), "--raters", "rater-a", "rater-b"])
    print(f"Recorded: {(pilot1 / irr.RECORD).relative_to(REPO_ROOT)} (exit {rc})")
    irr.main(["gate", "--irr-dir", str(irr_dir)])

    banner("4. Round 2: raters split on the borderline answers -> below threshold, rubric sent back")
    pilot2 = irr_dir / "pilot-demo-2"
    # The same packet as round 1 (same seed, same logs), graded by a different pair.
    pilot2.mkdir(parents=True)
    (pilot2 / irr.PACKET).write_text((pilot1 / irr.PACKET).read_text(encoding="utf-8"), encoding="utf-8")
    key2 = {**key, "pilot_id": pilot2.name}
    (pilot2 / irr.KEY).write_text(json.dumps(key2, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    grade_round(pilot2, {"rater-a": ("strict", frozenset()), "rater-c": ("lenient", frozenset())})
    print()
    rc = irr.main(["score", str(pilot2), "--raters", "rater-a", "rater-c"])
    print(f"(exit {rc})")
    revision = (pilot2 / irr.REVISION).read_text(encoding="utf-8")
    print(f"\n--- {(pilot2 / irr.REVISION).relative_to(REPO_ROOT)} (first 40 lines) ---")
    print("\n".join(revision.splitlines()[:40]))
    print("--- ...\n")
    irr.main(["gate", "--irr-dir", str(irr_dir)])
    print(f"\nEverything above is under {run_dir.relative_to(REPO_ROOT)}. The answers and grades were scripted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
