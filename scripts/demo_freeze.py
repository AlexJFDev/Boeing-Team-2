"""Demo for freezing protocol corpus v1.0 (WBS 3.11, issue #66).

    python scripts/demo_freeze.py

1. Runs the freeze checks on this repository as it is. Until two team members grade the
   inter-rater pilot (3.9), the irr check fails and the real freeze is blocked.
2. With a stand-in passing pilot record, freezes the real corpus into a temporary folder:
   the record holds the corpus hash. Commits it to a throwaway git repository and tags
   that commit, showing the hash recorded and tagged in Git.
3. Opens a review issue on task b3-07 (a stand-in issue, read from a JSON file instead
   of GitHub) and freezes again: blocked, nothing written.

Nothing in this repository changes, and no tag is created here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench import freeze, irr  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'#' * 78}\n# {text}\n{'#' * 78}")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=demo", "-c", "user.email=demo@example.com", *args],
                          cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    banner("1. This repository as it is: the freeze checks")
    freeze.main(["check"])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        irr_dir = root / "irr"
        pilot = irr_dir / "pilot-standin"
        pilot.mkdir(parents=True)
        (pilot / irr.RECORD).write_text(json.dumps({
            "rubric_fingerprint": irr.rubric_fingerprint(), "status": "pass", "threshold": irr.ALPHA_THRESHOLD,
            "alpha": 0.9, "raters": ["standin-a", "standin-b"], "scored_at": "2026-11-27T00:00:00+00:00"}))
        no_issues = root / "issues.json"
        no_issues.write_text("[]")

        banner("2. Freeze v1.0 (stand-in passing pilot, no open review issues) and tag it")
        git_root = root / "repo"
        out = git_root / "data" / "frozen"
        rc = freeze.main(["freeze", "--out-dir", str(out), "--irr-dir", str(irr_dir),
                          "--review-issues", str(no_issues)])
        if rc:
            return rc
        record = json.loads(freeze.record_path(out).read_text(encoding="utf-8"))
        print("\nPer-family run hashes (what each family's EvalLogs record):")
        for fam, entry in record["families"].items():
            print(f"  {fam}  {entry['run_corpus_hash'][:16]}...  {entry['task_file']} "
                  f"(task {entry['task_version']}, {entry['tasks']} tasks)")

        git(git_root, "init", "-q")
        git(git_root, "add", "-A")
        git(git_root, "commit", "-qm", "Freeze protocol corpus v1.0.0")
        print("\nCommitted the record to a throwaway repository, then:")
        rc = freeze.main(["tag", "--out-dir", str(out), "--git-root", str(git_root)])
        if rc:
            return rc
        name = freeze.tag_name()
        print(f"\n$ git tag -n20 {name}")
        print(git(git_root, "tag", "-n20", name))
        print(f"\nThe tag points at commit {git(git_root, 'rev-list', '-n1', name)[:12]}, "
              f"which holds {freeze.record_path(out).relative_to(git_root).as_posix()}.")

        banner("3. A review issue opened on task b3-07 -> the freeze is blocked")
        review = root / "review_issues.json"
        review.write_text(json.dumps([{"number": 999, "title": "Review b3-07: is the gap rule's answer right?",
                                       "body": "Stand-in for an issue labelled task-review."}]))
        out2 = root / "frozen-again"
        rc = freeze.main(["freeze", "--out-dir", str(out2), "--irr-dir", str(irr_dir), "--review-issues", str(review)])
        print(f"\nExit code {rc}; record written: {freeze.record_path(out2).exists()}")
        return 0 if rc == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
