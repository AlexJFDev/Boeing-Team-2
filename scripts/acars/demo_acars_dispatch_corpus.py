"""Demo for task 5.1: sample messages next to their labeled fields and the checklist.

Usage (from the repo root):
    python scripts/demo_acars_dispatch_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.acars.dispatch_corpus import (  # noqa: E402
    CHECKLIST, CorpusRejection, add_message, compute_reference, load_corpus,
    sanitization_violations,
)
import build_acars_dispatch_corpus as gen  # noqa: E402

SHOW = ("valid", "valid_lookalike", "malformed", "label_mismatch", "adversarial_free_text")


def main() -> None:
    manifest = json.loads(gen.MANIFEST.read_text())
    corpus = load_corpus(gen.JSONL, gen.MANIFEST)
    print(f"Corpus {manifest['name']} v{manifest['version']}  "
          f"({manifest['message_count']} messages)")
    print(f"  source : {manifest['source']['kind']} ({manifest['source']['generator']})")
    print(f"  sha256 : {manifest['sha256']}")
    print(f"  boundary: {manifest['data_boundary']}\n")

    print("== Sample messages and labeled fields ==")
    for cat in SHOW:
        m = next(x for x in corpus if x["reference"]["category"] == cat)
        r = m["reference"]
        failed = {v.code for v in sanitization_violations(m)}
        print(f"\n[{cat}] {m['id']}")
        print(f"  mode={m['mode']} label={m['label']} registration={m['registration']} "
              f"flight_id={m['flight_id']}")
        print(f"  text      : {m['text']}")
        print(f"  reference : well_formed={r['well_formed']} "
              f"body_matches_label={r['body_matches_label']} "
              f"free_text={r['free_text_fields']} "
              f"instruction_like={r['contains_instruction_like_text']}")
        print(f"  fields    : {r['fields']}")
        print("  checklist : " + " ".join(f"{i}={'FAIL' if i in failed else 'ok'}"
                                         for i, _ in CHECKLIST))

    print("\n== Sanitization checklist ==")
    for i, rule in CHECKLIST:
        print(f"  {i}  {rule}")

    print("\n== Rejection demo: messages containing real-looking identifiers ==")
    base = next(x for x in corpus if x["reference"]["category"] == "valid")
    for label, edit in (
        ("real tail number", {"registration": "N12345"}),
        ("real flight id", {"flight_id": "UA123"}),
        ("tail in free text", {"text": base["text"] + " RMK TAIL N12345"}),
    ):
        bad = {**base, "id": "DEMO-1", **edit}
        bad["reference"] = {**base["reference"], **compute_reference(bad["label"], bad["text"])}
        try:
            add_message([], bad)
            print(f"  {label}: ACCEPTED (unexpected)")
        except CorpusRejection as e:
            print(f"  {label}: REJECTED -> " + "; ".join(str(v) for v in e.violations))


if __name__ == "__main__":
    main()
