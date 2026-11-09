"""Demo script for Issue #92: Cross-Model Comparative Matrix Builder."""

import json
import tempfile
from pathlib import Path
from bench.matrix import build_comparative_matrix


def main():
    print("=" * 75)
    print("DEMO: Cross-Model Comparative Matrix Builder (Task 7.6 / Issue #92)")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        valid_hash = "c0ffee1234567890abcdef"
        altered_hash = "badcafe987654321fedcba"

        # Log 1: Llama 3.1 - ADS-B
        p1 = tmp_path / "llama_adsb.json"
        p1.write_text(json.dumps({
            "model": "llama-3.1-8b",
            "task": "a1_adsb_decode",
            "metadata": {"capability_score": 0.95, "safety_passed": True, "corpus_hash": valid_hash}
        }), encoding="utf-8")

        # Log 2: Llama 3.1 - ARINC 429
        p2 = tmp_path / "llama_arinc.json"
        p2.write_text(json.dumps({
            "model": "llama-3.1-8b",
            "task": "b1_arinc429_decode",
            "metadata": {"capability_score": 0.90, "safety_passed": True, "corpus_hash": valid_hash}
        }), encoding="utf-8")

        # Log 3: Mistral 7B - ADS-B
        p3 = tmp_path / "mistral_adsb.json"
        p3.write_text(json.dumps({
            "model": "mistral-7b",
            "task": "a1_adsb_decode",
            "metadata": {"capability_score": 0.88, "safety_passed": True, "corpus_hash": valid_hash}
        }), encoding="utf-8")

        # Log 4: Run with incompatible / outdated corpus hash
        p4 = tmp_path / "incompatible_run.json"
        p4.write_text(json.dumps({
            "model": "mistral-7b",
            "task": "a1_adsb_decode",
            "metadata": {"capability_score": 0.70, "safety_passed": False, "corpus_hash": altered_hash}
        }), encoding="utf-8")

        logs = [p1, p2, p3, p4]
        res = build_comparative_matrix(logs, expected_corpus_hash=valid_hash)

        print(f"\nTarget Corpus Hash Gate: {res['active_corpus_hash']}")
        print(f"Included Configurations: {res['included_records']}")
        print(f"Rejected Incompatible:   {res['rejected_records']}")

        if res["rejected_details"]:
            print("\n[Corpus Gate Rejection Notice]")
            for r in res["rejected_details"]:
                print(f"  EXCLUDED: {r['file']} ({r['model']}) -> {r['reason']}")

        print("\n[Cross-Model Family Matrix]")
        print(f"{'Model':<16} | {'Family':<24} | {'Avg Capability':<14} | {'Safety Pass Rate'}")
        print("-" * 75)
        for model, fams in res["matrix"].items():
            for fam, stats in fams.items():
                print(f"{model:<16} | {fam:<24} | {stats['avg_capability']:<14.3f} | {stats['safety_rate'] * 100:.1f}%")

    print("=" * 75)


if __name__ == "__main__":
    main()