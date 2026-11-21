"""Demo script for Issue #91: EvalLog Cryptographic Integrity Audit."""

import json
import tempfile
from pathlib import Path
from bench.audit import (
    compute_file_sha256,
    generate_log_manifest,
    verify_log_integrity,
)


def main():
    print("=" * 70)
    print("DEMO: EvalLog Cryptographic Integrity Audit (Task 7.5 / Issue #91)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        log1 = tmp_path / "adsb_eval_20261109.eval"
        log2 = tmp_path / "arinc429_eval_20261109.eval"

        log1.write_text('{"task": "a1_adsb_decode", "score": 1.0}', encoding="utf-8")
        log2.write_text('{"task": "b1_arinc429_decode", "score": 1.0}', encoding="utf-8")

        # Step 1: Generate reference manifest
        manifest = generate_log_manifest([log1, log2])
        print("\n[Step 1: Generated Initial Log Manifest]")
        for fname, h in manifest.items():
            print(f"  {fname} -> SHA-256: {h[:16]}...")

        # Step 2: Audit untouched logs
        print("\n[Step 2: Auditing Untouched Logs]")
        for fname, h in manifest.items():
            passed, actual, msg = verify_log_integrity(tmp_path / fname, h)
            print(f"  {fname}: {msg}")

        # Step 3: Tamper with one single byte in log1
        print("\n[Step 3: Tampering with 1 byte in adsb_eval_20261109.eval]")
        log1.write_text('{"task": "a1_adsb_decode", "score": 0.0}', encoding="utf-8")

        # Step 4: Re-audit to demonstrate tamper detection
        print("\n[Step 4: Re-Auditing Logs After Tamper]")
        for fname, h in manifest.items():
            passed, actual, msg = verify_log_integrity(tmp_path / fname, h)
            status_tag = "VALID" if passed else "SECURITY ALERT"
            print(f"  [{status_tag}] {msg}")

    print("=" * 70)


if __name__ == "__main__":
    main()