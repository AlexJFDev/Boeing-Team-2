"""Unit tests for EvalLog cryptographic integrity audit (Issue #91)."""

import json
import pytest
from bench.audit import (
    compute_file_sha256,
    generate_log_manifest,
    verify_log_integrity,
    audit_log_directory,
)


def test_audit_untouched_log_verifies(tmp_path):
    """Untouched log files must verify with expected hash."""
    log_file = tmp_path / "run_sample.eval"
    log_file.write_text('{"task": "adsb_test", "status": "success"}', encoding="utf-8")

    manifest = generate_log_manifest([log_file])
    expected_hash = manifest[log_file.name]

    passed, actual, msg = verify_log_integrity(log_file, expected_hash)
    assert passed is True
    assert actual == expected_hash
    assert "PASSED" in msg


def test_audit_single_byte_edit_detected_as_tampered(tmp_path):
    """Modifying even one byte of a log must be reported as tampered."""
    log_file = tmp_path / "run_sample.eval"
    log_file.write_text('{"score": 1.0, "status": "success"}', encoding="utf-8")

    # Lock initial manifest hash
    initial_hash = compute_file_sha256(log_file)

    # Tamper with single byte (1.0 -> 0.0)
    log_file.write_text('{"score": 0.0, "status": "success"}', encoding="utf-8")

    passed, actual, msg = verify_log_integrity(log_file, initial_hash)
    assert passed is False
    assert actual != initial_hash
    assert "TAMPERED" in msg


def test_audit_directory_batch(tmp_path):
    log1 = tmp_path / "run1.eval"
    log2 = tmp_path / "run2.eval"
    log1.write_text('{"run": 1}', encoding="utf-8")
    log2.write_text('{"run": 2}', encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"
    manifest_data = generate_log_manifest([log1, log2])
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    report = audit_log_directory(manifest_file, tmp_path)
    assert report["all_passed"] is True
    assert report["total_audited"] == 2