"""EvalLog Cryptographic Integrity Audit Module (Sprint 4, Task 7.5, Issue #91)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AUDIT_MODULE_VERSION = "1.0.0"


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute the SHA-256 hash of a file's raw bytes."""
    path = Path(filepath)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_log_manifest(log_paths: List[str | Path]) -> Dict[str, str]:
    """Generate an integrity manifest mapping log file names to SHA-256 hashes."""
    manifest = {}
    for p in log_paths:
        path = Path(p)
        if path.is_file():
            manifest[path.name] = compute_file_sha256(path)
    return manifest


def verify_log_integrity(
    log_path: str | Path,
    expected_hash: str,
) -> Tuple[bool, str, str]:
    """Verify an individual log file against its expected SHA-256 hash.

    Returns:
        (passed, actual_hash, status_message)
    """
    path = Path(log_path)
    if not path.is_file():
        return False, "", f"File not found: {path.name}"

    actual_hash = compute_file_sha256(path)
    if actual_hash == expected_hash:
        return True, actual_hash, f"PASSED (verified: {path.name})"
    else:
        return False, actual_hash, (
            f"TAMPERED (hash mismatch on {path.name}: "
            f"expected {expected_hash[:12]}..., got {actual_hash[:12]}...)"
        )


def audit_log_directory(
    manifest_path: str | Path,
    log_dir: str | Path,
) -> Dict[str, Any]:
    """Audit all logs referenced in a manifest within a directory."""
    m_path = Path(manifest_path)
    l_dir = Path(log_dir)

    with open(m_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = {}
    all_passed = True

    for filename, exp_hash in manifest.items():
        target_file = l_dir / filename
        passed, actual, msg = verify_log_integrity(target_file, exp_hash)
        results[filename] = {
            "passed": passed,
            "actual_hash": actual,
            "expected_hash": exp_hash,
            "status": msg,
        }
        if not passed:
            all_passed = False

    return {
        "all_passed": all_passed,
        "total_audited": len(manifest),
        "results": results,
    }