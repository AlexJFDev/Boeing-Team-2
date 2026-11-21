"""Cross-Model Comparative Matrix Builder (Sprint 4, Task 7.6, Issue #92)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_task_family(task_id: str) -> str:
    """Classify task identifier into its protocol family."""
    task_lower = task_id.lower()
    if task_lower.startswith("a") or "adsb" in task_lower:
        return "A-Family (ADS-B)"
    if task_lower.startswith("b") or "arinc429" in task_lower:
        return "B-Family (ARINC 429)"
    if task_lower.startswith("c") or "afdx" in task_lower:
        return "C-Family (AFDX)"
    if task_lower.startswith("d") or "injection" in task_lower or "safety" in task_lower:
        return "D-Family (Safety / Injection)"
    return "Other"


def load_eval_record(log_path: str | Path) -> Dict[str, Any]:
    """Load evaluation record and extract model, task, corpus hash, and scores."""
    p = Path(log_path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle native inspect format or benchmark runner structure
    task_id = data.get("task", data.get("task_id", "unknown"))
    model = data.get("model", data.get("eval", {}).get("model", "unknown_model"))
    
    # Metadata extraction
    metadata = data.get("metadata", data.get("eval", {}).get("metadata", {}))
    corpus_hash = (
        metadata.get("corpus_hash")
        or data.get("corpus_hash")
        or "unknown_corpus"
    )

    cap_score = (
        metadata.get("capability_score")
        if "capability_score" in metadata
        else data.get("score", 0.0)
    )
    safety_passed = metadata.get("safety_passed", True)

    return {
        "file": p.name,
        "model": model,
        "task_id": task_id,
        "family": extract_task_family(task_id),
        "corpus_hash": corpus_hash,
        "capability_score": float(cap_score) if cap_score is not None else 0.0,
        "safety_passed": bool(safety_passed),
    }


def build_comparative_matrix(
    log_files: List[str | Path],
    expected_corpus_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate logs by model and task family, rejecting mismatched corpus hashes."""
    valid_records: List[Dict[str, Any]] = []
    rejected_logs: List[Dict[str, str]] = []

    # Identify reference corpus hash if not provided
    ref_hash = expected_corpus_hash

    for lf in log_files:
        rec = load_eval_record(lf)
        if ref_hash is None:
            ref_hash = rec["corpus_hash"]

        if rec["corpus_hash"] != ref_hash:
            rejected_logs.append({
                "file": rec["file"],
                "model": rec["model"],
                "corpus_hash": rec["corpus_hash"],
                "reason": f"Corpus hash mismatch: expected {ref_hash[:10]}..., got {rec['corpus_hash'][:10]}...",
            })
        else:
            valid_records.append(rec)

    # Build matrix aggregated by model and family
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in valid_records:
        m = r["model"]
        fam = r["family"]
        if m not in matrix:
            matrix[m] = {}
        if fam not in matrix[m]:
            matrix[m][fam] = {
                "capability_scores": [],
                "safety_passes": 0,
                "total_runs": 0,
            }
        matrix[m][fam]["capability_scores"].append(r["capability_score"])
        if r["safety_passed"]:
            matrix[m][fam]["safety_passes"] += 1
        matrix[m][fam]["total_runs"] += 1

    # Summarize metrics
    summary_matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m, fams in matrix.items():
        summary_matrix[m] = {}
        for fam, stats in fams.items():
            avg_cap = sum(stats["capability_scores"]) / stats["total_runs"] if stats["total_runs"] else 0.0
            safety_rate = stats["safety_passes"] / stats["total_runs"] if stats["total_runs"] else 0.0
            summary_matrix[m][fam] = {
                "avg_capability": round(avg_cap, 3),
                "safety_rate": round(safety_rate, 3),
                "total_runs": stats["total_runs"],
            }

    return {
        "active_corpus_hash": ref_hash,
        "included_records": len(valid_records),
        "rejected_records": len(rejected_logs),
        "rejected_details": rejected_logs,
        "matrix": summary_matrix,
    }