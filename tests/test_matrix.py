"""Unit tests for Cross-Model Comparative Matrix Builder (Issue #92)."""

import json
import pytest
from bench.matrix import build_comparative_matrix, extract_task_family


def test_task_family_extraction():
    assert extract_task_family("a1_adsb_decode") == "A-Family (ADS-B)"
    assert extract_task_family("b2_arinc429_ssm") == "B-Family (ARINC 429)"
    assert extract_task_family("c1_afdx_pcap") == "C-Family (AFDX)"
    assert extract_task_family("d2_prompt_injection") == "D-Family (Safety / Injection)"


def test_matrix_builder_aggregation(tmp_path):
    log1 = tmp_path / "run_llama_adsb.json"
    log2 = tmp_path / "run_llama_arinc.json"

    corpus_sha = "abc1234567890def"
    log1.write_text(json.dumps({
        "model": "llama3.1",
        "task": "a1_adsb_decode",
        "corpus_hash": corpus_sha,
        "score": 1.0,
        "metadata": {"safety_passed": True, "corpus_hash": corpus_sha, "capability_score": 1.0}
    }), encoding="utf-8")

    log2.write_text(json.dumps({
        "model": "llama3.1",
        "task": "b1_arinc429_decode",
        "corpus_hash": corpus_sha,
        "score": 0.8,
        "metadata": {"safety_passed": True, "corpus_hash": corpus_sha, "capability_score": 0.8}
    }), encoding="utf-8")

    result = build_comparative_matrix([log1, log2], expected_corpus_hash=corpus_sha)
    assert result["included_records"] == 2
    assert result["rejected_records"] == 0
    assert "llama3.1" in result["matrix"]
    assert result["matrix"]["llama3.1"]["A-Family (ADS-B)"]["avg_capability"] == 1.0
    assert result["matrix"]["llama3.1"]["B-Family (ARINC 429)"]["avg_capability"] == 0.8


def test_matrix_builder_corpus_gate_rejection(tmp_path):
    """Logs with different corpus hashes must be excluded from comparison."""
    log_valid = tmp_path / "log_corpus_v1.json"
    log_mismatch = tmp_path / "log_corpus_v2.json"

    log_valid.write_text(json.dumps({
        "model": "mistral-7b",
        "task": "a1_adsb_decode",
        "corpus_hash": "corpus_hash_v1_0_0",
        "score": 1.0,
        "metadata": {"safety_passed": True, "corpus_hash": "corpus_hash_v1_0_0"}
    }), encoding="utf-8")

    log_mismatch.write_text(json.dumps({
        "model": "mistral-7b",
        "task": "a1_adsb_decode",
        "corpus_hash": "corpus_hash_v2_0_0_DIFFERENT",
        "score": 0.5,
        "metadata": {"safety_passed": True, "corpus_hash": "corpus_hash_v2_0_0_DIFFERENT"}
    }), encoding="utf-8")

    result = build_comparative_matrix([log_valid, log_mismatch], expected_corpus_hash="corpus_hash_v1_0_0")
    assert result["included_records"] == 1
    assert result["rejected_records"] == 1
    assert result["rejected_details"][0]["file"] == "log_corpus_v2.json"