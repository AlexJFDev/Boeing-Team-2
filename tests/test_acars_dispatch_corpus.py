"""Tests for the sanitized ACARS dispatch corpus (task 5.1, issue #76).

Run from the repo root:  python -m pytest tests/test_acars_dispatch_corpus.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "acars"))

from bench.acars.dispatch_corpus import (  # noqa: E402
    CHECKLIST, GRAMMARS, IN_SCOPE_LABELS, REQUIRED_FIELDS, REQUIRED_REFERENCE_FIELDS,
    CorpusRejection, add_message, compute_reference, corpus_hash, load_corpus, parse_body,
    serialize, validate_message,
)

import build_acars_dispatch_corpus as gen  # type: ignore[import-not-found]  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(gen.JSONL, gen.MANIFEST)


def make(text="REL SYN1234 ZZ01-ZZ02 FUEL 5000", label="H1", **over):
    msg = {
        "id": "T-1", "mode": "2", "label": label,
        "registration": "SYN-A001", "flight_id": "SYN1234", "text": text,
        "sanitization": {"method": "synthesized", "checklist_version": "1.0"},
    }
    ref = compute_reference(label, text)
    ref["category"] = "valid" if ref["well_formed"] else "malformed"
    ref["contains_instruction_like_text"] = False
    msg["reference"] = ref
    msg.update(over)
    return msg


def codes(msg):
    return {v.code for v in validate_message(msg)}


# ---------------------------------------------------- AC1: fields + checklist
def test_corpus_loads_and_matches_manifest(corpus):
    manifest = json.loads(gen.MANIFEST.read_text())
    assert len(corpus) == manifest["message_count"] == 63
    assert corpus_hash(corpus) == manifest["sha256"]
    assert manifest["source"]["kind"] == "synthetic"
    assert manifest["version"] and manifest["grammar_version"]


def test_every_message_has_labeled_fields_and_reference_values(corpus):
    for m in corpus:
        for f in REQUIRED_FIELDS:
            assert f in m, (m["id"], f)
        for f in REQUIRED_REFERENCE_FIELDS:
            assert f in m["reference"], (m["id"], f)
        assert m["mode"] and m["label"] in IN_SCOPE_LABELS
        assert m["registration"] and m["text"]


def test_every_message_passes_every_checklist_item(corpus):
    assert all(validate_message(m) == [] for m in corpus)
    assert [c[0] for c in CHECKLIST] == [f"S{i}" for i in range(1, 9)]


def test_reference_values_follow_from_structure(corpus):
    for m in corpus:
        derived = compute_reference(m["label"], m["text"])
        for k, v in derived.items():
            assert m["reference"][k] == v, (m["id"], k)


def test_corpus_covers_every_label_and_category(corpus):
    seen = {(m["label"], m["reference"]["category"]) for m in corpus}
    cats = {"valid", "valid_lookalike", "malformed", "label_mismatch", "adversarial_free_text"}
    assert seen == {(l, c) for l in IN_SCOPE_LABELS for c in cats}


def test_safety_labelled_share_is_about_a_fifth(corpus):
    share = sum(m["reference"]["contains_instruction_like_text"] for m in corpus) / len(corpus)
    assert 0.15 <= share <= 0.25


def test_free_text_fields_are_identified(corpus):
    for m in corpus:
        if m["reference"]["contains_instruction_like_text"]:
            assert m["reference"]["free_text_fields"] == ["rmk"]
            assert "rmk" in m["reference"]["fields"]


def test_each_body_matches_at_most_one_grammar(corpus):
    for m in corpus:
        hits = [l for l, g in GRAMMARS.items() if g.fullmatch(m["text"])]
        assert len(hits) <= 1


# ------------------------------------------------- AC2: real ids are rejected
@pytest.mark.parametrize("over,expected", [
    ({"registration": "N12345"}, "S1"),
    ({"registration": "G-ABCD"}, "S1"),
    ({"registration": "SYN-A001X"}, "S1"),
    ({"flight_id": "UA123"}, "S2"),
    ({"flight_id": "UAL1234"}, "S2"),
])
def test_real_looking_structured_fields_are_rejected(over, expected):
    assert expected in codes(make(**over))


@pytest.mark.parametrize("rmk,expected", [
    ("AIRCRAFT N12345 DELAYED", "S3"),
    ("AIRCRAFT N123AB DELAYED", "S3"),
    ("TAIL G-ABCD AT GATE", "S3"),
    ("TAIL VH-OQA AT GATE", "S3"),
    ("SEE UA123 FOR CONNECTION", "S4"),
    ("SEE UAL123 FOR CONNECTION", "S4"),
    ("SEE 9W402 FOR CONNECTION", "S4"),
    ("CONTACT ops@example.com", "S5"),
    ("SEE HTTPS://EXAMPLE.COM/X", "S5"),
    ("CALL 703-555-0142", "S5"),
    ("POS 38.9445 77.4558", "S6"),
    ("POS N3856W07712", "S6"),
    ("DEPARTED 2024-05-17", "S7"),
])
def test_real_identifiers_in_text_are_rejected(rmk, expected):
    msg = make(f"REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK {rmk}")
    assert expected in codes(msg), rmk


def test_lowercase_real_identifier_is_still_rejected():
    assert "S3" in codes(make("REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK tail n12345 late"))


def test_non_ascii_or_multiline_text_is_rejected():
    assert "S8" in codes(make("REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK CAF\u00c9"))
    assert "S8" in codes(make("REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK A\nB"))


def test_synthetic_identifiers_in_text_are_allowed():
    msg = make("REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK SEE SYN-A002 AND SYN4321 AT G07")
    assert validate_message(msg) == []


def test_add_message_rejects_real_identifier_and_leaves_corpus_untouched():
    store: list = []
    add_message(store, make())
    before = copy.deepcopy(store)
    bad = make("REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK TAIL N12345", **{"id": "T-2"})
    with pytest.raises(CorpusRejection) as exc:
        add_message(store, bad)
    assert any(v.code == "S3" for v in exc.value.violations)
    assert store == before


@pytest.mark.parametrize("missing", list(REQUIRED_FIELDS))
def test_missing_required_field_is_rejected(missing):
    msg = make()
    del msg[missing]
    with pytest.raises(CorpusRejection):
        add_message([], msg)


def test_duplicate_id_is_rejected():
    store: list = []
    add_message(store, make())
    with pytest.raises(CorpusRejection):
        add_message(store, make())


def test_wrong_reference_value_is_rejected():
    msg = make()
    msg["reference"]["well_formed"] = False
    assert "F3" in codes(msg)


def test_category_that_contradicts_structure_is_rejected():
    msg = make("REL SYN1234 ZZ01")             # malformed body
    msg["reference"]["category"] = "valid"
    assert "F4" in codes(msg)


# --------------------------------------------- load-time data boundary check
def _write_variant(tmp_path, mutate):
    lines = gen.JSONL.read_text().splitlines()
    msgs = [json.loads(l) for l in lines]
    mutate(msgs)
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_bytes(serialize(msgs))
    manifest = json.loads(gen.MANIFEST.read_text())
    manifest["sha256"] = corpus_hash(msgs)      # re-hash so only the boundary check can fail
    manifest["message_count"] = len(msgs)
    man = tmp_path / "m.json"
    man.write_text(json.dumps(manifest))
    return jsonl, man


def test_load_fails_when_a_real_registration_is_smuggled_in(tmp_path):
    def mutate(msgs):
        msgs[0]["registration"] = "N98765"
    jsonl, man = _write_variant(tmp_path, mutate)
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(jsonl, man)
    assert "data boundary check" in str(exc.value)
    assert any(v.code == "S1" for v in exc.value.violations)


def test_load_fails_when_a_real_flight_id_is_in_text(tmp_path):
    def mutate(msgs):
        m = next(x for x in msgs if x["label"] == "5Z" and x["reference"]["category"] == "valid")
        m["text"] += " RMK SEE UAL123"
        m["reference"] = {**m["reference"], **compute_reference("5Z", m["text"])}
    jsonl, man = _write_variant(tmp_path, mutate)
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(jsonl, man)
    assert any(v.code == "S4" for v in exc.value.violations)


def test_load_fails_on_hash_mismatch(tmp_path):
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_bytes(gen.JSONL.read_bytes().replace(b"NIL", b"NIM", 1))
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(jsonl, gen.MANIFEST)
    assert exc.value.violations[0].code == "P2"


def test_load_fails_when_manifest_lacks_provenance(tmp_path):
    manifest = json.loads(gen.MANIFEST.read_text())
    del manifest["source"]
    man = tmp_path / "m.json"
    man.write_text(json.dumps(manifest))
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(gen.JSONL, man)
    assert exc.value.violations[0].code == "P1"


# ------------------------------------------- added: gaps found in review
@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected(text):
    assert "F2" in codes(make(text))


def test_out_of_scope_mode_and_label_are_rejected():
    assert "F2" in codes(make(mode="1"))
    msg = make()
    msg["label"] = "Q0"
    assert "F2" in codes(msg)


def test_unknown_category_is_rejected():
    msg = make()
    msg["reference"]["category"] = "mystery"
    assert "F4" in codes(msg)


def test_instruction_flag_must_match_category():
    msg = make()
    msg["reference"]["contains_instruction_like_text"] = True
    assert "F4" in codes(msg)


@pytest.mark.parametrize("bad", [
    {"method": "scraped", "checklist_version": "1.0"},
    {"method": "synthesized", "checklist_version": "0.9"},
    {},
    "synthesized",
])
def test_bad_sanitization_record_is_rejected(bad):
    assert "F6" in codes(make(sanitization=bad))


@pytest.mark.parametrize("rmk,expected", [
    ("DEPARTED 05/17/2024", "S7"),
    ("DEPARTED 5/17/24", "S7"),
    ("DEPARTED 20240517", "S7"),
    ("DEPARTED 17MAY24", "S7"),
    ("DEPARTED 17MAY2024", "S7"),
    ("CALL 7035550142", "S5"),
    ("CALL +17035550142", "S5"),
])
def test_wider_date_and_phone_forms_are_rejected(rmk, expected):
    assert expected in codes(make(f"REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK {rmk}"))


def test_conservative_false_positive_is_rejected_by_design():
    assert "S4" in codes(make("REL SYN1234 ZZ01-ZZ02 FUEL 5000 RMK ENG1 CHECK"))


def test_ordinary_times_fuel_and_gates_do_not_trigger_dates_or_phones(corpus):
    assert all(not ({"S5", "S7"} & {v.code for v in validate_message(m)}) for m in corpus)


def _write_manifest_variant(tmp_path, mutate):
    manifest = json.loads(gen.MANIFEST.read_text())
    mutate(manifest)
    man = tmp_path / "m.json"
    man.write_text(json.dumps(manifest))
    return man


@pytest.mark.parametrize("field,value", [
    ("grammar_version", "syn-acars-grammar-0"),
    ("checklist_version", "0.9"),
    ("category_counts", {"valid": 63}),
])
def test_load_fails_when_manifest_versions_or_counts_disagree(tmp_path, field, value):
    man = _write_manifest_variant(tmp_path, lambda m: m.__setitem__(field, value))
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(gen.JSONL, man)
    assert exc.value.violations[0].code == "P5"


def test_load_fails_when_source_is_not_synthetic(tmp_path):
    man = _write_manifest_variant(tmp_path, lambda m: m["source"].__setitem__("kind", "captured"))
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(gen.JSONL, man)
    assert exc.value.violations[0].code == "P5"


# ------------------------------------------------------- reproducibility
def test_regeneration_is_byte_identical_to_committed_files():
    assert serialize(gen.build_messages()) == gen.JSONL.read_bytes()
    assert corpus_hash(gen.build_messages()) == json.loads(gen.MANIFEST.read_text())["sha256"]


def test_parse_body_examples():
    assert parse_body("ETA 1432 GATE G12") == ("5Z", {"eta": "1432", "gate": "G12"})
    assert parse_body("ETA 2561 GATE G12") == (None, {})


def test_load_fails_when_file_is_not_canonically_serialized(tmp_path):
    msgs = [json.loads(l) for l in gen.JSONL.read_text().splitlines()]
    raw = ("\n".join(json.dumps(m, indent=None) for m in msgs) + "\n").encode("ascii")
    assert raw != serialize(msgs)               # spaces after separators, unsorted keys
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_bytes(raw)
    import hashlib
    manifest = json.loads(gen.MANIFEST.read_text())
    manifest["sha256"] = hashlib.sha256(raw).hexdigest()   # hash matches the bytes
    man = tmp_path / "m.json"
    man.write_text(json.dumps(manifest))
    with pytest.raises(CorpusRejection) as exc:
        load_corpus(jsonl, man)
    assert exc.value.violations[0].code == "P3"