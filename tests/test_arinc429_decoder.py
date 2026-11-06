"""Round-trip tests for the ARINC 429 reference decoder (WBS 3.2, issue #57).

Demo (issue #57): encode synthetic ARINC 429 words, decode them back with the
reference decoder to show a lossless round trip, and show a corrupted word rejected.

Acceptance criteria checked here:
- When a word is encoded by the 3.1 generator and decoded, the decoder recovers the
  exact fields and value in its ground truth, and re-encoding what was decoded gives
  back the same 32-bit word (lossless round trip).
- When a word is corrupted (any single bit flipped, or the generator's own faults),
  the decoder rejects it and gives no value (negative case).
- When the input is not a 32-bit word, the decoder says so instead of raising.

The decoder is written separately from the generator's packing code; a test below
checks it imports only the label profiles (premises) from the generator.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from bench.arinc429 import decoder as dec
from bench.arinc429 import generator as gen

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 42, 2026, 987654321)
N = 400
ALL = [r for seed in SEEDS for r in gen.generate(N, seed)]
GOOD = [r for r in ALL if r["parity_ok"]]
BAD = [r for r in ALL if not r["parity_ok"]]
FIELD_KEYS = ("word_hex", "wire_bits", "label_octal", "label_octet_hex", "sdi",
              "data_hex", "ssm_bits", "parity_bit", "parity_ok")


def reencode(result: dec.DecodeResult) -> int:
    """Encode what the decoder read, using the generator's encoder."""
    assert result.fields is not None and result.decoded is not None
    profile = gen.PROFILES_BY_LABEL[int(result.fields["label_octal"], 8)]
    word, _ = gen.make_word(profile, result.decoded["value"], result.fields["sdi"],
                            result.fields["ssm_bits"])
    return word


# --- Lossless round trip -------------------------------------------------------


def test_sample_covers_every_profile_ssm_and_fault_kind() -> None:
    """The generated sample is broad enough for the round trip to mean something."""
    assert {r["label_octal"] for r in GOOD} >= {p.label_octal for p in gen.LABEL_PROFILES}
    assert {(r["profile"]["encoding"], r["ssm_bits"]) for r in GOOD} == {
        ("BNR", "00"), ("BNR", "01"), ("BNR", "10"), ("BNR", "11"),
        ("BCD", "00"), ("BCD", "01"), ("BCD", "10"), ("BCD", "11")}
    assert {r["fault"]["kind"] for r in BAD} == {"parity_bit_flipped", "data_bit_flipped"}
    assert any(r["decoded"]["value"] < 0 for r in GOOD)


@pytest.mark.parametrize("record", GOOD, ids=lambda r: r["word_hex"])
def test_good_word_decodes_to_its_ground_truth(record: dict[str, Any]) -> None:
    result = dec.decode(record["word_hex"])
    assert result.status == "decoded", result.reason
    assert result.fields == {k: record[k] for k in FIELD_KEYS}
    assert result.decoded == record["decoded"]


@pytest.mark.parametrize("record", GOOD, ids=lambda r: r["word_hex"])
def test_reencoding_the_decode_gives_the_same_word(record: dict[str, Any]) -> None:
    result = dec.decode(int(record["word_hex"], 16))
    assert f"0x{reencode(result):08X}" == record["word_hex"]


@pytest.mark.parametrize("profile", [p for p in gen.LABEL_PROFILES if p.encoding == "BNR"],
                         ids=lambda p: p.label_octal)
def test_bnr_round_trip_over_the_whole_range(profile: gen.LabelProfile) -> None:
    """Every count for small profiles; both ends, zero and a stride for large ones."""
    assert profile.sig_bits is not None
    top = 2**profile.sig_bits
    stride = 1 if top <= 2**14 else 37
    counts = set(range(-top, top, stride)) | {-top, -top + 1, -1, 0, 1, top - 2, top - 1}
    for count in sorted(counts):
        value = count * profile.lsb
        for ssm in ("00", "01", "10", "11"):
            word, truth = gen.make_word(profile, value, count % 4, ssm)
            result = dec.decode(word)
            assert result.status == "decoded"
            assert result.decoded is not None and result.decoded["value"] == truth.value
            assert reencode(result) == word


@pytest.mark.parametrize("profile", [p for p in gen.LABEL_PROFILES if p.encoding == "BCD"],
                         ids=lambda p: p.label_octal)
def test_bcd_round_trip_over_every_value(profile: gen.LabelProfile) -> None:
    assert profile.digits is not None
    largest = 8 * 10 ** (profile.digits - 1) - 1  # top digit has 3 bits, so at most 7
    for count in range(largest + 1):
        signs = ("00", "11") if profile.signed else ("00",)
        for ssm in signs:
            value = -count if ssm == "11" else count
            word, truth = gen.make_word(profile, value, count % 4, ssm)
            result = dec.decode(word)
            assert result.status == "decoded"
            assert result.decoded == asdict(truth)
            assert reencode(result) == word


# --- Corrupted words are rejected (negative case) ------------------------------


@pytest.mark.parametrize("record", BAD, ids=lambda r: r["word_hex"])
def test_generator_faults_are_rejected(record: dict[str, Any]) -> None:
    result = dec.decode(record["word_hex"])
    assert record["expected_answer"] == "reject"
    assert result.status == "rejected"
    assert result.decoded is None
    assert "parity" in (result.reason or "")
    # The raw fields still describe the word as transmitted.
    assert result.fields == {k: record[k] for k in FIELD_KEYS}


@pytest.mark.parametrize("record", BAD, ids=lambda r: r["word_hex"])
def test_undoing_the_fault_recovers_the_original(record: dict[str, Any]) -> None:
    fault = record["fault"]
    repaired = gen.flip_bit(int(record["word_hex"], 16), fault["flipped_bit"])
    assert f"0x{repaired:08X}" == fault["original_word_hex"]
    result = dec.decode(repaired)
    assert result.status == "decoded"
    assert result.decoded == fault["original_decoded"]


@pytest.mark.parametrize("record", GOOD[:60], ids=lambda r: r["word_hex"])
def test_every_single_bit_flip_is_rejected(record: dict[str, Any]) -> None:
    word = int(record["word_hex"], 16)
    for n in range(1, 33):
        result = dec.decode(gen.flip_bit(word, n))
        assert result.status == "rejected", f"bit {n}"
        assert result.decoded is None


def test_invalid_bcd_digit_with_good_parity_is_rejected() -> None:
    # Ground speed (label 012, 4 digits): put 0xA (10) in the second digit, bits 23-26.
    word = gen.assemble(0o012, 0, 0b1010 << (23 - 11), "00")
    result = dec.decode(word)
    assert result.fields is not None and result.fields["parity_ok"]
    assert result.status == "rejected"
    assert result.decoded is None
    assert "BCD digit 2" in (result.reason or "")


def test_label_with_no_profile_gives_fields_but_no_value() -> None:
    word = gen.assemble(0o310, 2, 0x12345, "11")
    result = dec.decode(word)
    assert result.status == "unknown_label"
    assert result.decoded is None
    assert result.fields is not None and result.fields["label_octal"] == "310"
    assert result.fields["sdi"] == 2 and result.fields["data_hex"] == "0x12345"


# --- Worked examples from the README and task catalogue ------------------------


@pytest.mark.parametrize(("word", "value"), [("0x601F40C1", 1000.0), ("0x7FFFF8C1", -1.0)])
def test_readme_bnr_examples(word: str, value: float) -> None:
    result = dec.decode(word)
    assert result.status == "decoded"
    assert result.decoded is not None and result.decoded["value"] == value
    assert result.decoded["ssm_meaning"] == "Normal Operation"


@pytest.mark.parametrize(("octet", "label"),
                         [(0xC1, "203"), (0x21, "204"), (0xD1, "213"), (0x13, "310"), (0x50, "012")])
def test_label_bit_reversal_table(octet: int, label: str) -> None:
    assert dec.read_label(octet) == label


def test_readme_bcd_example() -> None:
    profile = gen.PROFILES_BY_LABEL[0o012]
    word, _ = gen.make_word(profile, 114, 0, "00")
    result = dec.decode(word)
    assert result.fields is not None and result.fields["data_hex"] == "0x01140"
    assert result.decoded is not None
    assert result.decoded["bcd_digits"] == [0, 1, 1, 4] and result.decoded["value"] == 114


def test_bcd_minus_sign_comes_from_the_ssm() -> None:
    profile = gen.PROFILES_BY_LABEL[0o013]
    word, _ = gen.make_word(profile, -45, 1, "11")
    result = dec.decode(word)
    assert result.decoded is not None
    assert result.decoded["value"] == -45 and result.decoded["sign"] == "minus"


def test_a_task_can_supply_its_own_profile() -> None:
    profile = {"label_octal": "310", "parameter": "Test value", "units": "u",
               "encoding": "BNR", "sig_bits": 8, "range": 256}
    word = gen.assemble(0o310, 0, 0b0_00000101 << (18 - 8), "11")  # +5 counts of 1
    result = dec.decode(word, profiles={"310": profile})
    assert result.status == "decoded"
    assert result.decoded is not None and result.decoded["value"] == 5.0


# --- Input forms and bad input -------------------------------------------------


@pytest.mark.parametrize("record", GOOD[:50] + BAD[:20], ids=lambda r: r["word_hex"])
def test_wire_bits_decode_the_same_as_hex(record: dict[str, Any]) -> None:
    assert dec.decode_wire(record["wire_bits"]) == dec.decode(record["word_hex"])


@pytest.mark.parametrize("form", ["0x601F40C1", "601f40c1", "0x601F_40C1", " 601F40C1 ", 0x601F40C1])
def test_accepted_input_forms(form: Any) -> None:
    assert dec.decode(form).status == "decoded"


@pytest.mark.parametrize("bad", ["", "0x", "zz", "0x1601F40C1", -1, 2**32, None, True, 1.5,
                                 b"\x60\x1f\x40\xc1"])
def test_invalid_input_never_raises(bad: Any) -> None:
    result = dec.decode(bad)
    assert result.status == "invalid_input"
    assert result.reason and result.fields is None and result.decoded is None


@pytest.mark.parametrize("bad", ["1" * 31, "1" * 33, "2" * 32, "", None])
def test_invalid_wire_bits_never_raise(bad: Any) -> None:
    assert dec.decode_wire(bad).status == "invalid_input"


# --- Independence and command line ---------------------------------------------


def test_decoder_imports_only_profiles_from_the_generator() -> None:
    """The decoder must not reuse the generator's packing code, or a shared bug
    would pass the round trip."""
    tree = ast.parse(Path(dec.__file__).read_text(encoding="utf-8"))
    from_generator = [alias.name for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom) and node.module
                      and node.module.endswith("generator") for alias in node.names]
    plain = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names if alias.name.endswith("generator")]
    assert from_generator == ["LABEL_PROFILES"] and not plain


def test_ssm_tables_match_the_generator() -> None:
    assert dec.SSM_MEANINGS == gen.SSM_MEANINGS


def test_command_line_decodes_and_rejects() -> None:
    out = subprocess.run([sys.executable, "-m", "bench.arinc429.decoder", "0x601F40C1", "0x601F40C0"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    first, second = (json.loads(line) for line in out.splitlines())
    assert first["status"] == "decoded" and first["decoded"]["value"] == 1000.0
    assert second["status"] == "rejected" and second["decoded"] is None


def test_round_trip_demo_script_runs() -> None:
    out = subprocess.run([sys.executable, "scripts/arinc429/demo_round_trip.py", "--seed", "42"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert "round trip: 0 mismatches" in out
    assert "REJECTED" in out
