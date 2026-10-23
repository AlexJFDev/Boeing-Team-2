"""Checks for the ARINC 429 word generator (WBS 3.1, issue #56).

Acceptance criteria (issue #56):
- When the generator is run twice with the same seed, the output is identical and
  every word decodes to its recorded fields.
- When a word has bad parity, it is flagged as such in its ground truth.

Backlog criteria for 3.1:
- When a word is generated, its label bits are reversed per the spec and odd
  parity is correct.
- When the parity-flip option is set, the word is labeled invalid in the ground truth.

Words are decoded here from ``wire_bits`` one bit at a time, written separately
from the generator's packing code, so the generator is not checked against itself.
The 3.2 reference decoder will be a second, fuller check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.arinc429 import generator as gen

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 42, 2026, 987654321)
N = 400
SETS = {seed: gen.generate(N, seed) for seed in SEEDS}
ALL = [r for records in SETS.values() for r in records]

# SSM tables as published (Wikipedia, ARINC 429), keyed by bit 31 then bit 30.
PUBLISHED_SSM = {
    "BNR": {"00": "Failure Warning", "01": "No Computed Data",
            "10": "Functional Test", "11": "Normal Operation"},
    "BCD": {"00": "Plus, North, East, Right, To, Above", "01": "No Computed Data",
            "10": "Functional Test", "11": "Minus, South, West, Left, From, Below"},
}


# --- Independent decoding from the transmitted bit string ---------------------


def wire(record: dict[str, Any]) -> list[int]:
    """Bits indexed 1-32 in transmission order (index 0 unused)."""
    bits = record["wire_bits"]
    assert len(bits) == 32 and set(bits) <= {"0", "1"}
    return [0] + [int(b) for b in bits]


def read_raw(b: list[int]) -> dict[str, Any]:
    # Label: bit 1 is the label's most significant bit.
    label = 0
    for n in range(1, 9):
        label = label * 2 + b[n]
    data = sum(b[n] << (n - 11) for n in range(11, 30))  # bit 11 is the data LSB
    return {
        "label_octal": f"{label:03o}",
        "sdi": b[10] * 2 + b[9],
        "data_hex": f"0x{data:05X}",
        "ssm_bits": f"{b[31]}{b[30]}",
        "parity_bit": b[32],
        "parity_ok": sum(b[1:]) % 2 == 1,
    }


def read_value(b: list[int], profile: dict[str, Any]) -> tuple[float, list[int] | None]:
    if profile["encoding"] == "BNR":
        sig = profile["sig_bits"]
        # Two's complement: bit 29 is worth -range, bit 28 range/2, and so on down.
        value = -b[29] * profile["range"]
        weight = profile["range"]
        for n in range(28, 28 - sig, -1):
            weight /= 2
            value += b[n] * weight
        assert all(b[n] == 0 for n in range(11, 29 - sig)), "unused BNR bits must be zero"
        return value, None
    widths = {27: 3, 23: 4, 19: 4, 15: 4, 11: 4}
    digits = []
    for lsb in list(widths)[: profile["digits"]]:
        digits.append(sum(b[lsb + k] << k for k in range(widths[lsb])))
    used_low = list(widths)[profile["digits"] - 1]
    assert all(b[n] == 0 for n in range(11, used_low)), "unused BCD bits must be zero"
    magnitude = int("".join(map(str, digits))) * profile["resolution"]
    return (-magnitude if f"{b[31]}{b[30]}" == "11" else magnitude), digits


# --- Criterion 1: same seed, same output -------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_gives_identical_output(seed: int) -> None:
    first = gen.to_jsonl(gen.generate(N, seed))
    second = gen.to_jsonl(gen.generate(N, seed))
    assert first == second
    assert gen.to_jsonl(SETS[seed]) == first


def test_two_command_line_runs_are_byte_identical(tmp_path: Path) -> None:
    outputs = []
    for name in ("a.jsonl", "b.jsonl"):
        path = tmp_path / name
        subprocess.run([sys.executable, "-m", "bench.arinc429.generator", "--seed", "42",
                        "-n", "200", "--out", str(path)], cwd=REPO_ROOT, check=True,
                       capture_output=True)
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1] and len(outputs[0].splitlines()) == 200


def test_different_seeds_give_different_output() -> None:
    assert gen.to_jsonl(SETS[1]) != gen.to_jsonl(SETS[2026])


def test_a_longer_run_extends_a_shorter_one() -> None:
    assert gen.generate(500, 42)[:N] == SETS[42]


def test_seed_is_required_on_the_command_line() -> None:
    out = subprocess.run([sys.executable, "-m", "bench.arinc429.generator", "-n", "3"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode != 0 and "--seed" in out.stderr


# --- Criterion 1: every word decodes to its recorded fields -------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_every_word_decodes_to_its_recorded_raw_fields(seed: int) -> None:
    for r in SETS[seed]:
        b = wire(r)
        assert read_raw(b) == {k: r[k] for k in read_raw(b)}, r["word_id"]
        assert int(r["word_hex"], 16) == sum(b[n] << (n - 1) for n in range(1, 33)), r["word_id"]
        assert r["label_octet_hex"] == f"0x{int(r['word_hex'], 16) & 0xFF:02X}"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_good_word_decodes_to_its_recorded_value(seed: int) -> None:
    for r in SETS[seed]:
        if not r["parity_ok"]:
            continue
        b = wire(r)
        d = r["decoded"]
        assert r["label_octal"] == r["profile"]["label_octal"], r["word_id"]
        value, digits = read_value(b, r["profile"])
        if d["encoding"] == "BCD" and r["ssm_bits"] in ("01", "10"):
            value = abs(value)  # NCD/FT carry no sign
        assert d["value"] == value, r["word_id"]
        assert d["bcd_digits"] == digits, r["word_id"]
        assert d["ssm_meaning"] == PUBLISHED_SSM[d["encoding"]][r["ssm_bits"]], r["word_id"]
        assert (d["parameter"], d["units"], d["encoding"]) == (
            r["profile"]["parameter"], r["profile"]["units"], r["profile"]["encoding"])


def test_worked_bnr_example() -> None:
    """Label 203, SDI 0, SSM 11, 1000 ft with 17 significant bits (1 ft per count), by hand:

    data: 1000 counts in bits 28..12 (sign 0 in bit 29, bit 11 unused) = 1000 << 1 = 0x007D0
    word: 0xC1 | 0x007D0 << 10 | 0b11 << 29 = 0x601F40C1, which has 11 one-bits, so parity 0.
    """
    profile = gen.PROFILES_BY_LABEL[0o203]
    word, decoded = gen.make_word(profile, 1000, 0, "11")
    assert word == 0x601F40C1
    assert decoded.value == 1000 and decoded.ssm_meaning == "Normal Operation"


def test_worked_negative_bnr_example() -> None:
    """-1 ft: two's complement all ones in bits 29..12, bit 11 zero -> 0x7FFFF8C1, 23 one-bits."""
    word, decoded = gen.make_word(gen.PROFILES_BY_LABEL[0o203], -1, 0, "11")
    assert word == 0x7FFFF8C1 and decoded.value == -1
    assert read_value(wire(gen.fields_of(word)), gen.PROFILES_BY_LABEL[0o203].describe())[0] == -1


def test_worked_bcd_example() -> None:
    """Ground speed 114 kt, 4 digits: 0, 1, 1, 4 in bits 27-29, 23-26, 19-22, 15-18 -> data 0x01140."""
    word, decoded = gen.make_word(gen.PROFILES_BY_LABEL[0o012], 114, 0, "00")
    assert gen.fields_of(word)["data_hex"] == "0x01140"
    assert decoded.bcd_digits == [0, 1, 1, 4] and decoded.sign == "plus"


@pytest.mark.parametrize("profile", [p for p in gen.LABEL_PROFILES if p.encoding == "BNR"],
                         ids=lambda p: p.label_octal)
def test_bnr_extremes_round_trip(profile: gen.LabelProfile) -> None:
    assert profile.range is not None
    for value in (0, profile.lsb, -profile.lsb, profile.range - profile.lsb, -profile.range):
        word, decoded = gen.make_word(profile, value, 1, "11")
        assert read_value(wire(gen.fields_of(word)), profile.describe())[0] == value
    with pytest.raises(ValueError, match="outside"):
        gen.make_word(profile, profile.range, 1, "11")
    with pytest.raises(ValueError, match="resolution"):
        gen.make_word(profile, profile.lsb / 2, 1, "11")


def test_bcd_limits() -> None:
    speed = gen.PROFILES_BY_LABEL[0o012]
    gen.make_word(speed, 7999, 0, "00")
    with pytest.raises(ValueError, match="does not fit"):
        gen.make_word(speed, 8000, 0, "00")  # the first digit has only 3 bits
    with pytest.raises(ValueError, match="unsigned"):
        gen.make_word(speed, -5, 0, "11")
    with pytest.raises(ValueError, match="SSM 11"):
        gen.make_word(gen.PROFILES_BY_LABEL[0o013], -5, 0, "00")


# --- Backlog criterion 1: label bits reversed, odd parity --------------------


@pytest.mark.parametrize(("label", "octet"), [
    (0o203, 0xC1), (0o204, 0x21), (0o213, 0xD1), (0o310, 0x13),  # task catalogue, B1 table
    (0o012, 0b01010000),  # ALTA ARINC tutorial: label 012 (ground speed) as 01010000
])
def test_label_octet_is_bit_reversed(label: int, octet: int) -> None:
    assert gen.label_octet(label) == octet
    # A naive forward read gives a different but valid label (the B1 trap).
    if label != gen.reverse8(label):
        assert f"{octet:03o}" != f"{label:03o}"


def test_label_reversal_covers_every_label() -> None:
    for label in range(0o400):
        assert gen.reverse8(gen.label_octet(label)) == label
        word = gen.assemble(label, 0, 0, "11")
        assert read_raw(wire(gen.fields_of(word)))["label_octal"] == f"{label:03o}"
    with pytest.raises(ValueError):
        gen.label_octet(0o400)


def test_every_unfaulted_word_has_odd_parity() -> None:
    for r in ALL:
        if r["fault"] is None:
            assert r["parity_ok"] and r["wire_bits"].count("1") % 2 == 1, r["word_id"]


# --- Criterion 2 / backlog criterion 2: bad parity is flagged -----------------


def test_bad_parity_words_are_flagged() -> None:
    bad = [r for r in ALL if not r["parity_ok"]]
    assert bad, "expected some bad-parity words at the default rate"
    for r in bad:
        assert r["wire_bits"].count("1") % 2 == 0, r["word_id"]
        assert r["expected_answer"] == "reject" and r["decoded"] is None
        fault = r["fault"]
        original = int(fault["original_word_hex"], 16)
        assert gen.parity_ok(original)
        assert original ^ int(r["word_hex"], 16) == 1 << (fault["flipped_bit"] - 1)
        assert fault["kind"] == ("parity_bit_flipped" if fault["flipped_bit"] == 32
                                 else "data_bit_flipped")
        assert fault["original_decoded"]["value"] is not None


def test_good_words_are_never_flagged() -> None:
    for r in ALL:
        if r["parity_ok"]:
            assert r["expected_answer"] == "decode" and r["fault"] is None and r["decoded"]


def test_both_fault_kinds_occur() -> None:
    kinds = {r["fault"]["kind"] for r in ALL if r["fault"]}
    assert kinds == {"parity_bit_flipped", "data_bit_flipped"}


def test_parity_flip_option_marks_every_word_invalid() -> None:
    records = gen.generate(100, 7, bad_parity_rate=1.0)
    assert all(not r["parity_ok"] and r["expected_answer"] == "reject" for r in records)
    assert all(r["parity_ok"] for r in gen.generate(100, 7, bad_parity_rate=0.0))


def test_parity_bit_flip_leaves_the_fields_intact() -> None:
    word, _ = gen.make_word(gen.PROFILES_BY_LABEL[0o203], 1000, 2, "11")
    flipped = gen.fields_of(gen.flip_bit(word, 32))
    good = gen.fields_of(word)
    assert not flipped["parity_ok"]
    for key in ("label_octal", "sdi", "data_hex", "ssm_bits"):
        assert flipped[key] == good[key]


def test_any_single_bit_flip_breaks_parity() -> None:
    word, _ = gen.make_word(gen.PROFILES_BY_LABEL[0o213], -12.5, 3, "11")
    assert all(not gen.parity_ok(gen.flip_bit(word, n)) for n in range(1, 33))


# --- The mix of words ----------------------------------------------------------


def test_every_profile_ssm_and_sdi_appears() -> None:
    assert {r["profile"]["label_octal"] for r in ALL} == {p.label_octal for p in gen.LABEL_PROFILES}
    assert {r["sdi"] for r in ALL} == {0, 1, 2, 3}
    for encoding in ("BNR", "BCD"):
        assert {r["ssm_bits"] for r in ALL if r["profile"]["encoding"] == encoding} == {
            "00", "01", "10", "11"}


def test_negative_values_are_generated_for_signed_parameters() -> None:
    negatives = {r["profile"]["label_octal"] for r in ALL
                 if r["decoded"] and r["decoded"]["value"] < 0}
    assert {"203", "213", "324", "013"} <= negatives


def test_unsigned_bcd_never_carries_a_minus_sign() -> None:
    for r in ALL:
        if r["profile"]["label_octal"] == "012":
            assert r["ssm_bits"] != "11"


def test_bcd_ssm_agrees_with_the_sign() -> None:
    for r in ALL:
        d = r["decoded"]
        if d and d["encoding"] == "BCD":
            if r["ssm_bits"] == "11":
                assert d["value"] <= 0 and d["sign"] == "minus"
            elif r["ssm_bits"] == "00":
                assert d["value"] >= 0 and d["sign"] == "plus"
            else:
                assert d["value"] >= 0 and d["sign"] is None


def test_status_rate_zero_gives_only_valid_data_states() -> None:
    for r in gen.generate(200, 3, status_rate=0.0):
        valid = {"BNR": {"11"}, "BCD": {"00", "11"}}[r["profile"]["encoding"]]
        assert r["ssm_bits"] in valid


def test_profiles_generate_within_their_encodable_range() -> None:
    for p in gen.LABEL_PROFILES:
        if p.encoding == "BNR":
            assert p.range is not None
            assert -p.range <= p.gen_min and p.gen_max < p.range
        else:
            assert p.digits is not None and p.resolution is not None
            assert p.gen_max / p.resolution < 8 * 10 ** (p.digits - 1)


def test_ssm_tables_match_the_published_tables() -> None:
    assert gen.SSM_MEANINGS == PUBLISHED_SSM


@pytest.mark.parametrize("bad", [{"n": -1}, {"bad_parity_rate": 1.5}, {"status_rate": -0.1}])
def test_bad_arguments_are_rejected(bad: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = {"n": 5, "seed": 1} | bad
    with pytest.raises(ValueError):
        gen.generate(**kwargs)


def test_records_are_plain_json() -> None:
    line = gen.to_jsonl(SETS[42][:1])
    assert json.loads(line) == SETS[42][0]


def test_table_demo_runs() -> None:
    out = subprocess.run([sys.executable, "-m", "bench.arinc429.generator", "--seed", "42",
                          "-n", "16", "--table"], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True)
    assert "REJECT" in out.stdout and "sha256" in out.stderr
