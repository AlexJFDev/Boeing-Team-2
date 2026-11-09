"""ARINC 429 reference decoder (WBS 3.2, issue #57).

Reads a 32-bit ARINC 429 word back into its fields and the engineering value it
carries. It is the ground-truth decoder for the B-family tasks (3.3 to 3.6) and the
independent check on the 3.1 word generator: the round-trip tests encode words with
``generator.py`` and require this module to read every one back exactly.

Independence. Nothing here reuses the generator's packing code. The generator packs
a value as a two's-complement count shifted into place; this decoder instead reads
each bit by its position and adds up the weight the spec gives that bit. Labels are
reversed by reading bits 1 to 8 one at a time. The only thing taken from the
generator is ``LABEL_PROFILES``, which is data, not code: the profiles are premises
of the benchmark (which label carries what, at what scaling), and every B task
states the profile it uses. A test checks that ``LABEL_PROFILES`` is the only name
this module imports from the generator.

Conventions are the ones in bench/arinc429/README.md: bit 1 is sent first and is
the least significant bit of the integer form; bits 1-8 label (sent MSB first, so
bit-reversed), 9-10 SDI, 11-29 data, 30-31 SSM (written bit 31 then bit 30), 32 odd
parity.

Every input gets one of four statuses, and the decoder never raises:

- ``decoded``: parity checks, the label has a profile, and the data field is a valid
  encoding for it. ``decoded`` holds the engineering value.
- ``rejected``: the word is 32 bits but a receiver must not use it, because parity
  fails or a BCD digit is not 0-9. ``decoded`` is ``None``; the raw fields are still
  given so a task can explain why.
- ``unknown_label``: parity checks but no profile covers the label. The raw fields
  are given, but there is no scaling to turn the data into a value.
- ``invalid_input``: not a 32-bit word at all (wrong type, not hex, too long, a bit
  string that is not 32 characters of 0 and 1, negative).

The ``decoded`` dict has the same keys and meanings as the generator's ground truth
(parameter, units, encoding, value, ssm_meaning, bcd_digits, sign), and ``fields``
has the same keys as the generator's raw fields, so the two compare directly.

It does not apply bus conformance rules (for example, that an NCD word's data should
be ignored). Those belong to the B3 tasks (3.5).

Command-line demo:

    python -m bench.arinc429.decoder 0x601F40C1 0x7FFFF8C1
    python -m bench.arinc429.decoder --wire 10000011000000101111100000000110
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Literal, Mapping

from bench.arinc429.generator import LABEL_PROFILES

DECODER_VERSION = "1.0.0"

Status = Literal["decoded", "rejected", "unknown_label", "invalid_input"]

WORD_BITS = 32
HEX_DIGITS = frozenset(string.hexdigits)

# SSM tables as published (Wikipedia, ARINC 429; Logic Fruit), keyed by bit 31 then
# bit 30. Kept here rather than imported so a typo in the generator's copy is caught.
SSM_MEANINGS: dict[str, dict[str, str]] = {
    "BNR": {"00": "Failure Warning", "01": "No Computed Data",
            "10": "Functional Test", "11": "Normal Operation"},
    "BCD": {"00": "Plus, North, East, Right, To, Above", "01": "No Computed Data",
            "10": "Functional Test", "11": "Minus, South, West, Left, From, Below"},
}

# BCD digit slots, most significant first, as (highest bit, lowest bit). The top
# digit has 3 bits; the rest have 4.
BCD_SLOTS: tuple[tuple[int, int], ...] = ((29, 27), (26, 23), (22, 19), (18, 15), (14, 11))

# Profiles keyed by the label as three octal digits. Each is a plain dict, in the
# same shape a task states it in (LabelProfile.describe()).
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    p.label_octal: p.describe() for p in LABEL_PROFILES
}


@dataclass(frozen=True)
class DecodeResult:
    """What the decoder reads from one input."""

    status: Status
    reason: str | None
    fields: dict[str, Any] | None  # raw fields; None only for invalid_input
    decoded: dict[str, Any] | None  # engineering value; only when status is decoded
    profile: dict[str, Any] | None  # the profile used, if the label has one
    decoder: str = f"bench.arinc429.decoder=={DECODER_VERSION}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reading bits
# ---------------------------------------------------------------------------


def _bit(word: int, n: int) -> int:
    """Bit n (1-32) of the word, bit 1 being the least significant."""
    return (word >> (n - 1)) & 1


def _bits_msb_first(word: int, high: int, low: int) -> int:
    """The bits from ``high`` down to ``low`` read as a number, ``high`` the MSB."""
    value = 0
    for n in range(high, low - 1, -1):
        value = value * 2 + _bit(word, n)
    return value


def read_label(word: int) -> str:
    """The label as three octal digits.

    The label's most significant bit is sent first, in bit 1, so reading bits 1 to 8
    in order gives the label MSB first.
    """
    value = 0
    for n in range(1, 9):
        value = value * 2 + _bit(word, n)
    return f"{value:03o}"


def read_fields(word: int) -> dict[str, Any]:
    """The raw fields of a 32-bit word, good or bad, in the generator's format."""
    ones = sum(_bit(word, n) for n in range(1, WORD_BITS + 1))
    return {
        "word_hex": f"0x{word:08X}",
        "wire_bits": "".join(str(_bit(word, n)) for n in range(1, WORD_BITS + 1)),
        "label_octal": read_label(word),
        "label_octet_hex": f"0x{_bits_msb_first(word, 8, 1):02X}",
        "sdi": _bits_msb_first(word, 10, 9),
        "data_hex": f"0x{_bits_msb_first(word, 29, 11):05X}",
        "ssm_bits": f"{_bit(word, 31)}{_bit(word, 30)}",
        "parity_bit": _bit(word, 32),
        "parity_ok": ones % 2 == 1,
    }


# ---------------------------------------------------------------------------
# Data field
# ---------------------------------------------------------------------------


def _number(value: Fraction) -> float:
    """A Fraction as a float, with -0 written as 0."""
    out = float(value)
    return 0.0 if out == 0 else out


def decode_bnr(word: int, profile: Mapping[str, Any]) -> float:
    """The BNR value: bit 29 is worth -range, bit 28 range/2, bit 27 range/4, ...

    That is two's complement read bit by bit. Only the profile's significant bits
    (counted down from bit 28) carry weight; bits below them are padding.
    """
    full_range = Fraction(str(profile["range"]))
    value = -full_range * _bit(word, 29)
    for k in range(1, int(profile["sig_bits"]) + 1):
        value += _bit(word, 29 - k) * full_range / 2**k
    return _number(value)


def decode_bcd(word: int, profile: Mapping[str, Any]) -> tuple[list[int], int] | str:
    """The BCD digits (most significant first) and the magnitude in counts.

    Returns a reason string instead when a digit is not 0-9.
    """
    digits = []
    for position, (high, low) in enumerate(BCD_SLOTS[: int(profile["digits"])]):
        digit = _bits_msb_first(word, high, low)
        if digit > 9:
            return f"BCD digit {position + 1} (bits {low}-{high}) is {digit}, not 0-9"
        digits.append(digit)
    counts = 0
    for digit in digits:
        counts = counts * 10 + digit
    return digits, counts


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _invalid(reason: str) -> DecodeResult:
    return DecodeResult("invalid_input", reason, None, None, None)


def _to_word(word: Any) -> int | str:
    """The input as a 32-bit integer, or a reason it is not one."""
    if isinstance(word, bool):  # bool is an int subclass; True is not a word
        return "input is a bool, not a word"
    if isinstance(word, int):
        if not 0 <= word < 2**WORD_BITS:
            return f"integer {word} is not a 32-bit unsigned word"
        return word
    if isinstance(word, str):
        text = word.strip()
        if text[:2].lower() == "0x":
            text = text[2:]
        text = text.replace("_", "")
        if not text:
            return "empty string"
        if not set(text) <= HEX_DIGITS:
            return f"{word!r} is not hexadecimal"
        if len(text) > 8:
            return f"{word!r} has {len(text)} hex digits; a word has at most 8"
        return int(text, 16)
    return f"input is {type(word).__name__}, not an int or hex string"


def decode(word: Any, profiles: Mapping[str, Mapping[str, Any]] | None = None) -> DecodeResult:
    """Decode one word, given as an int or a hex string such as ``"0x601F40C1"``.

    ``profiles`` maps a label (three octal digits) to its profile; it defaults to the
    benchmark's ``LABEL_PROFILES``. A task that states its own profile can pass it.
    """
    value = _to_word(word)
    if isinstance(value, str):
        return _invalid(value)
    return _decode_int(value, DEFAULT_PROFILES if profiles is None else profiles)


def decode_wire(bits: Any, profiles: Mapping[str, Mapping[str, Any]] | None = None) -> DecodeResult:
    """Decode a word given as its 32 bits in transmission order, bit 1 first."""
    if not isinstance(bits, str):
        return _invalid(f"input is {type(bits).__name__}, not a bit string")
    text = bits.replace(" ", "").replace("_", "")
    if len(text) != WORD_BITS or not set(text) <= {"0", "1"}:
        return _invalid(f"a word on the wire is 32 characters of 0 and 1, got {bits!r}")
    word = sum(int(b) << n for n, b in enumerate(text))  # bit 1 is the LSB
    return _decode_int(word, DEFAULT_PROFILES if profiles is None else profiles)


def _decode_int(word: int, profiles: Mapping[str, Mapping[str, Any]]) -> DecodeResult:
    fields = read_fields(word)
    profile = profiles.get(fields["label_octal"])
    profile_out = dict(profile) if profile is not None else None

    # A receiver checks parity before anything else; a bad word is never used.
    if not fields["parity_ok"]:
        return DecodeResult("rejected", "parity check failed: the word has an even "
                            "number of 1 bits", fields, None, profile_out)
    if profile is None:
        return DecodeResult("unknown_label", f"no profile for label {fields['label_octal']}",
                            fields, None, None)

    encoding = profile["encoding"]
    ssm_bits = fields["ssm_bits"]
    meaning = SSM_MEANINGS[encoding][ssm_bits]
    if encoding == "BNR":
        decoded = {"parameter": profile["parameter"], "units": profile["units"],
                   "encoding": "BNR", "value": decode_bnr(word, profile),
                   "ssm_meaning": meaning, "bcd_digits": None, "sign": None}
        return DecodeResult("decoded", None, fields, decoded, profile_out)

    if encoding != "BCD":
        return DecodeResult("unknown_label", f"profile for label {fields['label_octal']} "
                            f"has unsupported encoding {encoding!r}", fields, None, profile_out)
    bcd = decode_bcd(word, profile)
    if isinstance(bcd, str):
        return DecodeResult("rejected", bcd, fields, None, profile_out)
    digits, counts = bcd
    magnitude = Fraction(counts) * Fraction(str(profile["resolution"]))
    # The BCD sign is carried by the SSM: 00 plus, 11 minus. NCD and FT carry none.
    sign = {"00": "plus", "11": "minus"}.get(ssm_bits)
    value = -magnitude if sign == "minus" else magnitude
    decoded = {"parameter": profile["parameter"], "units": profile["units"],
               "encoding": "BCD", "value": _number(value), "ssm_meaning": meaning,
               "bcd_digits": digits, "sign": sign}
    return DecodeResult("decoded", None, fields, decoded, profile_out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Decode ARINC 429 words.")
    parser.add_argument("words", nargs="+", help="words as hex (0x601F40C1), or bit strings with --wire")
    parser.add_argument("--wire", action="store_true",
                        help="read each word as 32 bits in transmission order, bit 1 first")
    args = parser.parse_args(argv)
    for word in args.words:
        result = decode_wire(word) if args.wire else decode(word)
        sys.stdout.write(json.dumps({"input": word, **result.to_dict()}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
