"""Seeded ARINC 429 32-bit word generator with ground truth (WBS 3.1, issue #56).

Every word comes with a ground-truth record of its fields: label, SDI, data, SSM,
parity, and the engineering value the data encodes. The same seed always gives
byte-identical output. Words can be generated with bad parity, and are then
flagged in their ground truth (``parity_ok`` false, ``expected_answer`` "reject").

Conventions (see bench/arinc429/README.md for sources):

Bit numbering
    Bits are numbered 1 to 32 in transmission order: bit 1 is sent first, bit 32
    last. Written as a 32-bit integer (``word_hex``), bit n has weight 2**(n-1),
    so bit 1 is the least significant bit. ``wire_bits`` lists the bits in
    transmission order, bit 1 first.

Fields
    bits 1-8    label            bits 9-10   SDI
    bits 11-29  data             bits 30-31  SSM
    bit 32      parity (odd: the word always has an odd number of 1 bits)

Label (octal, sent most significant bit first)
    A label is written as three octal digits, for example 203. Its most
    significant bit is sent first, in bit 1, so the label octet is bit-reversed
    relative to the label's numeric value: label 213 (0x8B) occupies bits 1-8 as
    0xD1, the low byte of ``word_hex``.

SSM
    Written as two bits, bit 31 then bit 30 (``ssm_bits`` "01" means bit 31 = 0,
    bit 30 = 1). Its meaning depends on the encoding; see SSM_MEANINGS.

BNR (binary)
    Two's complement. Bit 29 is the sign, bit 28 the most significant data bit.
    A profile gives the number of significant bits ``sig_bits`` (counted from bit
    28 down) and the ``range``: bit 28 is worth range/2 and the least significant
    bit range / 2**sig_bits (the resolution). Unused low bits are zero. Values run
    from -range to range - resolution.

BCD (binary-coded decimal)
    The most significant digit takes 3 bits (27-29); each later digit takes 4 bits
    (23-26, 19-22, 15-18, 11-14), most significant first. Unused low bits are zero.
    The value is unsigned in the data; the sign comes from the SSM (00 plus,
    11 minus).

Label profiles
    Which parameter, encoding and scaling a label carries is defined in ARINC 429
    Attachment 2, which is paid. LABEL_PROFILES are therefore premises of this
    benchmark: each B-family task states the profile it uses. ``assignment``
    records where a label's parameter name comes from, or says it is illustrative.

Command-line demo:

    python -m bench.arinc429.generator --seed 42 -n 10 --table
    python -m bench.arinc429.generator --seed 42 -n 200 --out words.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

GENERATOR_VERSION = "1.0.0"

Encoding = Literal["BNR", "BCD"]

SSM_MEANINGS: dict[str, dict[str, str]] = {
    # Keyed by bit 31 then bit 30.
    "BNR": {"00": "Failure Warning", "01": "No Computed Data",
            "10": "Functional Test", "11": "Normal Operation"},
    "BCD": {"00": "Plus, North, East, Right, To, Above", "01": "No Computed Data",
            "10": "Functional Test", "11": "Minus, South, West, Left, From, Below"},
}

DEFAULT_BAD_PARITY_RATE = 0.1
# Share of words whose SSM is not the "data valid" state (FW, NCD or FT for BNR;
# NCD or FT for BCD). The rest carry valid data.
DEFAULT_STATUS_RATE = 0.25

BCD_DIGIT_LSB = (27, 23, 19, 15, 11)  # lowest bit of each digit, most significant digit first
BCD_DIGIT_WIDTH = (3, 4, 4, 4, 4)


@dataclass(frozen=True)
class LabelProfile:
    """How one label's data field is encoded. A premise of the benchmark."""

    label: int  # numeric value of the octal label, e.g. 0o203
    parameter: str
    units: str
    encoding: Encoding
    assignment: str  # where the parameter name for this label comes from
    sig_bits: int | None = None  # BNR
    range: float | None = None  # BNR: bit 28 is worth range/2
    digits: int | None = None  # BCD
    resolution: float | None = None  # BCD: value of the least significant digit
    signed: bool = True  # BCD: whether the SSM may carry a minus sign
    gen_min: float = 0.0  # values the generator draws from
    gen_max: float = 0.0

    @property
    def label_octal(self) -> str:
        return f"{self.label:03o}"

    @property
    def lsb(self) -> float:
        """Resolution: the value of one count of the encoded data."""
        if self.encoding == "BNR":
            assert self.range is not None and self.sig_bits is not None
            return self.range / 2**self.sig_bits
        assert self.resolution is not None
        return self.resolution

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label_octal": self.label_octal, "parameter": self.parameter,
                               "units": self.units, "encoding": self.encoding,
                               "assignment": self.assignment}
        if self.encoding == "BNR":
            out |= {"sig_bits": self.sig_bits, "range": self.range, "resolution": self.lsb}
        else:
            out |= {"digits": self.digits, "resolution": self.resolution, "signed": self.signed}
        return out


WIKIPEDIA = "label named in Wikipedia, ARINC 429; scaling is a benchmark premise"
ALTA = "label named in the ALTA ARINC tutorial; scaling is a benchmark premise"
ILLUSTRATIVE = "illustrative; not checked against ARINC 429 Attachment 2"

LABEL_PROFILES: tuple[LabelProfile, ...] = (
    LabelProfile(0o203, "Pressure altitude", "ft", "BNR", WIKIPEDIA,
                 sig_bits=17, range=131072, gen_min=-1000, gen_max=45000),
    LabelProfile(0o204, "Baro-corrected altitude", "ft", "BNR", ILLUSTRATIVE,
                 sig_bits=17, range=131072, gen_min=-1000, gen_max=45000),
    LabelProfile(0o206, "Computed airspeed", "kt", "BNR", ILLUSTRATIVE,
                 sig_bits=14, range=1024, gen_min=0, gen_max=450),
    LabelProfile(0o213, "Static air temperature", "degC", "BNR", ILLUSTRATIVE,
                 sig_bits=11, range=512, gen_min=-70, gen_max=50),
    LabelProfile(0o324, "Pitch angle", "deg", "BNR", ILLUSTRATIVE,
                 sig_bits=14, range=180, gen_min=-30, gen_max=30),
    LabelProfile(0o012, "Ground speed", "kt", "BCD", ALTA,
                 digits=4, resolution=1, signed=False, gen_min=0, gen_max=650),
    LabelProfile(0o013, "Total air temperature", "degC", "BCD", ILLUSTRATIVE,
                 digits=3, resolution=1, signed=True, gen_min=-60, gen_max=60),
)
PROFILES_BY_LABEL = {p.label: p for p in LABEL_PROFILES}


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------


def reverse8(value: int) -> int:
    """Reverse the 8 bits of an octet: the label's numeric value <-> bits 1-8."""
    return int(f"{value:08b}"[::-1], 2)


def label_octet(label: int) -> int:
    """The value of bits 1-8 (the low byte of the word) for a numeric label."""
    if not 0 <= label <= 0o377:
        raise ValueError(f"label {label:o} is not an 8-bit octal label (000-377)")
    return reverse8(label)


def bit(word: int, n: int) -> int:
    """Bit n (1-32) of a word."""
    return (word >> (n - 1)) & 1


def ones(word: int) -> int:
    return bin(word & 0xFFFFFFFF).count("1")


def parity_ok(word: int) -> bool:
    """True when the word has odd parity (an odd number of 1 bits)."""
    return ones(word) % 2 == 1


def with_odd_parity(word31: int) -> int:
    """Set bit 32 so the whole word has an odd number of 1 bits."""
    word31 &= 0x7FFFFFFF
    return word31 | ((ones(word31) % 2 == 0) << 31)


def wire_bits(word: int) -> str:
    """The 32 bits in transmission order, bit 1 first."""
    return "".join(str(bit(word, n)) for n in range(1, 33))


def assemble(label: int, sdi: int, data: int, ssm_bits: str) -> int:
    """A word with odd parity from its fields. ``data`` is bits 11-29, bit 11 lowest."""
    if not 0 <= sdi <= 3:
        raise ValueError(f"SDI must be 0-3, got {sdi}")
    if not 0 <= data < 2**19:
        raise ValueError(f"data must fit in 19 bits, got {data}")
    if ssm_bits not in ("00", "01", "10", "11"):
        raise ValueError(f"SSM must be two bits, got {ssm_bits!r}")
    ssm = int(ssm_bits, 2)  # bit 31 is the high bit of the pair
    return with_odd_parity(label_octet(label) | sdi << 8 | data << 10 | ssm << 29)


# ---------------------------------------------------------------------------
# Data encodings
# ---------------------------------------------------------------------------


def encode_bnr(value: float, profile: LabelProfile) -> int:
    """Data field (bits 11-29) for a BNR value. The value must be on the resolution grid."""
    assert profile.sig_bits is not None and profile.range is not None
    counts = value / profile.lsb
    if counts != int(counts):
        raise ValueError(f"{value} is not a multiple of the resolution {profile.lsb}")
    counts = int(counts)
    if not -(2**profile.sig_bits) <= counts < 2**profile.sig_bits:
        raise ValueError(f"{value} is outside -{profile.range}..{profile.range} (exclusive)")
    field = counts & (2 ** (profile.sig_bits + 1) - 1)  # two's complement incl. sign bit
    return field << (18 - profile.sig_bits)  # sign lands on bit 29, MSB on bit 28


def encode_bcd(value: float, profile: LabelProfile) -> tuple[int, list[int]]:
    """Data field and digits (most significant first) for a BCD magnitude."""
    assert profile.digits is not None and profile.resolution is not None
    counts = round(abs(value) / profile.resolution)
    if abs(counts * profile.resolution - abs(value)) > 1e-9:
        raise ValueError(f"{value} is not a multiple of the resolution {profile.resolution}")
    digits = [int(d) for d in f"{counts:0{profile.digits}d}"]
    if len(digits) != profile.digits or digits[0] > 7:
        raise ValueError(f"{value} does not fit in {profile.digits} BCD digits")
    data = 0
    for d, lsb_bit in zip(digits, BCD_DIGIT_LSB):
        data |= d << (lsb_bit - 11)
    return data, digits


def _grid_values(profile: LabelProfile) -> tuple[int, int]:
    """The range of counts the generator draws from."""
    lo = int(-(-profile.gen_min // profile.lsb))  # ceil
    hi = int(profile.gen_max // profile.lsb)
    return lo, hi


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decoded:
    """What a correct receiver reads from a word with good parity."""

    parameter: str
    units: str
    encoding: Encoding
    value: float
    ssm_meaning: str
    bcd_digits: list[int] | None = None
    sign: str | None = None  # BCD: "plus"/"minus" from the SSM, else None


def fields_of(word: int) -> dict[str, Any]:
    """The raw fields of a word as transmitted. Used for every word, good or bad."""
    return {
        "word_hex": f"0x{word:08X}",
        "wire_bits": wire_bits(word),
        "label_octal": f"{reverse8(word & 0xFF):03o}",
        "label_octet_hex": f"0x{word & 0xFF:02X}",
        "sdi": (word >> 8) & 0b11,
        "data_hex": f"0x{(word >> 10) & 0x7FFFF:05X}",
        "ssm_bits": f"{bit(word, 31)}{bit(word, 30)}",
        "parity_bit": bit(word, 32),
        "parity_ok": parity_ok(word),
    }


def _decoded(profile: LabelProfile, value: float, ssm_bits: str,
             digits: list[int] | None) -> Decoded:
    sign = None
    if profile.encoding == "BCD" and ssm_bits in ("00", "11"):
        sign = "plus" if ssm_bits == "00" else "minus"
    return Decoded(profile.parameter, profile.units, profile.encoding, value,
                   SSM_MEANINGS[profile.encoding][ssm_bits], digits, sign)


def make_word(profile: LabelProfile, value: float, sdi: int, ssm_bits: str) -> tuple[int, Decoded]:
    """Encode one word with good parity and return it with its decoded ground truth."""
    if profile.encoding == "BNR":
        data, digits = encode_bnr(value, profile), None
    else:
        if value < 0 and ssm_bits != "11":
            raise ValueError("a negative BCD value needs SSM 11 (minus)")
        if value < 0 and not profile.signed:
            raise ValueError(f"label {profile.label_octal} is unsigned")
        data, digits = encode_bcd(value, profile)
        if ssm_bits not in ("00", "11"):
            value = abs(value)  # no sign is conveyed with NCD or FT
    return assemble(profile.label, sdi, data, ssm_bits), _decoded(profile, value, ssm_bits, digits)


def flip_bit(word: int, n: int) -> int:
    """The word with bit n (1-32) inverted. One flip always breaks odd parity."""
    if not 1 <= n <= 32:
        raise ValueError(f"bit must be 1-32, got {n}")
    return word ^ (1 << (n - 1))


def record(word_id: str, word: int, profile: LabelProfile, decoded: Decoded,
           fault: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ground-truth record for a generated word."""
    fields = fields_of(word)
    good = fields["parity_ok"]
    return {
        "word_id": word_id,
        **fields,
        "expected_answer": "decode" if good else "reject",
        "profile": profile.describe(),
        # A correct receiver uses a word only when its parity checks.
        "decoded": asdict(decoded) if good else None,
        "fault": fault,
    }


def _choose_ssm(rng: random.Random, profile: LabelProfile, negative: bool,
                status_rate: float) -> str:
    if rng.random() < status_rate:
        statuses = ["00", "01", "10"] if profile.encoding == "BNR" else ["01", "10"]
        return rng.choice(statuses)
    if profile.encoding == "BNR":
        return "11"
    return "11" if negative else "00"


def generate(n: int, seed: int, bad_parity_rate: float = DEFAULT_BAD_PARITY_RATE,
             status_rate: float = DEFAULT_STATUS_RATE,
             profiles: tuple[LabelProfile, ...] = LABEL_PROFILES) -> list[dict[str, Any]]:
    """Generate n words with ground truth. Same arguments, same output.

    A share ``bad_parity_rate`` of words get one bit flipped after encoding:
    half the time the parity bit itself (data intact), otherwise a random bit
    1-31 (a transmission error). Either way parity fails, the word is flagged
    ``parity_ok`` false with ``expected_answer`` "reject", and the fault records
    the bit and the word as it was before the flip.
    """
    if n < 0:
        raise ValueError("n must not be negative")
    if not 0 <= bad_parity_rate <= 1 or not 0 <= status_rate <= 1:
        raise ValueError("rates must be between 0 and 1")
    rng = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        profile = rng.choice(profiles)
        lo, hi = _grid_values(profile)
        value = rng.randint(lo, hi) * profile.lsb
        sdi = rng.randint(0, 3)
        ssm_bits = _choose_ssm(rng, profile, value < 0, status_rate)
        if profile.encoding == "BCD" and value < 0 and ssm_bits != "11":
            value = abs(value)  # a status word carries no sign; keep the magnitude
        word, decoded = make_word(profile, value, sdi, ssm_bits)
        fault = None
        if rng.random() < bad_parity_rate:
            flipped = 32 if rng.random() < 0.5 else rng.randint(1, 31)
            fault = {
                "kind": "parity_bit_flipped" if flipped == 32 else "data_bit_flipped",
                "flipped_bit": flipped,
                "original_word_hex": f"0x{word:08X}",
                "original_decoded": asdict(decoded),
            }
            word = flip_bit(word, flipped)
        out.append(record(f"a429-{i:04d}", word, profile, decoded, fault))
    return out


def to_jsonl(records: list[dict[str, Any]]) -> str:
    """Canonical JSON Lines: sorted keys, one record per line, trailing newline."""
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _table(records: list[dict[str, Any]]) -> str:
    rows = [f"{'word_id':<10} {'word_hex':<10} {'label':>5} {'sdi':>3} {'ssm':>3} "
            f"{'parity':<7} {'value':>20}  meaning / fault"]
    for r in records:
        d = r["decoded"]
        if d:
            number = d["value"]
            value = f"{int(number) if number == int(number) else number} {d['units']}"
            note = d["ssm_meaning"]
        else:
            value = "-"
            f = r["fault"]
            note = f"REJECT: {f['kind']} (bit {f['flipped_bit']}), was {f['original_word_hex']}"
        rows.append(f"{r['word_id']:<10} {r['word_hex']:<10} {r['label_octal']:>5} {r['sdi']:>3} "
                    f"{r['ssm_bits']:>3} {'ok' if r['parity_ok'] else 'BAD':<7} {value:>20}  {note}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate ARINC 429 words with ground truth.")
    parser.add_argument("--seed", type=int, required=True, help="random seed (required)")
    parser.add_argument("-n", type=int, default=20, help="number of words (default 20)")
    parser.add_argument("--bad-parity-rate", type=float, default=DEFAULT_BAD_PARITY_RATE,
                        help=f"share of words with a flipped bit (default {DEFAULT_BAD_PARITY_RATE})")
    parser.add_argument("--status-rate", type=float, default=DEFAULT_STATUS_RATE,
                        help=f"share of words with a non-normal SSM (default {DEFAULT_STATUS_RATE})")
    parser.add_argument("--out", type=Path, help="write JSON Lines here instead of stdout")
    parser.add_argument("--table", action="store_true", help="print a readable table instead of JSON")
    args = parser.parse_args(argv)
    records = generate(args.n, args.seed, args.bad_parity_rate, args.status_rate)
    text = to_jsonl(records)
    if args.table:
        print(_table(records))
    elif args.out:
        args.out.write_bytes(text.encode("utf-8"))
    else:
        sys.stdout.write(text)
    if args.table or args.out:
        bad = sum(not r["parity_ok"] for r in records)
        print(f"\n{len(records)} words, {bad} with bad parity, seed {args.seed}, "
              f"generator {GENERATOR_VERSION}, sha256 {sha256_text(text)}"
              + (f", written to {args.out}" if args.out else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
