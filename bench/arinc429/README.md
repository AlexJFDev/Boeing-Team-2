# ARINC 429 word generator

`generator.py` makes synthetic ARINC 429 words from a seed, each with a ground-truth
record of its fields and the value it encodes. It covers WBS 3.1 (issue #56). The
B-family tasks (3.3 to 3.6) are built from its output, and the 3.2 reference decoder
will be checked against it.

The ARINC 429 specification is paid, so nothing here is quoted from it. The
conventions below come from free secondary sources, listed under [Sources](#sources).

## Conventions

### Bit numbering

Bits are numbered 1 to 32 in **transmission order**: bit 1 is sent first and bit 32
last. Written as a 32-bit integer (`word_hex`), bit *n* has weight 2^(*n*−1), so bit 1
is the least significant bit and bit 32 the most significant. `wire_bits` gives the
same word as a string in transmission order, bit 1 first.

| Bits | Field |
| --- | --- |
| 1–8 | Label |
| 9–10 | SDI (source/destination identifier); bit 10 is the high bit |
| 11–29 | Data (bit 11 is the least significant) |
| 30–31 | SSM (sign/status matrix) |
| 32 | Parity: **odd**, so every valid word has an odd number of 1 bits |

### Labels are octal and sent most significant bit first

A label is written as three octal digits, such as `203`. Its most significant bit goes
in bit 1 and is sent first, while the rest of the word is sent least significant bit
first. So the label octet (bits 1–8, the low byte of `word_hex`) is the label's
numeric value **bit-reversed**:

| Label (octal) | Numeric value | Bits 1–8 (low byte) | A naive forward read gives |
| --- | --- | --- | --- |
| 203 | 0x83 | 0xC1 | 301 |
| 204 | 0x84 | 0x21 | 041 |
| 213 | 0x8B | 0xD1 | 321 |
| 310 | 0xC8 | 0x13 | 023 |
| 012 | 0x0A | 0x50 (01010000) | 120 |

### SSM

Written as two bits, **bit 31 then bit 30** (`ssm_bits = "01"` means bit 31 is 0 and
bit 30 is 1). The same bits mean different things for BNR and BCD data, which is what
the B2 tasks test:

| Bit 31, bit 30 | BNR | BCD |
| --- | --- | --- |
| 00 | Failure Warning | Plus, North, East, Right, To, Above |
| 01 | No Computed Data | No Computed Data |
| 10 | Functional Test | Functional Test |
| 11 | Normal Operation | Minus, South, West, Left, From, Below |

### BNR (binary)

Two's complement. **Bit 29 is the sign** and bit 28 the most significant data bit.
A label profile gives the number of significant bits (`sig_bits`, counted down from
bit 28) and the `range`. Bit 28 is worth range/2 and the least significant bit
range / 2^sig_bits, which is the resolution. Unused bits below the least significant
bit are zero. Values run from −range to range − resolution.

Example: label 203 with 17 significant bits and range 131,072 ft has a resolution of
1 ft. 1,000 ft with SDI 0 and SSM 11 is word `0x601F40C1`; −1 ft is `0x7FFFF8C1`.

### BCD (binary-coded decimal)

The most significant digit takes 3 bits (27–29, so at most 7). Each later digit takes
4 bits: 23–26, 19–22, 15–18, 11–14. A profile with fewer than 5 digits leaves the low
bits zero. The data holds the magnitude; the sign comes from the SSM (00 plus,
11 minus). Words with SSM 01 or 10 carry no sign.

Example: ground speed 114 kt on label 012 (4 digits) has digits 0, 1, 1, 4, so the
data field reads `0x01140`.

## Label profiles are premises

Which parameter a label carries, and how it is scaled, is defined in ARINC 429
Attachment 2, which we don't have. `LABEL_PROFILES` therefore defines the scaling for
this benchmark, and every B-family task must state the profile it uses (encoding,
significant bits and range, or digits and resolution). That keeps each answer
determinate whatever Attachment 2 says. Each profile's `assignment` field records
whether its parameter name comes from a source or is illustrative.

| Label | Parameter | Encoding | Scaling | Generated values | Name source |
| --- | --- | --- | --- | --- | --- |
| 203 | Pressure altitude | BNR | 17 bits, range 131,072 ft (1 ft) | −1,000 to 45,000 ft | Wikipedia |
| 204 | Baro-corrected altitude | BNR | 17 bits, range 131,072 ft (1 ft) | −1,000 to 45,000 ft | Illustrative |
| 206 | Computed airspeed | BNR | 14 bits, range 1,024 kt (0.0625 kt) | 0 to 450 kt | Illustrative |
| 213 | Static air temperature | BNR | 11 bits, range 512 °C (0.25 °C) | −70 to 50 °C | Illustrative |
| 324 | Pitch angle | BNR | 14 bits, range 180° (≈0.011°) | −30° to 30° | Illustrative |
| 012 | Ground speed | BCD | 4 digits, 1 kt, unsigned | 0 to 650 kt | ALTA tutorial |
| 013 | Total air temperature | BCD | 3 digits, 1 °C, signed by SSM | −60 to 60 °C | Illustrative |

## What the generator does

`generate(n, seed, bad_parity_rate=0.1, status_rate=0.25)` returns `n` records.
For each word it:

1. Picks a profile, a value on that profile's resolution grid, and an SDI.
2. Picks an SSM. For a share `status_rate` of words it is a status (FW, NCD or FT for
   BNR; NCD or FT for BCD). Otherwise it is the valid-data state: 11 for BNR, or 00/11
   by the value's sign for BCD.
3. Encodes the word with odd parity.
4. For a share `bad_parity_rate` of words, flips one bit. Half of these flip the parity
   bit itself, leaving the data intact. The rest flip a random bit from 1 to 31, like a
   transmission error.

Everything comes from `random.Random(seed)` in a fixed order, so the same arguments
give byte-identical output, and a longer run starts with the shorter run's words.

It does not apply bus conformance rules: an NCD word still carries data, for example.
Those rules belong to the B3 tasks (3.5).

## Ground-truth record

| Field | Meaning |
| --- | --- |
| `word_id` | `a429-0001` onward |
| `word_hex` | The word as a 32-bit integer, bit *n* = 2^(*n*−1) |
| `wire_bits` | The 32 bits in transmission order, bit 1 first |
| `label_octal`, `label_octet_hex` | The label as read from bits 1–8, and the raw octet |
| `sdi`, `data_hex`, `ssm_bits`, `parity_bit` | The other raw fields as read from the word |
| `parity_ok` | Whether the word has odd parity |
| `expected_answer` | `decode`, or `reject` when parity fails |
| `profile` | The label profile the word was generated from |
| `decoded` | For a good word: parameter, units, encoding, `value`, `ssm_meaning`, `bcd_digits`, BCD `sign`. `null` for a bad word: a receiver must not use it. |
| `fault` | `null`, or the kind (`parity_bit_flipped` / `data_bit_flipped`), `flipped_bit`, `original_word_hex` and `original_decoded` |

The raw fields always describe the word as transmitted, even after a fault. If a fault
flips a label bit, `label_octal` shows the corrupted label, while `profile` shows the
one the word was generated from.

## Running it

From the repo root:

```bash
python -m bench.arinc429.generator --seed 42 -n 16 --table      # readable table
python -m bench.arinc429.generator --seed 42 -n 200 --out a.jsonl
python -m bench.arinc429.generator --seed 42 -n 200 --out b.jsonl
pytest tests/test_arinc429_generator.py
```

The two `--out` runs print the same sha256 and write identical files. `--seed` is
required, so a run can't silently depend on an unrecorded seed.

## Sources

- [Wikipedia, ARINC 429](https://en.wikipedia.org/wiki/ARINC_429): word layout,
  transmission order, label sent MSB first with the 213 → 0xD1 example, odd parity, BNR
  sign in bit 29 as two's complement, SSM tables for BNR, BCD and discrete (bit 31 then
  bit 30), label 203 as barometric altitude.
- [Logic Fruit, Designer's guide to ARINC 429](https://www.logic-fruit.com/designers-guide-arinc-429-standard-specification/):
  BNR MSB in bit 28; BCD uses 4 bits per digit with up to 5 digits, the most significant
  having 3 bits (27–29); odd parity; SSM tables.
- [ALTA Data Technologies, ARINC tutorial and reference](https://altadt.com/download/arinc-tutorial-and-reference):
  labels are octal with bits reversed from transmission order; label 012 (ground speed)
  as 01010000; odd parity.
- Task catalogue, B1: the 203/204/213/310 bit-reversal table.

All are secondary. The AIM ARINC 429 tutorial that the catalogue lists was not reachable
when this was written (it blocks automated access). The per-digit BCD bit positions below
the top digit (23–26, 19–22, 15–18, 11–14) follow from the 3-bit top digit and 4-bit digits
packed downward; no free source we found states them bit by bit.
