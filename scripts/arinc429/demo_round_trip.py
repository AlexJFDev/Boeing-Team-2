"""Demo for task 3.2 (issue #57): ARINC 429 encode -> decode round trip.

Encodes synthetic words with the 3.1 generator, decodes each with the reference
decoder, re-encodes what was decoded, and checks the word comes back bit for bit.
Then corrupts a good word and shows the decoder rejecting it.

Usage (from the repo root):
    python scripts/arinc429/demo_round_trip.py --seed 42
    python scripts/arinc429/demo_round_trip.py --seed 42 -n 5000 --show 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench.arinc429 import decoder as dec  # noqa: E402
from bench.arinc429 import generator as gen  # noqa: E402


def _value(decoded: dict | None) -> str:
    if decoded is None:
        return "-"
    number = decoded["value"]
    shown = int(number) if number == int(number) else round(number, 4)
    return f"{shown} {decoded['units']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="generator seed (required)")
    parser.add_argument("-n", type=int, default=200, help="words to round-trip (default 200)")
    parser.add_argument("--show", type=int, default=8, help="rows to print (default 8)")
    args = parser.parse_args(argv)

    # Generator faults are off here so every word should round-trip;
    # corruption is shown separately below, and the tests cover the generator's faults.
    records = gen.generate(args.n, args.seed, bad_parity_rate=0.0)
    print(f"Generator {gen.GENERATOR_VERSION}, decoder {dec.DECODER_VERSION}, "
          f"seed {args.seed}, {len(records)} words\n")
    print(f"{'word_id':<10} {'encoded':<10} {'label':>5} {'ssm':>3} {'decoded value':>18} "
          f"{'re-encoded':<10}  result")

    mismatches = 0
    for i, record in enumerate(records):
        word = int(record["word_hex"], 16)
        result = dec.decode(word)
        ok = result.status == "decoded" and result.decoded == record["decoded"]
        again = None
        if ok and result.fields is not None and result.decoded is not None:
            profile = gen.PROFILES_BY_LABEL[int(result.fields["label_octal"], 8)]
            again, _ = gen.make_word(profile, result.decoded["value"], result.fields["sdi"],
                                     result.fields["ssm_bits"])
            ok = again == word
        mismatches += not ok
        if i < args.show or not ok:
            shown_again = f"0x{again:08X}" if again is not None else "-"
            print(f"{record['word_id']:<10} {record['word_hex']:<10} {record['label_octal']:>5} "
                  f"{record['ssm_bits']:>3} {_value(result.decoded):>18} {shown_again:<10}  "
                  f"{'identical' if ok else 'MISMATCH ' + str(result.reason)}")
    print(f"\nround trip: {mismatches} mismatches in {len(records)} words\n")

    # Negative case: corrupt a good word and show it is rejected.
    original = records[0]
    word = int(original["word_hex"], 16)
    print(f"Corrupting {original['word_id']} {original['word_hex']} "
          f"({_value(original['decoded'])}):")
    for flipped in (32, 12, 1):
        bad = gen.flip_bit(word, flipped)
        result = dec.decode(bad)
        verdict = "REJECTED" if result.status == "rejected" else f"NOT REJECTED ({result.status})"
        print(f"  bit {flipped:>2} flipped -> 0x{bad:08X}: {verdict}, value {_value(result.decoded)}"
              f" ({result.reason})")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
