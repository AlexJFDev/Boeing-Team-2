"""B1 task family: ARINC 429 word decode and the label bit order (WBS 3.3, issue #58).

Each B1 task shows the model one synthetic 32-bit ARINC 429 word and asks for its
label, SDI, SSM bits and encoded value, or a rejection when parity fails. Ground
truth is the reference decoder (bench/arinc429/decoder.py, WBS 3.2). Words come from
the seeded generator (bench/arinc429/generator.py, WBS 3.1).

The discriminator (task catalogue, B1): the label is sent most significant bit
first, so its octet is bit-reversed relative to the octal label. A forward read of
the octet gives a different label that still looks valid, for example 213 read as
321. Every sample records that wrong answer as ``naive_label`` so a run can count
how often a model makes this mistake.

Which words are used (``select_words``), all from ``generate(N_WORDS, SEED)``:

- for each label profile, the first good-parity word for each SSM value, up to
  ``PER_PROFILE`` per label (7 profiles, 21 words, BNR and BCD, valid and status SSMs)
- the first ``PER_FAULT`` words of each generator fault kind: parity bit flipped
  (data intact) and a data, label or SDI bit flipped (a transmission error). The
  answer is a rejection.
- ``UNKNOWN_LABEL_WORDS`` good-parity words on label 310, which has no profile.
  Label 310 is the fourth label in the catalogue's bit-reversal table. The label,
  SDI and SSM can be read, but the value cannot be determined without a profile.

B1 answer fields (keys in alphabetical order, canonical JSON):

    good word with a profile   label, sdi, ssm, value
    good word, no profile      label, sdi, ssm, value = "cannot_be_determined"
    bad parity                 answer = "reject", reason = "parity_check_failed"

``label`` is three octal digits, ``sdi`` an integer 0-3, ``ssm`` two characters
(bit 31 then bit 30). ``value`` is the number in the data field under the label's
profile: signed for BNR, and the magnitude for BCD. A BCD word's sign is carried by
the SSM, and SSM meaning is B2 (3.4), so B1 does not ask for it.

The label map with every profile is in the family preamble (``render_preamble``,
data/preambles/arinc429_v1.0.0.txt), not in the task, because a per-task profile
would name the label and give the answer away.

Validation (``validate_samples``) decodes every sample's word again with the reference
decoder and fails if the target differs. It also checks that the word is still the
one the generator makes for that word ID. A hand edit to the task file, generator or
decoder drift, or a profile change therefore stops the run instead of silently
changing the answer key.

Build the task file (output is deterministic; the tests check the committed file
matches it):

    python scripts/arinc429/build_b1_tasks.py

Check the committed task file against the decoder and print one sample:

    python -m bench.arinc429.b1
"""

from __future__ import annotations

import argparse
import functools
import json
import random
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample

from bench.arinc429 import decoder
from bench.arinc429 import generator as gen

B1_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = REPO_ROOT / "data" / "tasks" / f"b1_arinc429_word_decode_v{B1_VERSION}.jsonl"
PREAMBLE = "arinc429_v1.0.0"
PREAMBLE_FILE = REPO_ROOT / "data" / "preambles" / f"{PREAMBLE}.txt"
SOURCE = "spec-synthesized"

SEED = 4290
N_WORDS = 1000
PER_PROFILE = 3
PER_FAULT = 2
UNKNOWN_LABEL = 0o310
UNKNOWN_LABEL_WORDS = 2

CANNOT_BE_DETERMINED = "cannot_be_determined"
REJECT = {"answer": "reject", "reason": "parity_check_failed"}

TASK_TEXT = """\
Decode the ARINC 429 word below.

Word: {word}

Answer with a single JSON object and nothing else: keys in alphabetical order, no \
spaces or line breaks, strings in double quotes.

If the word passes the parity check, report these keys:
- "label": the label as three octal digits, as a string (for example "123")
- "sdi": the SDI as an integer 0-3
- "ssm": the SSM as a two-character string, bit 31 then bit 30 (for example "01")
- "value": the value encoded in the data field, using the label's profile from the \
label map. For BNR, the signed value; for BCD, the magnitude in the data field. Write \
it as a plain decimal number in the label's units, exact and not rounded, with no \
exponent, and as an integer when it is a whole number. If the label is not in the \
label map, use the string "cannot_be_determined".

If the parity check fails, answer exactly {{"answer":"reject","reason":"parity_check_failed"}}"""


class B1ValidationError(ValueError):
    """A B1 sample's target does not match the reference decoder."""


# ---------------------------------------------------------------------------
# Preamble
# ---------------------------------------------------------------------------


def _number(value: float) -> str:
    """A number as the preamble and targets write it: exact, integer when whole."""
    exact = Decimal(value)
    if exact == exact.to_integral_value():
        return str(int(exact))
    return format(exact.normalize(), "f")


def _profile_line(profile: gen.LabelProfile) -> str:
    if profile.encoding == "BNR":
        assert profile.sig_bits is not None
        lowest = 29 - profile.sig_bits
        return (f"- {profile.label_octal}: {profile.parameter.lower()}, {profile.units}. "
                f"BNR, {profile.sig_bits} data bits (bits 28 down to {lowest}) below the "
                f"sign bit, resolution {_number(profile.lsb)} {profile.units}.")
    assert profile.digits is not None
    return (f"- {profile.label_octal}: {profile.parameter.lower()}, {profile.units}. "
            f"BCD, {profile.digits} digits, resolution {_number(profile.lsb)} {profile.units}.")


def render_preamble(profiles: Iterable[gen.LabelProfile] = gen.LABEL_PROFILES) -> str:
    """The text of the ARINC 429 family preamble. The committed file must equal it."""
    label_map = "\n".join(_profile_line(p) for p in profiles)
    return f"""\
Operational premises for ARINC 429 tasks. They apply to the task below unless the task itself says otherwise.

1. Each word is one 32-bit ARINC 429 word, written as 8 hexadecimal digits after 0x. Bits are numbered 1 to 32 in transmission order, and bit n has the value 2^(n-1) in the hexadecimal number, so bit 1 is the least significant bit.
2. Bits 1-8 are the label, bits 9-10 the SDI (bit 10 is the more significant), bits 11-29 the data field, bits 30-31 the SSM, and bit 32 the parity bit. Labels are written as three octal digits, following ARINC 429.
3. Parity is odd: a valid word has an odd number of 1 bits. If the parity check fails, the word is corrupt: reject it and do not report fields from it.
4. BNR data is two's complement. Bit 29 is the sign bit and bit 28 the most significant data bit. A label's profile gives how many data bits are used below the sign bit; bits below those are zero. The least significant data bit used is worth the resolution.
5. BCD data is binary-coded decimal, most significant digit first. The first digit takes bits 27-29 and each later digit 4 bits: 23-26, 19-22, 15-18, then 11-14. A profile with fewer than 5 digits leaves the lowest bits zero. The data field holds the magnitude; the sign is not part of the data field.
6. Decode the data field whatever the SSM says.
7. Label map for this bus. A label not listed here has no known profile:
{label_map}
8. If the information provided is not enough to determine an answer, say that it cannot be determined rather than guessing.
9. All words are synthetic."""


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def naive_label(word: int) -> str:
    """The label a forward read of the octet gives: the catalogue's wrong answer."""
    return f"{word & 0xFF:03o}"


def reference_answer(word: int) -> dict[str, Any]:
    """The B1 answer for a word, from the reference decoder."""
    result = decoder.decode(word)
    if result.status == "rejected":
        if not (result.reason or "").startswith("parity"):
            raise B1ValidationError(f"word 0x{word:08X} is rejected for a reason B1 does not "
                                    f"ask about: {result.reason}")
        return dict(REJECT)
    if result.status not in ("decoded", "unknown_label") or result.fields is None:
        raise B1ValidationError(f"word 0x{word:08X} is not usable for B1: {result.reason}")
    fields = result.fields
    answer: dict[str, Any] = {"label": fields["label_octal"], "sdi": fields["sdi"],
                              "ssm": fields["ssm_bits"]}
    if result.status == "unknown_label":
        answer["value"] = CANNOT_BE_DETERMINED
        return answer
    assert result.decoded is not None
    value = result.decoded["value"]
    if result.decoded["encoding"] == "BCD":
        value = abs(value)  # B1 asks for the magnitude; the sign is the SSM's (B2)
    exact = Decimal(value)
    if exact == exact.to_integral_value():
        answer["value"] = int(exact)
    else:
        if Decimal(repr(value)) != exact:  # JSON must print the exact value
            raise B1ValidationError(f"value {exact} of word 0x{word:08X} has no exact short form")
        answer["value"] = value
    return answer


def canonical(answer: dict[str, Any]) -> str:
    """Canonical JSON, the form the target takes."""
    return json.dumps(answer, sort_keys=True, separators=(",", ":"))


def _case(record: dict[str, Any]) -> str:
    if record.get("fault"):
        return "parity_bit_fault" if record["fault"]["kind"] == "parity_bit_flipped" else "data_bit_fault"
    if record["profile"] is None:
        return "unknown_label"
    return record["profile"]["encoding"].lower()


# ---------------------------------------------------------------------------
# Word selection
# ---------------------------------------------------------------------------


@functools.cache
def generated_words() -> tuple[dict[str, Any], ...]:
    return tuple(gen.generate(N_WORDS, SEED))


def unknown_label_words() -> list[dict[str, Any]]:
    """Good-parity words on a label with no profile, from their own seeded stream."""
    if UNKNOWN_LABEL in gen.PROFILES_BY_LABEL:
        raise B1ValidationError(f"label {UNKNOWN_LABEL:o} has a profile; pick another")
    rng = random.Random(f"b1-unknown-{SEED}")
    out = []
    for i in range(1, UNKNOWN_LABEL_WORDS + 1):
        word = gen.assemble(UNKNOWN_LABEL, rng.randint(0, 3), rng.randint(0, 2**19 - 1),
                            rng.choice(("00", "01", "10", "11")))
        out.append({"word_id": f"unknown-{i:02d}", **gen.fields_of(word),
                    "profile": None, "fault": None})
    return out


def select_words(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The generated words B1 uses, in generation order (see the module docstring)."""
    chosen = []
    seen_ssm: dict[str, set[str]] = {}
    faults: dict[str, int] = {}
    for record in records:
        fault = record["fault"]
        if fault is not None:
            if faults.get(fault["kind"], 0) < PER_FAULT:
                faults[fault["kind"]] = faults.get(fault["kind"], 0) + 1
                chosen.append(record)
            continue
        label = record["profile"]["label_octal"]
        ssms = seen_ssm.setdefault(label, set())
        if len(ssms) < PER_PROFILE and record["ssm_bits"] not in ssms:
            ssms.add(record["ssm_bits"])
            chosen.append(record)
    return chosen


def build_records() -> list[dict[str, Any]]:
    """Task records in the loader's format (bench/loader.py), one per word."""
    words = select_words(generated_words()) + unknown_label_words()
    records = []
    for record in words:
        word = int(record["word_hex"], 16)
        answer = reference_answer(word)
        # Cross-check: the decoder must agree with the generator's own ground truth.
        if record["profile"] is not None or record["fault"] is not None:
            truth = record["decoded"]
            if (answer == REJECT) != (truth is None):
                raise B1ValidationError(f"{record['word_id']}: decoder and generator disagree "
                                        "on whether the word is rejected")
            if truth is not None:
                expected = abs(truth["value"]) if truth["encoding"] == "BCD" else truth["value"]
                if answer["value"] != expected:
                    raise B1ValidationError(f"{record['word_id']}: decoder value {answer['value']} "
                                            f"!= generator value {expected}")
        records.append({
            "id": f"b1-{record['word_id']}",
            "family": "B",
            "preamble": PREAMBLE,
            "source": SOURCE,
            "input": TASK_TEXT.format(word=record["word_hex"]),
            "target": canonical(answer),
            "metadata": {
                "subfamily": "B1",
                "b1_version": B1_VERSION,
                "case": _case(record),
                "word": record["word_hex"],
                "word_id": record["word_id"],
                "naive_label": naive_label(word),
                "expected_status": "rejected" if answer == REJECT else "decoded",
                "generator_seed": SEED,
                "generator_version": gen.GENERATOR_VERSION,
                "decoder": f"bench.arinc429.decoder=={decoder.DECODER_VERSION}",
            },
        })
    return records


def write_task_file(path: Path = TASK_FILE) -> Path:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in build_records()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def write_preamble(path: Path = PREAMBLE_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((render_preamble() + "\n").encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@functools.cache
def _word_by_id() -> dict[str, str]:
    words = list(generated_words()) + unknown_label_words()
    return {r["word_id"]: r["word_hex"] for r in words}


def sample_problems(sample: Sample) -> list[str]:
    """Why a sample fails validation; empty when it passes."""
    meta = sample.metadata or {}
    word_hex = meta.get("word")
    if not isinstance(word_hex, str) or not word_hex:
        return ["metadata has no word"]
    problems = []
    word_id = meta.get("word_id")
    if word_id is not None and _word_by_id().get(word_id) != word_hex:
        problems.append(f"word {word_hex} is not word {word_id!r} from generator seed {SEED}")
    if f"Word: {word_hex}" not in str(sample.input):
        problems.append(f"prompt does not contain the word {word_hex}")
    if meta.get("preamble") not in (None, PREAMBLE):
        problems.append(f"preamble {meta['preamble']!r} is not {PREAMBLE!r}")
    if not isinstance(sample.target, str):
        return [*problems, "target must be a single JSON string"]
    try:
        expected = reference_answer(int(word_hex, 16))
    except (B1ValidationError, ValueError) as exc:
        return [*problems, str(exc)]
    if sample.target != canonical(expected):
        problems.append(f"target {sample.target} does not match the reference decoder "
                        f"{canonical(expected)}")
    status = "rejected" if expected == REJECT else "decoded"
    if meta.get("expected_status") not in (None, status):
        problems.append(f"expected_status {meta['expected_status']!r} but the decoder gives {status!r}")
    return problems


def validate_samples(samples: Iterable[Sample]) -> None:
    """Raise B1ValidationError listing every sample that does not match the decoder."""
    failures = [f"{s.id}: {'; '.join(p)}" for s in samples if (p := sample_problems(s))]
    if failures:
        raise B1ValidationError(
            f"B1 task set failed validation ({len(failures)} sample(s)):\n" + "\n".join(failures)
        )


def main() -> None:
    from bench.loader import load_tasks

    parser = argparse.ArgumentParser(description="Validate the B1 task file and show a sample.")
    parser.add_argument("tasks", nargs="?", default=str(TASK_FILE), help="B1 task file")
    parser.add_argument("--show", default=None, help="sample ID to print (default: the first)")
    args = parser.parse_args()
    dataset = load_tasks(args.tasks)
    try:
        validate_samples(dataset)
    except B1ValidationError as exc:
        raise SystemExit(f"error: {exc}") from None
    sample = next((s for s in dataset if s.id == args.show), None) if args.show else dataset[0]
    if sample is None:
        raise SystemExit(f"error: no sample {args.show!r}")
    print(f"{len(dataset)} samples match the reference decoder "
          f"(bench.arinc429.decoder=={decoder.DECODER_VERSION}).\n")
    meta = sample.metadata or {}
    print(f"===== {sample.id} ({meta.get('case')})")
    print(f"----- word:   {meta.get('word')}  (a forward read of the label gives {meta.get('naive_label')})")
    print(f"----- target: {sample.target}")
    print("----- prompt:")
    print(sample.input)


if __name__ == "__main__":
    main()
