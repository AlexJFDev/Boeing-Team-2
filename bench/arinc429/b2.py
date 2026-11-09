"""B2 task family: ARINC 429 sign/status matrix interpretation (WBS 3.4, issue #59).

Each B2 task shows the model two synthetic ARINC 429 words that carry the same SSM
bits, one BNR and one BCD. It asks for each word's SSM bits and what they mean. Ground
truth is the reference decoder (bench/arinc429/decoder.py, WBS 3.2). Words come from
the seeded generator (bench/arinc429/generator.py, WBS 3.1).

The discriminator (task catalogue, B2): the same two bits mean different things for
BNR and BCD data. In BNR, 11 is Normal Operation; in BCD, it is Minus. In BNR, 00 is
Failure Warning; in BCD, it is Plus. A model that has learned one table and applies it
to both gets a confident wrong answer on half of each pair. Every sample stores the
answers a model gives with the wrong tables, so a run can name the mistake:

- ``swapped_answer``: each word read with the other encoding's table
- ``bnr_table_answer`` / ``bcd_table_answer``: both words read with one table

For SSM 01 and 10 both tables agree (No Computed Data, Functional Test), so those
items are controls: all three wrong-table answers equal the correct one.

The SSM tables, keyed by bit 31 then bit 30 (from the catalogue and the decoder):

    bits  BNR                BCD
    00    failure_warning    plus   (Plus, North, East, Right, To, Above)
    01    no_computed_data   no_computed_data
    10    functional_test    functional_test
    11    normal_operation   minus  (Minus, South, West, Left, From, Below)

Which words are used (``build_items``), all from ``generate(N_WORDS, SEED)``, good
parity unless noted:

- ``PAIRS[ssm]`` pairs for each SSM value: 4 for 11 and for 00 (the discriminating
  cases), and 2 each for 01 and 10 (controls). Each pair is one BNR and one BCD word
  with that SSM, labels rotated so each pair uses labels not used just before.
- ``FAULT_PAIRS`` pairs in which one word has failed parity: a generator fault word
  whose flipped bit is neither in the label nor in the SSM, so the label and SSM still
  read as generated. It is paired with a good word of the other encoding that has the
  same SSM bits. The corrupt word's answer is a rejection.
- Which word comes first alternates from item to item (BNR first in odd items, BCD
  first in even ones), so position never gives the encoding away.

The prompt gives each word with its label (for example ``Word 1: 0x60B603D1 (label
213)``), so B2 does not depend on reading the label bit order, which is B1. The model
finds each label's encoding in the label map of the family preamble
(data/preambles/arinc429_v1.0.0.txt), which does not state the SSM tables.

B2 answer (canonical JSON, keys in alphabetical order):

    word_1_meaning, word_1_ssm, word_2_meaning, word_2_ssm

A meaning is one of the codes in ``MEANING_CODES``; ``ssm`` is two characters, bit 31
then bit 30. A word that fails parity gets ``word_N_meaning`` = "reject" and no
``word_N_ssm`` key, because nothing is reported from a corrupt word.

Validation (``validate_samples``) decodes every word again with the reference decoder
and fails if the target, the label shown in the prompt or the word itself differs. A
hand edit, generator or decoder drift therefore stops the run.

Build the task file (deterministic; the tests check the committed file matches it):

    python scripts/arinc429/build_b2_tasks.py

Check the committed file against the decoder and print one sample:

    python -m bench.arinc429.b2
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample

from bench.arinc429 import decoder
from bench.arinc429 import generator as gen
from bench.arinc429.b1 import PREAMBLE, SOURCE, canonical

B2_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = REPO_ROOT / "data" / "tasks" / f"b2_arinc429_ssm_interpretation_v{B2_VERSION}.jsonl"

SEED = 5290
N_WORDS = 1000
# Pairs per SSM value, in the order items are written. 11 and 00 discriminate; 01 and
# 10 mean the same in both tables and are controls.
PAIRS: dict[str, int] = {"11": 4, "00": 4, "01": 2, "10": 2}
FAULT_PAIRS = 2
ENCODINGS = ("BNR", "BCD")

REJECT = "reject"

# The decoder's SSM meanings (decoder.SSM_MEANINGS) as answer codes.
MEANING_CODES: dict[str, str] = {
    "Failure Warning": "failure_warning",
    "No Computed Data": "no_computed_data",
    "Functional Test": "functional_test",
    "Normal Operation": "normal_operation",
    "Plus, North, East, Right, To, Above": "plus",
    "Minus, South, West, Left, From, Below": "minus",
}

TASK_TEXT = """\
Interpret the sign/status matrix (SSM) of each ARINC 429 word below.

Word 1: {word_1} (label {label_1})
Word 2: {word_2} (label {label_2})

Answer with a single JSON object and nothing else: keys in alphabetical order, no \
spaces or line breaks, strings in double quotes. For each word N (1 and 2), report:
- "word_N_ssm": the SSM bits as a two-character string, bit 31 then bit 30 (for \
example "01")
- "word_N_meaning": what the SSM means for this word, given the label's data type in \
the label map. Use exactly one of these codes:
  "failure_warning", "no_computed_data", "functional_test", "normal_operation",
  "plus" (plus, north, east, right, to, above),
  "minus" (minus, south, west, left, from, below)

If a word fails the parity check, set its "word_N_meaning" to "reject" and leave out \
its "word_N_ssm"."""


class B2ValidationError(ValueError):
    """A B2 sample's target does not match the reference decoder."""


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def word_answer(word: int) -> dict[str, str]:
    """One word's part of the answer, from the reference decoder."""
    result = decoder.decode(word)
    if result.status == "rejected":
        if not (result.reason or "").startswith("parity"):
            raise B2ValidationError(f"word 0x{word:08X} is rejected for a reason B2 does not "
                                    f"ask about: {result.reason}")
        return {"meaning": REJECT}
    if result.status != "decoded" or result.decoded is None or result.fields is None:
        raise B2ValidationError(f"word 0x{word:08X} is not usable for B2: {result.reason}")
    return {"meaning": MEANING_CODES[result.decoded["ssm_meaning"]],
            "ssm": result.fields["ssm_bits"]}


def table_word_answer(word: int, table: str) -> dict[str, str]:
    """The answer for a word whose SSM is read with ``table`` ("BNR", "BCD", or "other")."""
    answer = word_answer(word)
    if answer["meaning"] == REJECT:
        return answer
    result = decoder.decode(word)
    assert result.decoded is not None
    if table == "other":
        table = "BCD" if result.decoded["encoding"] == "BNR" else "BNR"
    return {**answer, "meaning": MEANING_CODES[decoder.SSM_MEANINGS[table][answer["ssm"]]]}


def combine(parts: list[dict[str, str]]) -> dict[str, str]:
    """Per-word answers as the flat B2 answer object."""
    out: dict[str, str] = {}
    for n, part in enumerate(parts, start=1):
        out[f"word_{n}_meaning"] = part["meaning"]
        if "ssm" in part:
            out[f"word_{n}_ssm"] = part["ssm"]
    return out


def reference_answer(words: list[int]) -> dict[str, str]:
    return combine([word_answer(w) for w in words])


def table_answer(words: list[int], table: str) -> dict[str, str]:
    """The whole answer with every SSM read using ``table`` ("BNR", "BCD" or "other")."""
    return combine([table_word_answer(w, table) for w in words])


def shown_label(word: int) -> str:
    """The label the prompt states for a word: what bits 1-8 carry, read correctly."""
    return decoder.read_label(word)


# ---------------------------------------------------------------------------
# Item selection
# ---------------------------------------------------------------------------


@functools.cache
def generated_words() -> tuple[dict[str, Any], ...]:
    return tuple(gen.generate(N_WORDS, SEED))


def _encoding(record: dict[str, Any]) -> str:
    return record["profile"]["encoding"]


class _Pool:
    """Good words by (encoding, SSM), handed out in order, rotating labels."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.free: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in records:
            if r["parity_ok"]:
                self.free.setdefault((_encoding(r), r["ssm_bits"]), []).append(r)
        self.recent: dict[tuple[str, str], list[str]] = {}

    def take(self, encoding: str, ssm: str) -> dict[str, Any]:
        key = (encoding, ssm)
        words = self.free.get(key, [])
        if not words:
            raise B2ValidationError(f"seed {SEED} has too few {encoding} words with SSM {ssm}")
        recent = self.recent.setdefault(key, [])
        labels = {w["label_octal"] for w in words}
        if labels <= set(recent):
            recent.clear()  # every label used; start the rotation again
        pick = next(w for w in words if w["label_octal"] not in recent)
        words.remove(pick)
        recent.append(pick["label_octal"])
        return pick


def _fault_words(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generator fault words whose label and SSM still read as generated."""
    keep = []
    for r in records:
        fault = r["fault"]
        if fault is not None and not (1 <= fault["flipped_bit"] <= 8 or 30 <= fault["flipped_bit"] <= 31):
            keep.append(r)
    return keep


def build_items() -> list[dict[str, Any]]:
    """The B2 items: each a case and its two generated word records, in prompt order."""
    records = generated_words()
    pool = _Pool(records)
    items: list[dict[str, Any]] = []

    def ordered(pair: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bnr_first = len(items) % 2 == 0  # item 1 (index 0) has BNR first
        return pair if (_encoding(pair[0]) == "BNR") == bnr_first else pair[::-1]

    for ssm, count in PAIRS.items():
        for _ in range(count):
            pair = ordered([pool.take("BNR", ssm), pool.take("BCD", ssm)])
            items.append({"case": f"ssm_{ssm}", "words": pair})
    for fault in _fault_words(records)[:FAULT_PAIRS]:
        other = "BCD" if _encoding(fault) == "BNR" else "BNR"
        pair = ordered([fault, pool.take(other, fault["ssm_bits"])])
        items.append({"case": "parity_fault", "words": pair})
    if sum(1 for i in items if i["case"] == "parity_fault") < FAULT_PAIRS:
        raise B2ValidationError(f"seed {SEED} has too few usable fault words")
    return items


def build_records() -> list[dict[str, Any]]:
    """Task records in the loader's format (bench/loader.py), one per pair."""
    records = []
    for n, item in enumerate(build_items(), start=1):
        words = [int(r["word_hex"], 16) for r in item["words"]]
        answer = reference_answer(words)
        # Cross-check: the decoder must agree with the generator's own ground truth.
        for i, r in enumerate(item["words"], start=1):
            truth = r["decoded"]
            got = answer[f"word_{i}_meaning"]
            expected = REJECT if truth is None else MEANING_CODES[truth["ssm_meaning"]]
            if got != expected:
                raise B2ValidationError(f"{r['word_id']}: decoder gives {got}, generator {expected}")
        labels = [shown_label(w) for w in words]
        records.append({
            "id": f"b2-{n:02d}",
            "family": "B",
            "preamble": PREAMBLE,
            "source": SOURCE,
            "input": TASK_TEXT.format(word_1=item["words"][0]["word_hex"], label_1=labels[0],
                                      word_2=item["words"][1]["word_hex"], label_2=labels[1]),
            "target": canonical(answer),
            "metadata": {
                "subfamily": "B2",
                "b2_version": B2_VERSION,
                "case": item["case"],
                "words": [r["word_hex"] for r in item["words"]],
                "word_ids": [r["word_id"] for r in item["words"]],
                "labels": labels,
                "encodings": [_encoding(r) for r in item["words"]],
                "ssm_bits": [r["ssm_bits"] for r in item["words"]],
                "discriminating": item["case"] in ("ssm_11", "ssm_00", "parity_fault"),
                "swapped_answer": canonical(table_answer(words, "other")),
                "bnr_table_answer": canonical(table_answer(words, "BNR")),
                "bcd_table_answer": canonical(table_answer(words, "BCD")),
                "expected_status": ["rejected" if r["decoded"] is None else "decoded"
                                    for r in item["words"]],
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@functools.cache
def _word_by_id() -> dict[str, str]:
    return {r["word_id"]: r["word_hex"] for r in generated_words()}


def sample_problems(sample: Sample) -> list[str]:
    """Why a sample fails validation; empty when it passes."""
    meta = sample.metadata or {}
    words_hex = meta.get("words")
    if not (isinstance(words_hex, list) and len(words_hex) == 2
            and all(isinstance(w, str) and w for w in words_hex)):
        return ["metadata does not have two words"]
    problems = []
    try:
        words = [int(w, 16) for w in words_hex]
    except ValueError:
        return [f"words {words_hex} are not hexadecimal"]
    for i, (word_hex, word_id) in enumerate(zip(words_hex, meta.get("word_ids") or [None, None]), start=1):
        if word_id is not None and _word_by_id().get(word_id) != word_hex:
            problems.append(f"word {i} {word_hex} is not word {word_id!r} from generator seed {SEED}")
        line = f"Word {i}: {word_hex} (label {shown_label(int(word_hex, 16))})"
        if line not in str(sample.input):
            problems.append(f"prompt does not contain {line!r}")
    if not isinstance(sample.target, str):
        return [*problems, "target must be a single JSON string"]
    try:
        expected = reference_answer(words)
    except B2ValidationError as exc:
        return [*problems, str(exc)]
    if sample.target != canonical(expected):
        problems.append(f"target {sample.target} does not match the reference decoder "
                        f"{canonical(expected)}")
    return problems


def validate_samples(samples: Iterable[Sample]) -> None:
    """Raise B2ValidationError listing every sample that does not match the decoder."""
    failures = [f"{s.id}: {'; '.join(p)}" for s in samples if (p := sample_problems(s))]
    if failures:
        raise B2ValidationError(
            f"B2 task set failed validation ({len(failures)} sample(s)):\n" + "\n".join(failures)
        )


def main() -> None:
    from bench.loader import load_tasks

    parser = argparse.ArgumentParser(description="Validate the B2 task file and show a sample.")
    parser.add_argument("tasks", nargs="?", default=str(TASK_FILE), help="B2 task file")
    parser.add_argument("--show", default=None, help="sample ID to print (default: the first)")
    args = parser.parse_args()
    dataset = load_tasks(args.tasks)
    try:
        validate_samples(dataset)
    except B2ValidationError as exc:
        raise SystemExit(f"error: {exc}") from None
    sample = next((s for s in dataset if s.id == args.show), None) if args.show else dataset[0]
    if sample is None:
        raise SystemExit(f"error: no sample {args.show!r}")
    print(f"{len(dataset)} samples match the reference decoder "
          f"(bench.arinc429.decoder=={decoder.DECODER_VERSION}).\n")
    meta = sample.metadata or {}
    print(f"===== {sample.id} ({meta.get('case')}; encodings {meta.get('encodings')})")
    print(f"----- target:  {sample.target}")
    print(f"----- swapped: {meta.get('swapped_answer')}  (each word read with the other table)")
    print(f"----- BNR-only: {meta.get('bnr_table_answer')}")
    print(f"----- BCD-only: {meta.get('bcd_table_answer')}")
    print("----- task text:")
    print(meta.get("task_text"))


if __name__ == "__main__":
    main()
