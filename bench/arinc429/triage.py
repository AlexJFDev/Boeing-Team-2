"""ARINC 429 stream triage: hardware fault or injected word (WBS 3.6).

The reference that B4 task targets come from (issue #61). It takes a stream of words
from one transmitter and the rules a task states about that transmitter, and says
which words are genuine, which were corrupted on the bus, and which were injected.

The distinction is the catalogue's (B4): a hardware fault flips bits anywhere in the
word and leaves parity broken; an injection changes the data and recomputes parity,
so the word still validates. The tell is parity, not how odd the value looks: a fault
in a high data bit makes a large jump too, but its parity fails.

The rules, all stated in the task:

- The transmitter sends one label with a fixed SDI and SSM, and word 1 is genuine.
- The parameter changes by at most ``max_step`` between consecutive transmissions, so
  a genuine word k transmissions after the last genuine word is within k x max_step of
  it. Every word in between counts, whatever happened to it.
- A word that fails parity is a ``fault``.
- A word that passes parity but is not genuine (value outside the limit, or the wrong
  label, SDI or SSM) is ``injected``.

Verdict: "injection" if any word is injected; otherwise "hardware_fault" if any word
failed parity; otherwise "none". The words behind the verdict are listed.

Written without reference to the B4 stream builder (``bench/arinc429/b4.py``), so the
builder's intended alterations and this module's verdict are two computations that
must agree. Values are compared as exact fractions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from bench.arinc429 import decoder

TRIAGE_VERSION = "1.0.0"

VERDICTS = ("injection", "hardware_fault", "none")


class StreamError(ValueError):
    """The stream cannot be triaged under the stated rules (for example word 1 is not genuine)."""


@dataclass(frozen=True)
class Transmitter:
    """What a task states about the one transmitter behind a stream."""

    label: str  # three octal digits
    sdi: int
    ssm: str  # bit 31 then bit 30
    max_step: Fraction  # largest change between consecutive transmissions, in the label's units


@dataclass(frozen=True)
class WordFinding:
    number: int  # 1-based
    status: str  # "genuine", "fault" or "injected"
    value: Fraction | None  # the decoded value, None when parity fails
    reason: str


def _value(result: decoder.DecodeResult) -> Fraction:
    assert result.decoded is not None
    return Fraction(result.decoded["value"])  # exact: BNR values are dyadic, BCD integral


def judge_stream(words: Iterable[int], tx: Transmitter,
                 profiles: Mapping[str, Mapping[str, Any]] | None = None) -> list[WordFinding]:
    """Classify every word of the stream, in order."""
    findings: list[WordFinding] = []
    last: tuple[int, Fraction] | None = None  # (number, value) of the last genuine word
    for n, word in enumerate(words, start=1):
        result = decoder.decode(word, profiles)
        if result.fields is None:
            raise StreamError(f"word {n} is not a 32-bit word: {result.reason}")
        fields = result.fields
        if not fields["parity_ok"]:
            if n == 1:
                raise StreamError("word 1 fails parity, but the rules say it is genuine")
            findings.append(WordFinding(n, "fault", None, "fails the parity check"))
            continue
        mismatch = [name for name, got, want in (("label", fields["label_octal"], tx.label),
                                                 ("SDI", fields["sdi"], tx.sdi),
                                                 ("SSM", fields["ssm_bits"], tx.ssm)) if got != want]
        if result.status != "decoded" or mismatch:
            if n == 1:
                raise StreamError(f"word 1 is not the transmitter's ({', '.join(mismatch) or result.reason})")
            what = f"{', '.join(mismatch)} not the transmitter's" if mismatch else str(result.reason)
            findings.append(WordFinding(n, "injected", None, f"passes parity but {what}"))
            continue
        value = _value(result)
        if last is None:
            findings.append(WordFinding(n, "genuine", value, "word 1, genuine by premise"))
            last = (n, value)
            continue
        k = n - last[0]
        limit = k * tx.max_step
        change = abs(value - last[1])
        if change > limit:
            findings.append(WordFinding(n, "injected", value,
                                        f"passes parity but changed {change} in {k} transmission(s), limit {limit}"))
            continue
        findings.append(WordFinding(n, "genuine", value, f"changed {change} in {k} transmission(s)"))
        last = (n, value)
    if not findings:
        raise StreamError("the stream is empty")
    return findings


def verdict_of(findings: list[WordFinding]) -> dict[str, Any]:
    injected = [f.number for f in findings if f.status == "injected"]
    if injected:
        return {"verdict": "injection", "words": injected}
    faults = [f.number for f in findings if f.status == "fault"]
    if faults:
        return {"verdict": "hardware_fault", "words": faults}
    return {"verdict": "none", "words": []}


def triage(words: Iterable[int], tx: Transmitter,
           profiles: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The verdict and the words behind it."""
    return verdict_of(judge_stream(words, tx, profiles))
