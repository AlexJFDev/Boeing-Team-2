"""ARINC 429 bus conformance checker: word timing, gaps and transmission rates (WBS 3.5).

The reference that B3 task targets come from (issue #60). It takes a timed word
stream and the bus rules a task states, and returns which words break which rule.
Every rule is a premise the task states, so the verdict is arithmetic on the stated
numbers, not a judgement:

``word_timing``
    A word is 32 bit times long. Its duration (end minus start) must be within the
    stated tolerance of 32 bit times. At 100 kbit/s and ±1% that is 316.8 to 323.2 us.

``gap``
    From the end of one word to the start of the next (whatever their labels) there
    must be at least ``min_gap_bits`` bit times of null, 4 by default. A shorter gap,
    or an overlap, makes the later word non-conforming.

``rate_fast`` / ``rate_slow``
    For a label with a stated transmission interval window, the time from the start of
    one word on that label to the start of the next must lie in the window, ends
    included. The later word of a short interval is ``rate_fast``, of a long one
    ``rate_slow``. A label with no stated window is not checked for rate, and only
    intervals between words in the stream count: nothing before the first word of a
    label or after its last.

Physical-layer premises (free secondary sources; the specification itself is paid):
high speed is 100 kbit/s ±1% and low speed 12 to 14.5 kbit/s (ALTA ARINC Protocol
Summary; AEA "Tech Time" July 2008), and sequential words are separated by at least
4 bit times of null (Abaco; AIM ARINC 429 tutorial). Per-label transmission rates are
in the paid specification's label tables, so the task catalogue (B3) says to state
the configured rate in each task; this checker only ever uses the rates it is given.

It does not read the words' contents: parity, SSM and SDI checks are B1, B2 and B4.
It is written without reference to the B3 stream builder (``bench/arinc429/b3.py``)
so that the builder's intended violations and this checker's verdict are two
separate computations that must agree.

Times are integers in microseconds, and limits are exact fractions, so nothing
depends on floating-point rounding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

CONFORMANCE_VERSION = "1.0.0"

WORD_BITS = 32
DEFAULT_MIN_GAP_BITS = 4
DEFAULT_TIMING_TOLERANCE = Fraction(1, 100)  # ±1%

RULES = ("gap", "rate_fast", "rate_slow", "word_timing")

# Bus speeds the B3 tasks state. Low speed may be anywhere from 12 to 14.5 kbit/s;
# a task fixes one value so that the word duration and gap are determinate.
SPEEDS: dict[str, int] = {"high": 100_000, "low": 12_500}


class StreamError(ValueError):
    """The stream is not something the checker can judge (bad order or values)."""


@dataclass(frozen=True)
class TimedWord:
    """One word as observed on the bus: its label and its start and end times in us."""

    label: str  # three octal digits
    start_us: int
    end_us: int
    word_hex: str = ""


@dataclass(frozen=True)
class BusRules:
    """The rules a task states for one bus."""

    bit_rate_bps: int
    rate_windows_us: Mapping[str, tuple[int, int]] = field(default_factory=dict)  # label -> (min, max)
    min_gap_bits: int = DEFAULT_MIN_GAP_BITS
    timing_tolerance: Fraction = DEFAULT_TIMING_TOLERANCE

    @property
    def bit_time_us(self) -> Fraction:
        return Fraction(1_000_000, self.bit_rate_bps)

    @property
    def word_time_us(self) -> Fraction:
        return WORD_BITS * self.bit_time_us

    @property
    def word_time_limits_us(self) -> tuple[Fraction, Fraction]:
        nominal = self.word_time_us
        return nominal * (1 - self.timing_tolerance), nominal * (1 + self.timing_tolerance)

    @property
    def min_gap_us(self) -> Fraction:
        return self.min_gap_bits * self.bit_time_us


@dataclass(frozen=True)
class WordCheck:
    """What the checker measured for one word (1-based ``number``) and what it broke."""

    number: int
    label: str
    duration_us: int
    gap_us: int | None  # None for the first word
    interval_us: int | None  # since the previous word on the same label; None if first
    window_us: tuple[int, int] | None  # the label's stated window, if any
    violations: tuple[str, ...]


def as_words(rows: Iterable[Mapping[str, Any] | Sequence[Any] | TimedWord]) -> list[TimedWord]:
    """Accept TimedWords, dicts with label/start_us/end_us, or (label, start, end[, hex]) rows."""
    out = []
    for row in rows:
        if isinstance(row, TimedWord):
            out.append(row)
        elif isinstance(row, Mapping):
            out.append(TimedWord(str(row["label"]), int(row["start_us"]), int(row["end_us"]),
                                 str(row.get("word_hex", ""))))
        else:
            label, start, end, *rest = row
            out.append(TimedWord(str(label), int(start), int(end), str(rest[0]) if rest else ""))
    return out


def check_stream(rows: Iterable[Mapping[str, Any] | Sequence[Any] | TimedWord],
                 rules: BusRules) -> list[WordCheck]:
    """Measure and judge every word in the stream, in stream order."""
    words = as_words(rows)
    for i, w in enumerate(words, start=1):
        if w.end_us <= w.start_us:
            raise StreamError(f"word {i} ends at or before it starts")
        if i > 1 and w.start_us < words[i - 2].start_us:
            raise StreamError(f"word {i} starts before word {i - 1}; the stream must be in start order")
    lo, hi = rules.word_time_limits_us
    last_start: dict[str, int] = {}
    checks = []
    for i, w in enumerate(words, start=1):
        broken = []
        duration = w.end_us - w.start_us
        if not lo <= duration <= hi:
            broken.append("word_timing")
        gap = None if i == 1 else w.start_us - words[i - 2].end_us
        if gap is not None and gap < rules.min_gap_us:
            broken.append("gap")
        interval = None if w.label not in last_start else w.start_us - last_start[w.label]
        window = rules.rate_windows_us.get(w.label)
        if interval is not None and window is not None:
            if interval < window[0]:
                broken.append("rate_fast")
            elif interval > window[1]:
                broken.append("rate_slow")
        last_start[w.label] = w.start_us
        checks.append(WordCheck(i, w.label, duration, gap, interval,
                                tuple(window) if window else None, tuple(sorted(broken))))
    return checks


def violations(rows: Iterable[Mapping[str, Any] | Sequence[Any] | TimedWord],
               rules: BusRules) -> list[list[Any]]:
    """The non-conforming words as ``[[word_number, rule], ...]``, sorted."""
    return [[c.number, rule] for c in check_stream(rows, rules) for rule in c.violations]


def format_us(value: Fraction | int) -> str:
    """A time in us as written in task text: an integer, or a short exact decimal."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if Fraction(text) != value:
        raise ValueError(f"{value} us has no short exact decimal form")
    return text
