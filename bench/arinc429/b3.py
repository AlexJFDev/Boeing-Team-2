"""B3 task family: ARINC 429 bus conformance - rates, gaps, word timing (WBS 3.5, issue #60).

Each B3 task shows the model a timed stream of synthetic ARINC 429 words, with each
word's start and end time and its label, and asks which words break the bus rules and
which rule each breaks. Ground truth is the conformance checker
(bench/arinc429/conformance.py). Words come from the seeded generator
(bench/arinc429/generator.py, WBS 3.1), and every one has good parity: B3 is about
timing, and a corrupt word is B1/B4's subject.

Every rule is stated in the task's own premises (the loader prints them after the
family preamble under "Premises for this task:"), because they change from task to
task. This follows the task catalogue's B3 scoping note: per-label transmission rates
are in the paid specification, so the task states the configured rate instead.

- bus speed and bit time (high speed 100 kbit/s, or low speed fixed at 12.5 kbit/s)
- ``word_timing``: a word lasts 32 bit times ±1%
- ``gap``: at least 4 bit times of null from the end of a word to the start of the next
- ``rate_fast`` / ``rate_slow``: for each label with a stated rate, the start-to-start
  interval between its words must lie within ±10% of the stated interval
- labels with no stated rate are not rate-checked; only intervals inside the stream count

The discriminator (catalogue B3): the verdict depends on the stated numbers. The same
label is sent every 31.25 ms in one task and every 62.5 ms in another, so a model that
brings its own idea of a label's rate, or of the bus speed, gives a confident,
well-formed and wrong list. Every sample stores two such wrong answers, computed by
the checker under rules that contradict the premises:

- ``contradicting_rate_answer``: every stated interval doubled (the label taken to be
  sent half as often as stated). Good intervals then read as ``rate_fast``, and a
  dropped transmission looks normal.
- ``wrong_speed_answer``: the other bus speed, so every word's duration is wrong.

Some tasks also carry a label with no stated rate whose words come at irregular
intervals (``distractor_label``). The builder puts no fault on its words, so flagging
one is always a mistake.

Cases (``ITEMS``), 11 at high speed and 5 at low speed: conforming streams (controls,
so a model that flags everything fails), a dropped transmission (``rate_slow``), an
extra transmission (two ``rate_fast`` words), a short gap or an overlap (``gap``), a
word too long or too short (``word_timing``), and mixed streams with two faults.

B3 answer (canonical JSON):

    {"violations":[[word_number,"rule"],...]}

sorted by word number then rule name; ``{"violations":[]}`` for a conforming stream.

The builder places the words and records the violations it meant to inject. The
checker then judges the finished stream on its own, and the build fails unless the two
agree. Validation (``validate_samples``) re-checks every sample's target, premises,
prompt rows and words against the checker, the metadata and the generator, so a hand
edit or drift stops the run.

Build the task file (deterministic; the tests check the committed file matches it):

    python scripts/arinc429/build_b3_tasks.py

Check the committed file and print one sample with the checker's measurements:

    python -m bench.arinc429.b3 --show b3-03
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample

from bench.arinc429 import conformance as cf
from bench.arinc429 import decoder
from bench.arinc429 import generator as gen
from bench.arinc429.b1 import PREAMBLE, SOURCE, canonical

B3_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = REPO_ROOT / "data" / "tasks" / f"b3_arinc429_bus_conformance_v{B3_VERSION}.jsonl"

SEED = 5350
N_WORDS = 3000
RATE_TOLERANCE = Fraction(1, 10)  # stated windows are ±10% of the stated interval
STREAM_START_US = 1000
SPAN_INTERVALS = 4  # a stream lasts this many of its slowest label's intervals
CLEAR_GAP_BITS = 6  # the builder keeps ordinary gaps at least this long (the rule is 4)
MAX_ATTEMPTS = 50
CONTRADICTING_RATE_FACTOR = 2

RULE_TEXT = '"gap", "rate_fast", "rate_slow" and "word_timing"'

TASK_TEXT = """\
Check the ARINC 429 word stream below against the bus rules stated above, and report \
every non-conforming word and the rule it breaks.

Each line gives the word number, the start and end time of the word in microseconds \
(us) from the start of the recording, its label, and the word. Every word passes the \
parity check.

{rows}

Answer with a single JSON object and nothing else, with no spaces or line breaks: \
{{"violations":[[N,"rule"],...]}} with one [word number, rule] pair for each rule a \
word breaks, sorted by word number and then by rule name. The rules are {rules}. If \
every word conforms, answer {{"violations":[]}}"""

ROW_HEADER = " #   start_us     end_us  label  word"
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-7]{3})\s+(0x[0-9A-F]{8})\s*$")
RATE_PREMISE_RE = re.compile(r"^Label ([0-7]{3}) is transmitted every [\d.]+ ms \((\d+) us\)\. "
                             r"The interval .* must be from (\d+) to (\d+) us inclusive\.")
SPEED_PREMISE_RE = re.compile(r"^This bus runs at (high|low) speed")


class B3ValidationError(ValueError):
    """A B3 sample's target does not match the conformance checker."""


# ---------------------------------------------------------------------------
# The items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    case: str
    speed: str  # key of conformance.SPEEDS
    rates: tuple[tuple[str, int], ...]  # (label, stated interval in us)
    faults: tuple[str, ...] = ()  # rate_slow, rate_fast, gap_short, overlap, word_long, word_short
    distractor: str | None = None  # a label with no stated rate, sent irregularly


ITEMS: tuple[Item, ...] = (
    # High speed, 100 kbit/s
    Item("conforming", "high", (("203", 62_500), ("206", 31_250))),
    Item("conforming", "high", (("324", 50_000), ("012", 25_000)), distractor="213"),
    Item("rate_slow", "high", (("203", 31_250), ("204", 62_500)), ("rate_slow",)),
    Item("rate_slow", "high", (("012", 50_000), ("206", 100_000)), ("rate_slow",)),
    Item("rate_fast", "high", (("324", 62_500), ("203", 31_250)), ("rate_fast",)),
    Item("rate_fast", "high", (("206", 50_000), ("204", 25_000)), ("rate_fast",), distractor="013"),
    Item("gap", "high", (("203", 62_500), ("206", 62_500), ("012", 31_250)), ("gap_short",)),
    Item("gap", "high", (("204", 50_000), ("324", 25_000)), ("overlap",)),
    Item("word_timing", "high", (("206", 31_250), ("012", 62_500)), ("word_long",)),
    Item("word_timing", "high", (("203", 25_000), ("324", 50_000)), ("word_short",), distractor="213"),
    Item("mixed", "high", (("204", 31_250), ("206", 62_500), ("012", 62_500)), ("rate_slow", "gap_short")),
    # Low speed, 12.5 kbit/s: words take 2.56 ms, so only slower rates fit
    Item("conforming", "low", (("203", 62_500), ("206", 100_000))),
    Item("rate_slow", "low", (("324", 50_000), ("012", 100_000)), ("rate_slow",)),
    Item("gap", "low", (("204", 62_500), ("206", 50_000)), ("gap_short",)),
    Item("word_timing", "low", (("203", 100_000), ("012", 50_000)), ("word_long",)),
    Item("mixed", "low", (("324", 62_500), ("204", 100_000)), ("rate_fast", "word_short"), distractor="013"),
)


# ---------------------------------------------------------------------------
# Rules and premises
# ---------------------------------------------------------------------------


def window(interval_us: int, tolerance: Fraction = RATE_TOLERANCE) -> tuple[int, int]:
    lo, hi = interval_us * (1 - tolerance), interval_us * (1 + tolerance)
    if lo.denominator != 1 or hi.denominator != 1:
        raise ValueError(f"interval {interval_us} us does not give a whole-microsecond window")
    return int(lo), int(hi)


def rules_for(speed: str, rates: Mapping[str, int], scale: int = 1) -> cf.BusRules:
    """The bus rules for a speed and stated intervals; ``scale`` multiplies each interval."""
    return cf.BusRules(cf.SPEEDS[speed], {label: window(i * scale) for label, i in rates.items()})


def _ms(us: int) -> str:
    return cf.format_us(Fraction(us, 1000))


def render_premises(speed: str, rates: Mapping[str, int]) -> list[str]:
    """The task's own premises: bus speed, word timing, gap, then one rate per label in label order."""
    rules = rules_for(speed, rates)
    lo, hi = rules.word_time_limits_us
    kbit = cf.format_us(Fraction(rules.bit_rate_bps, 1000))
    premises = [
        f"This bus runs at {speed} speed, {kbit} kbit/s, so one bit time is "
        f"{cf.format_us(rules.bit_time_us)} us.",
        f"Word timing: a word is 32 bit times ({cf.format_us(rules.word_time_us)} us). A word "
        f"whose duration (end time minus start time) is outside {cf.format_us(lo)} to "
        f"{cf.format_us(hi)} us inclusive (32 bit times plus or minus 1%) is non-conforming "
        f'under rule "word_timing".',
        f"Gap: from the end of one word to the start of the next word, whatever their labels, "
        f"there must be at least {rules.min_gap_bits} bit times "
        f"({cf.format_us(rules.min_gap_us)} us) of null. If the gap is shorter, or the words "
        f'overlap, the later word is non-conforming under rule "gap".',
    ]
    for label, interval in sorted(rates.items()):  # label order, whatever order the dict has
        lo_i, hi_i = window(interval)
        premises.append(
            f"Label {label} is transmitted every {_ms(interval)} ms ({interval} us). The interval "
            f"from the start of one label {label} word to the start of the next label {label} "
            f"word must be from {lo_i} to {hi_i} us inclusive. The later word of a shorter "
            f'interval is non-conforming under rule "rate_fast", and of a longer one under rule '
            f'"rate_slow".'
        )
    premises.append(
        "No other label has a stated transmission rate, so no other label is checked for rate. "
        "Only intervals between two words in this stream count: nothing before a label's first "
        "word or after its last."
    )
    return premises


# ---------------------------------------------------------------------------
# Building a stream
# ---------------------------------------------------------------------------


class _WordPool:
    """Good-parity generated words by label, handed out in order, never reused."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.free: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            if r["parity_ok"]:
                self.free.setdefault(r["label_octal"], []).append(r)

    def take(self, label: str) -> dict[str, Any]:
        words = self.free.get(label)
        if not words:
            raise B3ValidationError(f"seed {SEED} has too few good words on label {label}")
        return words.pop(0)


@functools.cache
def generated_words() -> tuple[dict[str, Any], ...]:
    return tuple(gen.generate(N_WORDS, SEED))


@dataclass
class _Slot:
    label: str
    start: int
    end: int
    intended: set[str]


def _base_stream(item: Item, rng: random.Random) -> list[_Slot]:
    """Every rated label on schedule with small jitter, plus the distractor, gaps kept clear."""
    rules = rules_for(item.speed, dict(item.rates))
    bit = rules.bit_time_us
    word = int(rules.word_time_us)
    span = SPAN_INTERVALS * max(i for _, i in item.rates)
    slots: list[_Slot] = []
    for label, interval in item.rates:
        t = STREAM_START_US + rng.randrange(0, interval // 2)
        while t <= STREAM_START_US + span:
            slots.append(_Slot(label, t, 0, set()))
            t = slots[-1].start + interval + rng.randint(-interval // 100, interval // 100)
    if item.distractor:
        t = STREAM_START_US + rng.randrange(0, span // 8)
        short = True
        while t <= STREAM_START_US + span:
            slots.append(_Slot(item.distractor, t, 0, set()))
            # Irregular on purpose (no rate is stated): short and long intervals alternate.
            t += int(span * (rng.uniform(0.06, 0.12) if short else rng.uniform(0.25, 0.4)))
            short = not short
    slots.sort(key=lambda s: s.start)
    clear = int(CLEAR_GAP_BITS * bit)
    jitter = word // 200  # durations within ±0.5%, inside the ±1% rule
    for i, s in enumerate(slots):
        if i:
            s.start = max(s.start, slots[i - 1].end + clear + rng.randint(0, int(2 * bit)))
        s.end = s.start + word + rng.randint(-jitter, jitter)
    return slots


def _interval_ok(slots: list[_Slot], idx: int, rates: Mapping[str, int], new_start: int) -> bool:
    """Builder's own check: moving slot ``idx`` to ``new_start`` keeps its label's intervals in window."""
    label = slots[idx].label
    if label not in rates:
        return True
    lo, hi = window(rates[label])
    same = [i for i, s in enumerate(slots) if s.label == label]
    pos = same.index(idx)
    starts = [slots[i].start for i in same]
    starts[pos] = new_start
    return all(lo <= b - a <= hi for a, b in zip(starts, starts[1:]))


def _clear(slots: list[_Slot], rules: cf.BusRules) -> bool:
    """Builder's own check that every gap is comfortably clear (before any gap fault)."""
    need = CLEAR_GAP_BITS * rules.bit_time_us
    return all(b.start - a.end >= need for a, b in zip(slots, slots[1:]))


def _middle_of_label(slots: list[_Slot], label: str) -> list[int]:
    same = [i for i, s in enumerate(slots) if s.label == label]
    return same[1:-1]


def _apply(fault: str, item: Item, slots: list[_Slot], rng: random.Random,
           rules: cf.BusRules, used: set[str]) -> bool:
    """Inject one fault and mark the words it should make non-conforming. False if no room."""
    rates = dict(item.rates)
    bit = rules.bit_time_us
    word = int(rules.word_time_us)
    if fault in ("rate_slow", "rate_fast"):
        label = next((lab for lab, _ in item.rates if lab not in used), None)
        if label is None:
            return False
        used.add(label)
        middle = _middle_of_label(slots, label)
        if len(middle) < 2:
            return False
        k = rng.choice(middle[:-1])
        if fault == "rate_slow":  # drop word k: the next word on the label comes two intervals later
            following = next(i for i in range(k + 1, len(slots)) if slots[i].label == label)
            slots[following].intended.add("rate_slow")
            del slots[k]
            return True
        # rate_fast: an extra word halfway between k and the next word on the label
        following = next(i for i in range(k + 1, len(slots)) if slots[i].label == label)
        mid = (slots[k].start + slots[following].start) // 2
        need = int(CLEAR_GAP_BITS * bit)
        for shift in range(0, rates[label] // 10, max(1, word // 4)):
            for start in (mid + shift, mid - shift):
                end = start + word
                if all(start - s.end >= need or s.start - end >= need for s in slots):
                    slots.append(_Slot(label, start, end, {"rate_fast"}))
                    slots.sort(key=lambda s: s.start)
                    # The next word on the label now follows only half an interval later too.
                    after = next(s for s in slots if s.label == label and s.start > start)
                    after.intended.add("rate_fast")
                    return True
        return False
    candidates = list(range(1, len(slots) - 1))
    rng.shuffle(candidates)
    if fault in ("gap_short", "overlap"):
        for i in candidates:
            prev, this = slots[i - 1], slots[i]
            if this.intended or prev.intended or prev.label == this.label or this.label == item.distractor:
                continue
            new_start = prev.end + int(2 * bit) if fault == "gap_short" else prev.end - int(4 * bit)
            dur = this.end - this.start
            # The builder's own judgement: the move must keep this label's rate and the next gap clear.
            if not _interval_ok(slots, i, rates, new_start):
                continue
            if slots[i + 1].start - (new_start + dur) < CLEAR_GAP_BITS * bit:
                continue
            this.start, this.end = new_start, new_start + dur
            this.intended.add("gap")
            return True
        return False
    if fault in ("word_long", "word_short"):
        factor = Fraction(103, 100) if fault == "word_long" else Fraction(97, 100)
        for i in candidates:
            this = slots[i]
            if this.intended or this.label == item.distractor:
                continue
            new_end = this.start + int(word * factor)
            if slots[i + 1].start - new_end < CLEAR_GAP_BITS * bit:
                continue
            this.end = new_end
            this.intended.add("word_timing")
            return True
        return False
    raise ValueError(f"unknown fault {fault!r}")


def build_stream(item: Item, n: int) -> list[_Slot]:
    """The timed stream for item ``n``, with the builder's intended violations."""
    rules = rules_for(item.speed, dict(item.rates))
    rates = dict(item.rates)
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(f"b3-{SEED}-{n}-{attempt}")
        slots = _base_stream(item, rng)
        # The builder's own acceptance of the base stream, before any fault.
        if not _clear(slots, rules):
            continue
        if not all(_interval_ok(slots, i, rates, s.start) for i, s in enumerate(slots)):
            continue
        used: set[str] = set()
        if all(_apply(f, item, slots, rng, rules, used) for f in item.faults):
            return slots
    raise B3ValidationError(f"b3-{n:02d}: could not place faults {item.faults} in {MAX_ATTEMPTS} attempts")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def stream_rows(stream: list[dict[str, Any]]) -> str:
    lines = [ROW_HEADER]
    for n, w in enumerate(stream, start=1):
        lines.append(f"{n:>2}  {w['start_us']:>9}  {w['end_us']:>9}  {w['label']}    {w['word_hex']}")
    return "\n".join(lines)


def answer(stream: list[dict[str, Any]], rules: cf.BusRules) -> dict[str, Any]:
    return {"violations": cf.violations(stream, rules)}


def bus_metadata(speed: str, rates: Mapping[str, int]) -> dict[str, Any]:
    return {
        "speed": speed,
        "bit_rate_bps": cf.SPEEDS[speed],
        "stated_intervals_us": dict(rates),
        "rate_windows_us": {label: list(window(i)) for label, i in rates.items()},
        "min_gap_bits": cf.DEFAULT_MIN_GAP_BITS,
        "word_timing_tolerance": str(cf.DEFAULT_TIMING_TOLERANCE),
    }


def build_records() -> list[dict[str, Any]]:
    pool = _WordPool(generated_words())
    records = []
    for n, item in enumerate(ITEMS, start=1):
        slots = build_stream(item, n)
        stream = []
        for s in slots:
            r = pool.take(s.label)
            stream.append({"word_id": r["word_id"], "word_hex": r["word_hex"], "label": s.label,
                           "start_us": s.start, "end_us": s.end})
        rates = dict(item.rates)
        target = answer(stream, rules_for(item.speed, rates))
        intended = sorted([i, rule] for i, s in enumerate(slots, start=1) for rule in s.intended)
        # Cross-check: the checker must find exactly the violations the builder placed.
        if target["violations"] != intended:
            raise B3ValidationError(f"b3-{n:02d}: checker finds {target['violations']}, "
                                    f"builder placed {intended}")
        other_speed = "low" if item.speed == "high" else "high"
        records.append({
            "id": f"b3-{n:02d}",
            "family": "B",
            "preamble": PREAMBLE,
            "source": SOURCE,
            "premises": render_premises(item.speed, rates),
            "input": TASK_TEXT.format(rows=stream_rows(stream), rules=RULE_TEXT),
            "target": canonical(target),
            "metadata": {
                "subfamily": "B3",
                "b3_version": B3_VERSION,
                "case": item.case,
                "faults": list(item.faults),
                "bus": bus_metadata(item.speed, rates),
                "distractor_label": item.distractor,
                "stream": stream,
                "contradicting_rate_answer": canonical(
                    answer(stream, rules_for(item.speed, rates, CONTRADICTING_RATE_FACTOR))),
                "wrong_speed_answer": canonical(answer(stream, rules_for(other_speed, rates))),
                "generator_seed": SEED,
                "generator_version": gen.GENERATOR_VERSION,
                "checker": f"bench.arinc429.conformance=={cf.CONFORMANCE_VERSION}",
            },
        })
    return records


def write_task_file(path: Path = TASK_FILE) -> Path:
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in build_records()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Reading a prompt back (used by validation, the demo and the tests)
# ---------------------------------------------------------------------------


def parse_prompt(text: str) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
    """The bus speed, stated intervals and stream rows that a B3 prompt states."""
    speed = None
    rates: dict[str, int] = {}
    stream = []
    for line in text.splitlines():
        body = line[2:] if line.startswith("- ") else line
        if m := SPEED_PREMISE_RE.match(body):
            speed = m.group(1)
        elif m := RATE_PREMISE_RE.match(body):
            rates[m.group(1)] = int(m.group(2))
        elif m := ROW_RE.match(line):
            stream.append({"label": m.group(4), "start_us": int(m.group(2)),
                           "end_us": int(m.group(3)), "word_hex": m.group(5)})
    if speed is None:
        raise B3ValidationError("prompt does not state the bus speed")
    return speed, rates, stream


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@functools.cache
def _word_by_id() -> dict[str, str]:
    return {r["word_id"]: r["word_hex"] for r in generated_words()}


def sample_problems(sample: Sample) -> list[str]:
    """Why a sample fails validation; empty when it passes."""
    meta = sample.metadata or {}
    bus, stream = meta.get("bus"), meta.get("stream")
    if not (isinstance(bus, dict) and isinstance(stream, list) and stream):
        return ["metadata does not have a bus and a stream"]
    problems = []
    speed, rates = bus.get("speed"), bus.get("stated_intervals_us") or {}
    if speed not in cf.SPEEDS or bus.get("bit_rate_bps") != cf.SPEEDS[speed]:
        return [f"bus speed {speed!r} at {bus.get('bit_rate_bps')} bit/s is not a B3 speed"]
    try:
        stated = {label: list(window(i)) for label, i in rates.items()}
    except (TypeError, ValueError) as exc:
        return [f"stated intervals {rates}: {exc}"]
    if bus.get("rate_windows_us") != stated:
        problems.append(f"rate windows {bus.get('rate_windows_us')} are not ±10% of {rates}")
    if meta.get("premises") != render_premises(speed, rates):
        problems.append("premises do not state the bus rules in the metadata")
    for n, w in enumerate(stream, start=1):
        word_hex, word_id = w.get("word_hex", ""), w.get("word_id")
        if word_id is not None and _word_by_id().get(word_id) != word_hex:
            problems.append(f"word {n} {word_hex} is not word {word_id!r} from generator seed {SEED}")
        try:
            word = int(word_hex, 16)
        except ValueError:
            problems.append(f"word {n} {word_hex!r} is not hexadecimal")
            continue
        result = decoder.decode(word)
        if result.status == "rejected" or result.fields is None:
            problems.append(f"word {n} {word_hex} does not pass the parity check")
        if decoder.read_label(word) != w.get("label"):
            problems.append(f"word {n} {word_hex} carries label {decoder.read_label(word)}, "
                            f"not {w.get('label')}")
    try:
        shown_speed, shown_rates, shown_stream = parse_prompt(str(sample.input))
    except B3ValidationError as exc:
        return [*problems, str(exc)]
    if shown_speed != speed or shown_rates != rates:
        problems.append(f"prompt states {shown_speed} speed and rates {shown_rates}, "
                        f"metadata {speed} and {rates}")
    if shown_stream != [{k: w.get(k) for k in ("label", "start_us", "end_us", "word_hex")} for w in stream]:
        problems.append("prompt rows do not match the stream in the metadata")
    if not isinstance(sample.target, str):
        return [*problems, "target must be a single JSON string"]
    try:
        expected = canonical(answer(stream, rules_for(speed, rates)))
    except cf.StreamError as exc:
        return [*problems, str(exc)]
    if sample.target != expected:
        problems.append(f"target {sample.target} does not match the conformance checker {expected}")
    return problems


def validate_samples(samples: Iterable[Sample]) -> None:
    """Raise B3ValidationError listing every sample that does not match the checker."""
    failures = [f"{s.id}: {'; '.join(p)}" for s in samples if (p := sample_problems(s))]
    if failures:
        raise B3ValidationError(
            f"B3 task set failed validation ({len(failures)} sample(s)):\n" + "\n".join(failures)
        )


def explain(sample: Sample) -> str:
    """A table of what the checker measured for each word of a sample, for demos."""
    meta = sample.metadata or {}
    bus = meta["bus"]
    rules = rules_for(bus["speed"], bus["stated_intervals_us"])
    lines = [" #  label  duration     gap  interval  window            verdict"]
    for c in cf.check_stream(meta["stream"], rules):
        win = f"{c.window_us[0]}-{c.window_us[1]}" if c.window_us else "(no rate)"
        lines.append(f"{c.number:>2}  {c.label:<5}  {c.duration_us:>8}  "
                     f"{'-' if c.gap_us is None else c.gap_us:>6}  "
                     f"{'-' if c.interval_us is None else c.interval_us:>8}  {win:<16}  "
                     f"{', '.join(c.violations) or 'ok'}")
    return "\n".join(lines)


def main() -> None:
    from bench.loader import load_tasks

    parser = argparse.ArgumentParser(description="Validate the B3 task file and show a sample.")
    parser.add_argument("tasks", nargs="?", default=str(TASK_FILE), help="B3 task file")
    parser.add_argument("--show", default=None, help="sample ID to print (default: the first)")
    args = parser.parse_args()
    dataset = load_tasks(args.tasks)
    try:
        validate_samples(dataset)
    except B3ValidationError as exc:
        raise SystemExit(f"error: {exc}") from None
    sample = next((s for s in dataset if s.id == args.show), None) if args.show else dataset[0]
    if sample is None:
        raise SystemExit(f"error: no sample {args.show!r}")
    print(f"{len(dataset)} samples match the conformance checker "
          f"(bench.arinc429.conformance=={cf.CONFORMANCE_VERSION}).\n")
    meta = sample.metadata or {}
    print(f"===== {sample.id} ({meta.get('case')}; {meta['bus']['speed']} speed)")
    print("----- premises for this task:")
    print("\n".join(f"- {p}" for p in meta.get("premises", [])))
    print("----- task text:")
    print(meta.get("task_text"))
    print("----- checker measurements (us):")
    print(explain(sample))
    print(f"----- target:              {sample.target}")
    print(f"----- contradicting rate:  {meta.get('contradicting_rate_answer')}  (every interval x2)")
    print(f"----- wrong bus speed:     {meta.get('wrong_speed_answer')}")


if __name__ == "__main__":
    main()
