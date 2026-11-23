"""Checks for the ARINC 429 bus conformance checker (WBS 3.5, issue #60).

The checker is the ground truth for B3, so each rule is tested at its edges: a value
exactly on a stated limit conforms, and one microsecond past it does not.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from bench.arinc429 import conformance as cf

HIGH = cf.BusRules(cf.SPEEDS["high"], {"203": (56_250, 68_750)})
LOW = cf.BusRules(cf.SPEEDS["low"], {"203": (56_250, 68_750)})


def stream(*rows: tuple[str, int, int]) -> list[tuple[str, int, int]]:
    return list(rows)


def test_derived_timing_matches_the_sources() -> None:
    # 100 kbit/s and 12.5 kbit/s, 32-bit words, at least 4 bit times of null between words.
    assert HIGH.bit_time_us == 10 and HIGH.word_time_us == 320 and HIGH.min_gap_us == 40
    assert LOW.bit_time_us == 80 and LOW.word_time_us == 2560 and LOW.min_gap_us == 320
    assert HIGH.word_time_limits_us == (Fraction(3168, 10), Fraction(3232, 10))
    assert LOW.word_time_limits_us == (Fraction(25344, 10), Fraction(25856, 10))


def test_clean_stream_has_no_violations() -> None:
    rows = stream(("203", 0, 320), ("206", 1000, 1320), ("203", 62_500, 62_820))
    assert cf.violations(rows, HIGH) == []


@pytest.mark.parametrize(("duration", "ok"), [(316, False), (317, True), (320, True), (323, True), (324, False)])
def test_word_timing_high_speed(duration: int, ok: bool) -> None:
    assert (cf.violations(stream(("203", 0, duration)), HIGH) == []) is ok


@pytest.mark.parametrize(("duration", "ok"), [(2534, False), (2535, True), (2585, True), (2586, False)])
def test_word_timing_low_speed(duration: int, ok: bool) -> None:
    assert (cf.violations(stream(("203", 0, duration)), LOW) == []) is ok


@pytest.mark.parametrize(("gap", "ok"), [(40, True), (39, False), (0, False), (-100, False)])
def test_gap_is_at_least_four_bit_times(gap: int, ok: bool) -> None:
    rows = stream(("206", 0, 320), ("213", 320 + gap, 640 + gap))
    expected = [] if ok else [[2, "gap"]]
    assert cf.violations(rows, HIGH) == expected


def test_gap_applies_across_labels_and_names_the_later_word() -> None:
    rows = stream(("206", 0, 320), ("213", 330, 650), ("206", 2000, 2320))
    assert cf.violations(rows, HIGH) == [[2, "gap"]]


@pytest.mark.parametrize(("interval", "rule"), [
    (56_249, "rate_fast"), (56_250, None), (62_500, None), (68_750, None), (68_751, "rate_slow"),
])
def test_rate_window_is_inclusive(interval: int, rule: str | None) -> None:
    rows = stream(("203", 0, 320), ("203", interval, interval + 320))
    assert cf.violations(rows, HIGH) == ([] if rule is None else [[2, rule]])


def test_rate_is_measured_start_to_start_per_label() -> None:
    rows = stream(("203", 0, 320), ("206", 30_000, 30_320), ("203", 125_000, 125_320),
                  ("203", 187_500, 187_820))
    assert cf.violations(rows, HIGH) == [[3, "rate_slow"]]


def test_label_without_a_stated_rate_is_not_rate_checked() -> None:
    rows = stream(("213", 0, 320), ("213", 1_000, 1_320), ("213", 900_000, 900_320))
    assert cf.violations(rows, HIGH) == []


def test_one_word_can_break_several_rules_sorted_by_name() -> None:
    rows = stream(("206", 0, 320), ("203", 330, 700), ("203", 10_000, 10_320))
    assert cf.violations(rows, HIGH) == [[2, "gap"], [2, "word_timing"], [3, "rate_fast"]]


def test_the_same_stream_gets_a_different_verdict_under_different_stated_rates() -> None:
    rows = stream(("203", 0, 320), ("203", 31_250, 31_570), ("203", 62_500, 62_820))
    at_31 = cf.BusRules(100_000, {"203": (28_125, 34_375)})
    assert cf.violations(rows, at_31) == []
    assert cf.violations(rows, HIGH) == [[2, "rate_fast"], [3, "rate_fast"]]


def test_unordered_or_impossible_streams_are_refused() -> None:
    with pytest.raises(cf.StreamError, match="start order"):
        cf.violations(stream(("203", 1000, 1320), ("206", 0, 320)), HIGH)
    with pytest.raises(cf.StreamError, match="ends at or before"):
        cf.violations(stream(("203", 1000, 1000)), HIGH)


def test_accepts_dict_rows() -> None:
    rows = [{"label": "203", "start_us": 0, "end_us": 330, "word_hex": "0x00000000"}]
    assert cf.violations(rows, HIGH) == [[1, "word_timing"]]


def test_format_us() -> None:
    assert cf.format_us(Fraction(3168, 10)) == "316.8"
    assert cf.format_us(320) == "320"
    assert cf.format_us(Fraction(125, 4)) == "31.25"


def test_checker_is_independent_of_the_b3_builder() -> None:
    source = (Path(cf.__file__)).read_text()
    assert "b3" not in [line.split()[1].split(".")[-1] for line in source.splitlines()
                        if line.startswith(("import ", "from "))]
    assert "bench.arinc429.b3" not in source.replace("``bench/arinc429/b3.py``", "")
