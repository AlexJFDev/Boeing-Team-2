"""Checks for the ARINC 429 fault-versus-injection triage reference (WBS 3.6, issue #61).

The reference is the ground truth for B4, so each rule is tested on hand-built streams:
a change exactly at the limit is genuine and one resolution step past it is injected,
a parity failure is a fault however its value looks, and faults count toward k.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from bench.arinc429 import generator as gen
from bench.arinc429 import triage as tr

ALT = gen.PROFILES_BY_LABEL[0o203]  # pressure altitude, 1 ft resolution
TX = tr.Transmitter("203", 1, "11", Fraction(10))


def word(value: float, sdi: int = 1, ssm: str = "11", profile: gen.LabelProfile = ALT) -> int:
    return gen.make_word(profile, value, sdi, ssm)[0]


def stream(*values: float) -> list[int]:
    return [word(v) for v in values]


def test_clean_stream() -> None:
    assert tr.triage(stream(1000, 1005, 998, 1003), TX) == {"verdict": "none", "words": []}


@pytest.mark.parametrize(("step", "status"), [(10, "genuine"), (11, "injected"), (-10, "genuine"), (-11, "injected")])
def test_limit_is_inclusive(step: int, status: str) -> None:
    findings = tr.judge_stream(stream(1000, 1000 + step), TX)
    assert findings[1].status == status


def test_injected_word_passes_parity_but_breaks_the_limit() -> None:
    words = stream(1000, 1004, 1600, 1008)
    assert all(gen.parity_ok(w) for w in words)
    assert tr.triage(words, TX) == {"verdict": "injection", "words": [3]}


def test_a_high_bit_fault_is_a_fault_not_an_injection() -> None:
    words = stream(1000, 1004, 1008, 1012)
    words[2] = gen.flip_bit(words[2], 28)  # a huge jump in value, but parity now fails
    assert tr.triage(words, TX) == {"verdict": "hardware_fault", "words": [3]}


def test_faulted_and_injected_words_count_toward_k() -> None:
    words = stream(1000, 1010, 1020, 1030)
    words[1] = gen.flip_bit(words[1], 32)
    words[2] = gen.flip_bit(words[2], 32)
    # word 4 is 3 transmissions after word 1: 30 ft is exactly the limit
    assert [f.status for f in tr.judge_stream(words, TX)] == ["genuine", "fault", "fault", "genuine"]


def test_later_words_are_compared_with_the_last_genuine_word() -> None:
    words = stream(1000, 1500, 1005, 1010)  # word 2 injected; 3 and 4 follow the true track
    assert tr.triage(words, TX) == {"verdict": "injection", "words": [2]}


def test_an_injection_outranks_faults() -> None:
    words = stream(1000, 1004, 1700, 1008)
    words[1] = gen.flip_bit(words[1], 15)
    assert tr.triage(words, TX) == {"verdict": "injection", "words": [3]}


@pytest.mark.parametrize("change", ["sdi", "ssm", "label"])
def test_good_parity_with_the_wrong_sdi_ssm_or_label_is_injected(change: str) -> None:
    other = {"sdi": word(1004, sdi=2), "ssm": word(1004, ssm="10"),
             "label": word(1004, profile=gen.PROFILES_BY_LABEL[0o204])}[change]
    assert tr.triage([word(1000), other, word(1008)], TX) == {"verdict": "injection", "words": [2]}


def test_word_one_must_be_genuine() -> None:
    with pytest.raises(tr.StreamError, match="word 1"):
        tr.triage([gen.flip_bit(word(1000), 32), word(1004)], TX)
    with pytest.raises(tr.StreamError, match="word 1"):
        tr.triage([word(1000, sdi=2), word(1004)], TX)


def test_bcd_values() -> None:
    gs = gen.PROFILES_BY_LABEL[0o012]
    tx = tr.Transmitter("012", 0, "00", Fraction(2))
    words = [gen.make_word(gs, v, 0, "00")[0] for v in (250, 252, 290, 251)]
    assert tr.triage(words, tx) == {"verdict": "injection", "words": [3]}


def test_reference_is_independent_of_the_b4_builder() -> None:
    source = Path(tr.__file__).read_text()
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("b4" in line for line in imports)
