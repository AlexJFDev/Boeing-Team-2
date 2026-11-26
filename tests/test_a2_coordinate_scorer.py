"""A2 coordinate scorer: numeric tolerance instead of exact text match (bench/scorers.py).

Also pins the reason for A2_TOLERANCE_DEG: on A2 v1.2.0, giving the older frame's
position instead of the newer one (the usual A2 mistake) must score 0 on every
global-pair task, while every correct decode scores 1.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState

from bench.adsb import a2, cpr
from bench.loader import load_tasks
from bench.scorers import (A2_COORDINATE_SCORER_VERSION, A2_TOLERANCE_DEG, ARINC429_SCORER_VERSION,
                           a2_cpr_coordinate_scorer, arinc429_bnr_bcd_capability_scorer)

SAMPLES = list(load_tasks(a2.TASK_FILE))


def by_case(case: str) -> list[Any]:
    return [s for s in SAMPLES if (s.metadata or {})["case"] == case]


def grade(completion: str, target: str, tolerance: float = A2_TOLERANCE_DEG, scorer=None) -> Score:
    state = MagicMock(spec=TaskState)
    state.sample_id, state.metadata, state.store, state.model = "s", {}, {}, None
    state.output = MagicMock()
    state.output.completion = completion
    fn = scorer or a2_cpr_coordinate_scorer(tolerance)
    score = asyncio.run(fn(state, Target(target)))
    assert isinstance(score, Score) and score.metadata is not None
    return score


def meta(score: Score) -> dict[str, Any]:
    assert score.metadata is not None
    return score.metadata


def older_frame_answer(sample: Any) -> str:
    """The position at the older frame of a global pair: the classic A2 mistake."""
    a, b = sample.metadata["frames"]
    swapped = cpr.decode_pair(cpr.parse_frame(a["hex"], b["timestamp"]), cpr.parse_frame(b["hex"], a["timestamp"]))
    assert swapped.status == "ok" and swapped.latitude is not None and swapped.longitude is not None
    return json.dumps({"latitude": round(swapped.latitude, 5), "longitude": round(swapped.longitude, 5)})


def test_every_target_scores_one() -> None:
    for s in SAMPLES:
        score = grade(str(s.target), str(s.target))
        assert score.value == 1.0, s.id


def test_formatting_rounding_and_prose_do_not_matter() -> None:
    s = by_case("global_pair")[0]
    t = json.loads(str(s.target))
    for answer in (f'The position is {{ "longitude": {t["longitude"]:.4f}, "latitude": {t["latitude"]:.4f} }}.',
                   json.dumps({"lat": t["latitude"], "lon": t["longitude"]}),
                   json.dumps({"latitude": str(t["latitude"]), "longitude": str(t["longitude"])})):
        assert grade(answer, str(s.target)).value == 1.0, answer


def test_the_older_frame_mistake_scores_zero_on_every_global_pair() -> None:
    pairs = by_case("global_pair")
    assert len(pairs) == 10
    assert all(grade(older_frame_answer(s), str(s.target)).value == 0.0 for s in pairs)
    # Why not the 0.001 deg of the earlier branch: it would give this mistake full credit.
    loose = [grade(older_frame_answer(s), str(s.target), tolerance=0.001).value for s in pairs]
    assert loose.count(1.0) == 3


def test_one_coordinate_right_is_still_the_wrong_place() -> None:
    t = json.loads(str(by_case("local_reference")[0].target))
    off = json.dumps({"latitude": t["latitude"], "longitude": t["longitude"] + 2 * A2_TOLERANCE_DEG})
    score = grade(off, json.dumps(t))
    assert score.value == 0.0 and meta(score)["lat_error_deg"] == 0.0


def test_longitude_is_compared_across_the_antimeridian() -> None:
    target = '{"latitude":10.0,"longitude":179.99995}'
    assert grade('{"latitude":10.0,"longitude":-179.99999}', target).value == 1.0
    assert grade('{"latitude":10.0,"longitude":-179.9}', target).value == 0.0


@pytest.mark.parametrize("answer, value", [
    ('{"answer":"cannot_be_determined"}', 1.0),
    ('Not enough data. {"answer": "Cannot_Be_Determined"}', 1.0),
    ('{"latitude":52.1,"longitude":4.3}', 0.0),
    ('{"answer":"reject"}', 0.0),
    ("I can't tell.", 0.0),
])
def test_no_decode_tasks_need_the_no_decode_answer(answer: str, value: float) -> None:
    target = str(by_case("stale_pair")[0].target)
    assert grade(answer, target).value == value


@pytest.mark.parametrize("answer", ['{"latitude":true,"longitude":1}', '{"latitude":"NaN","longitude":1}',
                                    '{"answer":"cannot_be_determined"}', "47.3, 7.5", ""])
def test_no_usable_position_scores_zero(answer: str) -> None:
    assert grade(answer, str(by_case("global_pair")[0].target)).value == 0.0


def test_raw_response_and_own_version_are_recorded() -> None:
    answer = "Position: " + str(by_case("global_pair")[0].target)
    score = grade(answer, str(by_case("global_pair")[0].target))
    assert meta(score)["raw_response"] == answer
    assert meta(score)["scorer_version"] == A2_COORDINATE_SCORER_VERSION
    assert meta(score)["score_type"] == "capability_only"
    assert meta(score)["tolerance_deg"] == A2_TOLERANCE_DEG


def test_arinc429_scorer_records_its_own_version_not_the_adsb_one() -> None:
    score = grade('{"label":"203","value":1}', '{"label":"203","value":1}',
                  scorer=arinc429_bnr_bcd_capability_scorer())
    assert meta(score)["scorer_version"] == ARINC429_SCORER_VERSION == "1.0.1"
