"""Run configuration: sampling parameters are required, never defaulted.

CLAUDE.md rule 3: anything that can change a result must be pinned. A config
that omits ``temperature`` or ``seed`` is refused rather than filled in.
"""

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("model", "temperature", "seed")


class RunConfigError(ValueError):
    """The run config is missing a required field or has a bad value."""


@dataclass(frozen=True)
class RunConfig:
    model: str
    temperature: float
    seed: int
    max_tokens: int | None = None


def parse_run_config(data: dict) -> RunConfig:
    missing = [f for f in REQUIRED_FIELDS if data.get(f) is None]
    if missing:
        raise RunConfigError(
            f"run config is missing required field(s): {', '.join(missing)}"
        )
    if isinstance(data["temperature"], bool) or not isinstance(
        data["temperature"], (int, float)
    ):
        raise RunConfigError("run config field 'temperature' must be a number")
    if isinstance(data["seed"], bool) or not isinstance(data["seed"], int):
        raise RunConfigError("run config field 'seed' must be an integer")
    return RunConfig(
        model=data["model"],
        temperature=float(data["temperature"]),
        seed=data["seed"],
        max_tokens=data.get("max_tokens"),
    )


def load_run_config(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        return parse_run_config(json.load(f))
