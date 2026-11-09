"""Guardrail profiles, selected in the run config (WBS 1.17, issue #25).

A guardrail profile is a named, versioned set of defensive settings, run as an
experimental variable: the same task under ``none`` (the baseline) and under a
profile, with the result recorded per condition. This module covers selection,
pinning and recording. What a real profile contains (hardened prompt, input and
output filters) is WBS 6.9.

A text profile is a system message stored as the versioned prompt template
``prompts/guardrail_<name>.v<N>.txt``, so the rules from ``runner.prompts`` apply
unchanged: it must be in ``prompts/manifest.json`` and an edited file is refused. The
user message is never touched, so the task text is identical across profiles.

``none`` is a profile too. It has no file and adds no system message, but it is
recorded in the log so every EvalLog says which condition it came from.
"""

from dataclasses import dataclass

from inspect_ai.solver import Solver, system_message

from runner.prompts import PromptTemplateError, load_prompt_template

NONE = "none"
FILE_PREFIX = "guardrail_"


class GuardrailProfileError(ValueError):
    """The profile is unknown, unregistered, or edited without a version bump."""


@dataclass(frozen=True)
class GuardrailProfile:
    name: str
    version: int
    sha256: str | None
    text: str | None

    def metadata(self) -> dict[str, str | int | None]:
        """Fields recorded in the EvalLog (``eval.metadata``)."""
        return {
            "guardrail_profile_name": self.name,
            "guardrail_profile_version": self.version,
            "guardrail_profile_sha256": self.sha256,
        }

    def pre_solvers(self) -> list[Solver]:
        """Solvers to run before the task's own: the profile as a system message.

        Empty for ``none``. Use as ``solver=[*profile.pre_solvers(), generate()]``,
        or ahead of any other solver.
        """
        return [system_message(self.text)] if self.text is not None else []


def load_guardrail_profile(name: str, version: int = 1) -> GuardrailProfile:
    if name == NONE:
        return GuardrailProfile(NONE, 0, None, None)
    try:
        template = load_prompt_template(f"{FILE_PREFIX}{name}", version)
    except PromptTemplateError as e:
        raise GuardrailProfileError(f"guardrail profile '{name}' version {version}: {e}") from e
    return GuardrailProfile(name, version, template.sha256, template.text)
