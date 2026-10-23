"""Versioned prompt templates (CLAUDE.md rule 4).

A template lives in an immutable file ``prompts/<name>.v<N>.txt``. Changing the
text means adding a new file with the next version, never editing the old one.
``prompts/manifest.json`` pins the sha256 of every released name+version. Loading
a template whose hash does not match the manifest, or that has no entry, is
refused before any model call.

A template is a system message, never a rewrite of the user message: the task text
stays identical across templates, so the hash covers only the part that varies.
Inspect supplies the solvers and logs the template text; this module only adds
the version, the hash and the refusal.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.solver import Solver, generate, system_message

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
MANIFEST_NAME = "manifest.json"


class PromptTemplateError(ValueError):
    """The template is missing, unregistered, or edited without a version bump."""


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    sha256: str
    text: str

    def metadata(self) -> dict[str, str | int]:
        """Fields recorded in the EvalLog (``eval.metadata``)."""
        return {
            "prompt_template_name": self.name,
            "prompt_template_version": self.version,
            "prompt_template_sha256": self.sha256,
        }

    def solver(self) -> list[Solver]:
        """Inspect's own solvers: the template as a system message, then ``generate()``.

        The user message is the task's own text, unchanged. Inspect fills ``{...}``
        in the text from sample metadata, so write a literal brace as ``{{`` or ``}}``.
        """
        return [system_message(self.text), generate()]


def hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_prompt_template(
    name: str, version: int, prompts_dir: str | Path | None = None
) -> PromptTemplate:
    directory = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    ref = f"prompt template '{name}' version {version}"
    path = directory / f"{name}.v{version}.txt"
    if not path.is_file():
        raise PromptTemplateError(f"{ref}: file {path.name} not found")
    raw = path.read_bytes()
    digest = hash_bytes(raw)

    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        manifest = {}
    expected = manifest.get(name, {}).get(str(version))
    if expected is None:
        raise PromptTemplateError(
            f"{ref} is not in {MANIFEST_NAME}; register its sha256 ({digest})"
        )
    if digest != expected:
        raise PromptTemplateError(
            f"{ref} was edited without a version bump: sha256 {digest} does not "
            f"match {MANIFEST_NAME} ({expected}). Restore the file or add "
            f"{name}.v{version + 1}.txt and register it."
        )
    return PromptTemplate(name, version, digest, raw.decode("utf-8"))
