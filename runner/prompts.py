"""Versioned prompt templates (CLAUDE.md rule 4).

A template lives in an immutable file ``prompts/<name>.v<N>.txt``. Changing the
text means adding a new file with the next version, never editing the old one.
``prompts/manifest.json`` pins the sha256 of every released name+version. Loading
a template whose hash does not match the manifest, or that has no entry, is
refused before any model call.

Inspect supplies the solvers and logs the template text; this module only adds
the version, the hash and the refusal.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.solver import Solver, generate, prompt_template, system_message

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

    def solver(self, kind: str = "prompt") -> list[Solver]:
        """Inspect's own solvers: the template, then ``generate()``.

        ``kind="prompt"`` wraps the user prompt (use ``{prompt}`` in the text);
        ``kind="system"`` sets the system message.
        """
        if kind == "prompt":
            return [prompt_template(self.text), generate()]
        if kind == "system":
            return [system_message(self.text), generate()]
        raise ValueError(f"kind must be 'prompt' or 'system', not {kind!r}")


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
