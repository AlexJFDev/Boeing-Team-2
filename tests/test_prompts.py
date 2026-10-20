"""Versioned prompt templates (Issue #23)."""

import json
import shutil
from pathlib import Path

import pytest
from inspect_ai import eval

from runner import prompts
from runner.prompts import PromptTemplateError, hash_bytes, load_prompt_template
from tasks.versioned_prompt_example import versioned_prompt_example

NAME = "single_turn_example"


@pytest.fixture
def prompts_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway copy of prompts/ that the loader uses by default."""
    dest = tmp_path / "prompts"
    shutil.copytree(prompts.PROMPTS_DIR, dest)
    monkeypatch.setattr(prompts, "PROMPTS_DIR", dest)
    return dest


def test_hash_and_version_recorded_in_evallog(tmp_path: Path):
    template = load_prompt_template(NAME, 1)
    logs = eval(
        versioned_prompt_example(),
        model="mockllm/model",
        log_dir=str(tmp_path),
        temperature=0.0,
        seed=1,
    )
    assert logs[0].status == "success"
    assert logs[0].eval.task_version == 1
    meta = logs[0].eval.metadata
    assert meta is not None
    assert meta["prompt_template_name"] == NAME
    assert meta["prompt_template_version"] == 1
    assert meta["prompt_template_sha256"] == template.sha256
    assert len(template.sha256) == 64


def test_edit_without_version_bump_is_refused(prompts_copy: Path):
    path = prompts_copy / f"{NAME}.v1.txt"
    path.write_text(path.read_text() + "Be polite.\n")
    with pytest.raises(PromptTemplateError, match=rf"'{NAME}' version 1"):
        load_prompt_template(NAME, 1)


def test_refused_run_makes_no_model_call_and_writes_no_log(
    prompts_copy: Path, tmp_path: Path
):
    path = prompts_copy / f"{NAME}.v1.txt"
    path.write_text("tampered {prompt}\n")
    log_dir = tmp_path / "logs"
    with pytest.raises(PromptTemplateError):
        eval(versioned_prompt_example(), model="mockllm/model", log_dir=str(log_dir))
    assert not log_dir.exists() or not list(log_dir.glob("*.eval"))


def test_unregistered_version_is_refused(prompts_copy: Path):
    (prompts_copy / f"{NAME}.v2.txt").write_text("New.\n{prompt}\n")
    with pytest.raises(PromptTemplateError, match="not in manifest"):
        load_prompt_template(NAME, 2)


def test_registered_new_version_loads_with_new_hash(prompts_copy: Path):
    v2 = prompts_copy / f"{NAME}.v2.txt"
    v2.write_text("New.\n{prompt}\n")
    manifest_path = prompts_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[NAME]["2"] = hash_bytes(v2.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    v1 = load_prompt_template(NAME, 1)
    v2_template = load_prompt_template(NAME, 2)
    assert v2_template.sha256 != v1.sha256
