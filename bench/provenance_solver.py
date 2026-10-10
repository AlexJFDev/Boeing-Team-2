"""Solver that records corpus and prompt-template hashes in each sample (issue #16)."""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver

from bench.provenance import STORE_KEY, sha256_text


@solver
def record_provenance(corpus_hash: str, template: str) -> Solver:
    """Write the corpus hash and the prompt template hash into the sample store.

    Put this first in a task's solver chain. `template` must be the same string
    the task hands to Inspect's prompt_template solver, so the recorded hash
    matches the prompt the model actually saw.
    """
    template_hash = sha256_text(template)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set(
            STORE_KEY,
            {"corpus_hash": corpus_hash, "prompt_template_hash": template_hash},
        )
        return state

    return solve