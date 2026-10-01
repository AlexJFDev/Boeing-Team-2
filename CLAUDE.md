# CLAUDE.md

Guidance for AI agents working in this repo. Humans should read `README.md`; setup
steps are in `docs/setup.md`.

## What this project is

An evaluation benchmark for AI assistants on commercial aircraft datalink analysis
(ADS-B/Mode S, ARINC 429, ARINC 664/AFDX, ACARS/VDL2), built on Inspect AI with tasks
in the METR Task Standard. It is a Virginia Tech capstone for the Boeing Aerospace Red
Team. The runner, scorers, and task corpus are mostly **not written yet**; check the
files that exist before assuming a module is there.

## Commands

Always use the project virtual environment (`.venv`). Do not install packages globally.

```bash
source .venv/bin/activate
pip install -r requirements.txt
inspect eval tasks/smoke_test.py --model mockllm/model     # no server needed
inspect eval tasks/smoke_test.py --model ollama/llama3.1   # needs local Ollama
pytest
```

Eval logs go to `logs/` (git-ignored).

## Where the requirements are

- **GitHub issues** are the source of truth for scope. Each has acceptance criteria
  written as "When X, Y happens" with a negative case. Read the issue before starting
  and make sure the work satisfies both cases.
- Issue titles carry a WBS ID (for example `1.5`) matching the product backlog.
- Labels give the area (`runner`, `sandbox`, `ads-b`, `scoring`, ...), the sprint
  (`sprint-01`, ...), and the type (`enhancement`, `documentation`, `testing`,
  `evaluation`, `release`, `chore`, `bug`).

## Rules that must hold

1. **Synthetic or sanitized data only.** Never add real operational captures, real
   registrations or flight identifiers, proprietary Boeing data, or real secrets, not
   even as examples.
2. **Never combine safety and capability into one score.** They are reported
   separately. A safety failure leaves the capability score unchanged and produces no
   composite.
3. **Reproducibility.** Anything that can change a result must be pinned or recorded:
   dependency versions, model identity and weights, sampling parameters and seed,
   prompt template hash, corpus version and hash, scorer version. Do not leave out
   `temperature` or `seed` and rely on defaults.
4. **Version bumps.** Editing a task, scorer, or prompt template requires a version bump.
   Never edit a frozen corpus or the regression baseline in place.
5. **Ground truth comes from a specification or reference decoder** (for example
   pyModeS, Wireshark/tshark, scapy), not from opinion. State premises (bus rates,
   resync rules, protocol version) in the task so the answer is determinate.
6. **Secrets.** Each developer has a personal `.env` that is never committed. Do not
   read, print, or commit it. Do not add keys to code, docs, or logs.
7. **Sandbox isolation.** Tool code that runs model-driven input belongs in the Docker
   sandbox with networking disabled and no host writes.
8. **Keep raw output.** Store the full model response alongside the parsed answer so
   results can be rescored.

## Git workflow

- `development` is the default branch; `production` holds releases and changes only by
  pull request. Do not push to either directly.
- Work on an issue branch named `{type}-{number}/{name}` where `{number}` is the
  GitHub issue number and `{name}` is lowercase words joined by hyphens. Allowed types:
  `feat`, `fix`, `docs`, `test`, `eval`, `release`, `chore`. A GitHub Action
  (`.github/workflows/branch-name.yml`) rejects other names.
- Open PRs into `development` using the template in `docs/pull_request_template.md`.
  Write `Closes #<number>` to link the issue. One approving review is required.
- Commit and push only when the user asks.

## Conventions

- Python, formatted and typed in the same style as the surrounding code. Pylance runs in
  `standard` mode.
- Pin new dependencies exactly in `requirements.txt` and explain why.
- Match the existing structure: tasks in `tasks/`, human docs in `docs/`. If a new
  directory is needed, say so in the PR.
- Prefer Inspect AI's built-in components (solvers, scorers, `EvalLog`, sandboxing,
  Ollama provider) over custom code. Add only the domain-specific parts.

## Gotchas

- Inspect's Ollama provider needs the `openai` package, which is pinned in
  `requirements.txt`.
- `mockllm/model` returns filler text, so scores of 0 against it are expected; it only
  proves the harness runs.
- Docker is required for sandbox work but may not be installed on every machine.
