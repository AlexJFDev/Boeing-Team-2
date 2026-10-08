# Boeing Team 2: Aerospace AI Cyber Benchmark

A Virginia Tech capstone project, sponsored by the Boeing Aerospace Red Team.

AI assistants are entering aerospace engineering and cyber analysis workflows faster
than we have ways to evaluate them. General capability benchmarks say little about
narrow, high-stakes domain work. This project builds a way to measure that, and then
uses it.

## What we are building

1. **A reusable benchmark:** a versioned set of aviation cyber analysis tasks, scorers,
   and a runner that executes models against them and records results so someone who was
   not involved can reproduce them.
2. **A comparative analysis:** how different combinations of model, prompt, and
   guardrail configuration perform on those tasks, ending in a plain-language
   recommendation.

The benchmark is the part that outlives the project. A comparison of today's models
ages quickly, while a well-built instrument can be re-run against models that do not
exist yet.

### Scope

- **Domain:** commercial aircraft datalinks in four task families:
  - A: ADS-B / Mode S
  - B: ARINC 429
  - C: ARINC 664 (AFDX)
  - D: ACARS / VDL2
- **Two scores, never combined:** *capability* (is the answer right?) and *safety* (did
  the system obey injected instructions or give unsafe advice?). An accurate decoder
  that obeys a prompt injection still fails safety.
- **Data:** synthetic or sanitized only. No operational captures, proprietary systems,
  or production access.
- **Compute:** none provided. Open-weight models run locally with Ollama or on
  VT Advanced Research Computing (ARC).

## How it works

The runner is built on [Inspect AI](https://inspect.aisi.org.uk/), with tasks defined
in the [METR Task Standard](https://github.com/METR/task-standard) format and tool
execution sandboxed in Docker. Every task attempt writes a provenance record (corpus
version, model, sampling settings, prompt hash, raw output, scores) so results can be
audited and reproduced.

Components: task loader, model adapter, scaffold/solver, sandboxed tools,
capability and safety scorers, provenance logs, aggregation and reporting, and a
regression task set that gates every change.

## Getting started

Full instructions, including Ollama, Docker, and troubleshooting, are in
[docs/setup.md](docs/setup.md). The short version:

```bash
git clone git@github.com:AlexJFDev/Boeing-Team-2.git
cd Boeing-Team-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
inspect eval tasks/smoke_test.py --model mockllm/model
```

## Repository layout

| Path | Contents |
| --- | --- |
| `tasks/` | Inspect AI tasks; currently the install smoke test |
| `bench/` | Runner package; `bench/adsb/oracle.py` is the pyModeS ground-truth oracle, `bench/adsb/cpr.py` the independent CPR position verifier |
| `data/` | Versioned task data with manifests; `data/adsb/` holds the ADS-B frame set |
| `scripts/` | Build tools that generate the data, e.g. `scripts/adsb/build_frames.py` |
| `tests/` | pytest suite (`pytest` from the repo root) |
| `docs/` | Setup guide and the PR template |
| `.github/workflows/` | CI: branch-name check |
| `requirements.txt` | Pinned Python dependencies |
| `.devcontainer/`, `.vscode/` | Codespaces / VS Code configuration |

The runner, task corpus, scorers, and CI regression gate are still to come. Their
progress is tracked in the issues.

## Contributing

Work is planned as GitHub issues, one per task, and tracked on the team project board.
Sprint 1 runs Sep 27 to Oct 11, 2026; the full plan runs to the Dec 6 symposium.

- **Branches:** `development` is the default and holds work that has passed the
  regression set. `production` holds released versions and only changes by pull request.
  Do your work on an issue branch named `{type}-{number}/{name}`, where `{number}` is
  the issue number, for example `feat-26/run-config-schema`. Allowed types: `feat`,
  `fix`, `docs`, `test`, `eval`, `release`, `chore`. A CI check rejects PRs from
  other names.
- **Pull requests:** open a PR into `development` and fill in the template. One approving
  review is required.
- **Secrets:** each person keeps their own `.env` file with their own keys. Never
  commit it.
- **Versioning:** changes to tasks, scorers, or prompt templates need a version bump.
  Changing them silently invalidates comparisons between runs.

## Team

Alex Fuhrig, Adam Futerman, Kean Jaldin Guzman, Kayrene Woods, Sahiti Srikakolapu.
