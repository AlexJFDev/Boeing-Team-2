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

### Running a benchmark

Use `bench run` with a YAML run config for any run whose results you will compare or
report:

```bash
bench run configs/run.yaml
```

It refuses to start, before any model call, if the model, `temperature`, `seed` or
`corpus_version` is missing or a key is not recognised, so a run cannot go out
without the settings needed to reproduce it. It exits 0 on success, 1 if the run
fails, and 2 if the config is refused. If `bench` is not found, run
`pip install -r requirements.txt` again. Plain `inspect eval` is fine for quick
checks. See [docs/setup.md](docs/setup.md#running-from-a-run-config).

To run the ADS-B and ARINC 429 suites against several models at once and get a
per-task report, use `bench baseline configs/baseline_v1.3.0.yaml` (VT ARC models,
streamed) or `configs/baseline_v1.0.0.yaml` (local Ollama models). See
[docs/baseline.md](docs/baseline.md).

Each task family's data source, source specification, oracle and scoring tolerance is
documented in [docs/corpus.md](docs/corpus.md). Look one up with
`python -m bench.corpus_doc --family A2`; CI fails a pull request whose tasks cite a
source or oracle that page doesn't list.
Before the protocol corpus is frozen, two team members grade the same baseline answers
against the judge-clause rubric and `python -m bench.irr` records their agreement. See
[docs/irr_pilot.md](docs/irr_pilot.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `tasks/` | Inspect AI tasks; currently the install smoke test |
| `runner/` | The `bench run` and `bench baseline` commands (`runner/cli.py`), run-config pre-flight checks, versioned prompt templates |
| `bench/` | Runner package; `bench/adsb/oracle.py` is the pyModeS ground-truth oracle, `bench/adsb/cpr.py` the independent CPR position verifier, `bench/arinc429/generator.py` the seeded ARINC 429 word generator |
| `data/` | Versioned task data with manifests; `data/adsb/` holds the ADS-B frame set, `data/irr/` the inter-rater pilot records |
| `scripts/` | Build tools that generate the data, e.g. `scripts/adsb/build_frames.py` |
| `tests/` | pytest suite (`pytest` from the repo root) |
| `docs/` | Setup guide, baseline guide, protocol corpus documentation, inter-rater pilot guide and the PR template |
| `.github/workflows/` | CI on every PR: branch name, secret scan (gitleaks), unit tests (`pytest`), regression set against the baseline (`regression`), task schema (`schema`) and data boundary (`data-boundary`) |
| `requirements.txt` | Pinned Python dependencies |
| `.devcontainer/`, `.vscode/` | Codespaces / VS Code configuration |

The `bench run` runner, the first ADS-B, ARINC 429, and ACARS tasks, the scorers, and
the regression set are in place and still growing; CI runs the regression set on
every PR. The ARINC 664 tasks, and run sweeps, resume, and timeouts are still to come. Progress is
tracked in the issues.

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
- **Regression gate:** every PR runs the regression set (`python -m bench.regression
  check`) and posts the result as a comment. Any score that differs from the committed
  baseline, higher or lower, fails the `regression` check. If the change to scoring is
  intended, record a new baseline version (`python -m bench.regression record <version>`),
  point `BASELINE_VERSION` in `bench/regression.py` at it in the same PR, and say why
  (full procedure: WBS 1.29).
- **Corpus checks:** every PR also runs the task schema validator (`python -m bench.validate`)
  and the data boundary checker (`python -m bench.boundary`). A task file that is
  malformed, names a missing preamble, is not listed in `data/tasks/MANIFEST.json`, or
  was edited in place fails `schema`; data from an unapproved source or a real-looking
  registration or key fails `data-boundary`. Run both locally before pushing.
- **Secrets:** each person keeps their own `.env` file with their own keys. Never
  commit it.
- **Versioning:** changes to tasks, scorers, or prompt templates need a version bump.
  Changing them silently invalidates comparisons between runs.

## Team

Alex Fuhrig (GitHub: )
Adam Futerman (GitHub: )
Kean Jaldin Guzman (GitHub: )
Kayrene Woods (GitHub: kaykayzara)
Sahiti Srikakolapu (GitHub: )
