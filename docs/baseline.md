# Baseline evaluation of the ADS-B and ARINC 429 suites

WBS 3.8, issue #63. This runs every task in the protocol suites against every model in
a baseline config, then reports each sample's score per model and flags the items no
model gets right for ground-truth review.

| Suite | Task | Version | Samples | Capability scorer | Safety scorer |
| --- | --- | --- | --- | --- | --- |
| A1 | `tasks/a1_adsb_decode.py` | 1.2.0 | 29 | exact match | unsafe-if rules (6 checked) |
| A2 | `tasks/a2_cpr_position.py` | 1.2.0 | 28 | exact match | unsafe-if rules (8 checked) |
| B1 | `tasks/b1_arinc429_word_decode.py` | 1.0.0 | 27 | exact match | none |
| B2 | `tasks/b2_arinc429_ssm_interpretation.py` | 1.0.0 | 14 | exact match | none |
| B3 | `tasks/b3_arinc429_bus_conformance.py` | 1.0.0 | 16 | exact match | none |
| B4 | `tasks/b4_arinc429_fault_vs_injection.py` | 1.0.0 | 15 | B4 triage (1, 0.5, 0) | unsafe-if rules (6 checked) |

Capability and safety are reported in separate columns and never combined (CLAUDE.md
rule 2).

## Running it

You need a machine with a model server. Locally, that's Ollama with each model in the
config pulled:

```bash
ollama pull llama3.1
ollama pull qwen2.5:7b
ollama pull mistral
source .venv/bin/activate
bench baseline configs/baseline_v1.0.0.yaml
```

On Windows PowerShell, run `.venv\Scripts\Activate.ps1` instead of `source`, and run
each command on its own line.

The config (`configs/baseline_v1.0.0.yaml`) lists the six tasks, the models, and the
generate config (temperature 0, seed 42, max_tokens 1024). It is checked before any model
call, like `bench run`. A missing model list, temperature, seed or corpus version, an
unknown key, or a task file that doesn't exist exits with code 2 and writes no log.

Every task runs against every model through Inspect's `eval_set`, which does the batch,
retries failed pairs up to `retry_attempts` times, and writes one EvalLog per task and
model. Logs go to `logs/baseline/v1.0.0` (git-ignored). Running the same command again
resumes: complete pairs are skipped and the rest are retried. Exit code 0 means every
pair succeeded and 1 means at least one failed; the report is written either way.

To change the models, edit `models` in a copy of the config and give it a new
`baseline_version`, so its logs don't mix with an earlier run's.

### On VT ARC

Copy the config, list the `vllm/...` model (for example `vllm/gpt-oss-120b`, as in
`configs/arc.json`), and set `VLLM_BASE_URL` (and `VLLM_API_KEY` if the endpoint needs
it) in your `.env`. As with `bench run`, each vLLM endpoint is queried before the run: an
unreachable endpoint exits with 2, and what the endpoint lists for the model is recorded
in every log under `endpoint_models`. The Slurm side is the VT ARC smoke test's (WBS 1.21).

## The report

`bench baseline` writes `report/` into the log directory and prints a summary:

| File | Contents |
| --- | --- |
| `report.md` | Capability mean and unsafe/checked count per task and model, with the log file for each; the review list; every sample's score per model |
| `report.json` | The same, plus every model's full raw answer, for rescoring (rule 8) |
| `per_sample.csv` | One row per sample and model |

Each pair's row also records the task version, the capability scorer used, temperature,
seed and git commit from its EvalLog. Open the logs themselves with:

```bash
inspect view --log-dir logs/baseline/v1.0.0
```

Rebuild the report from existing logs, without running anything:

```bash
bench baseline --report-only logs/baseline/v1.0.0
```

## Ground-truth review

A sample that scores 0 on capability for every model is flagged. When no model gets an
item right, the item itself may be at fault: an ambiguous prompt, a missing premise, or a
bad target. The flag asks a person to check; it doesn't say the item is wrong, because a
hard item can be correct and still beat every model.

For each flagged sample, `report.json` gives the target, the task text and every model's
answer. Check, in order:

1. **The target.** Every target is re-derived from its reference before every run (the
   decoder, CPR verifier, conformance checker or triage reference), so a target that
   disagrees with its reference can't reach a run. If the reference itself is in doubt,
   check it against its source.
2. **The premises.** Is everything needed to reach the answer stated in the task? This is
   the catalogue's most common defect.
3. **The format.** Did the models answer correctly in a form the scorer doesn't accept?
   Exact match gives 0 for the right answer in the wrong format.
4. **The answers.** Do the models make the same mistake? A shared mistake that matches a
   stored trap answer (for example B2's `swapped_answer`) points to the item doing its job.

Record the decision for each flagged item. If an item has to change, bump that task's
version and build a new file; never edit a frozen task file (rule 4).

Rules for the flag:

- A sample is flagged only when every model in the run has a score for it. A
  task-model pair with no successful log is listed as missing, and its samples are not
  flagged.
- A run with one model flags every sample that model scored 0, and the report says so.
- When a pair has more than one log (after a retry), the newest successful one is used.

## Rehearsal without a model server

```bash
python scripts/rehearse_baseline.py
```

This runs the real baseline config, with every task through `eval_set`, but replaces
the models with two scripted stand-ins. Each stand-in answers with a wrong answer the
task files already store (B1's forward-read label, B2's swapped or one-table SSM, B3's
contradicting rate or wrong bus speed, B4's parity-only or values-only verdict), and with
the target elsewhere. It writes to `logs/baseline-rehearsal/v1.0.0`, never to the real
baseline directory. It flags 24 samples: B2's 8 discriminating SSM 11/00 items and all 16
B3 items. Those items are correct, so the rehearsal also shows a flag is a question, not
a verdict. The stand-ins are not models, and their scores mean nothing about any model.
