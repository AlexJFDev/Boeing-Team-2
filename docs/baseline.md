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

Use `configs/baseline_v1.3.0.yaml`. It runs the same six tasks against three models from
the [VT ARC LLM API](https://docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html):
`gpt-oss-120b`, `GLM-5.3` and `DeepSeek-V4.1-Flash`. Set these in your `.env`, and connect
to the VT VPN when off campus:

```
ARC_BASE_URL=https://llm-api.arc.vt.edu/api/v1
ARC_API_KEY=<your ARC key>
```

Then run:

```bash
bench baseline configs/baseline_v1.3.0.yaml
```

**Why it streams.** ARC caps non-streaming requests at 8000 output tokens and says to
stream longer outputs. The reasoning models need more: in v1.2.0, which used Inspect's
`vllm/` provider at the 8000 cap, all 203 blank answers stopped on `max_tokens`
(DeepSeek 86, GLM 88 and gpt-oss 29 of 129). Inspect's `vllm/` provider can't stream,
so v1.3.0 calls the models through Inspect's OpenAI-compatible provider
(`openai-api/arc/<model>`, which reads `ARC_BASE_URL` and `ARC_API_KEY`) with
`model_args: {stream: true}`, and raises `max_tokens` to 32000.

Other settings follow ARC's published limits. `max_connections` is 5, because ARC allows
10 requests per model and 20 in total per user. Reasoning effort is left at each model's
endpoint default (gpt-oss medium, GLM max, DeepSeek high), and the report should say so.
As with `vllm/` models, each `openai-api/...` endpoint is queried before the run. An
unset `ARC_BASE_URL`, an unreachable endpoint or an unlisted model exits with 2, and what
the endpoint lists for each model is recorded in every log.

`configs/baseline_v1.2.0.yaml` (the `vllm/` version, capped at 8000) is kept for the
record. For another ARC model, copy v1.3.0, change `models` and give it a new
`baseline_version`.

## The report

`bench baseline` writes `report/` into the log directory and prints a summary:

| File | Contents |
| --- | --- |
| `report.md` | Capability mean and unsafe/checked count per task and model, with the log file for each; the review list; every sample's score per model |
| `report.json` | The same, plus every model's full raw answer, for rescoring (rule 8) |
| `per_sample.csv` | One row per sample and model |

Each pair's row also records the task version, the capability scorer used, temperature,
seed, `max_tokens`, its `model_args` (for example `stream=True`), how many samples got no
answer (with stop reasons) or were cut off, and the git commit from its EvalLog. Open the logs themselves with:

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
- A sample where any model returned no text at all, or was cut off by `max_tokens`
  partway through its answer, is not flagged. A blank answer, usually
  a reasoning model cut off by `max_tokens`, says nothing about the item. It still scores 0
  in the capability mean, but the report counts it under "No answer" with the model's stop
  reason, and lists such samples separately as not reviewable. Rerun those models with a
  larger `max_tokens` before reviewing them. In v1.1.0, gpt-oss with 4096 tokens left 62 of
  129 samples blank, and 50 of the 94 flagged samples were flagged only because of that.
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
