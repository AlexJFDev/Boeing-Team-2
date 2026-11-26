"""Rehearse the 3.8 baseline without a model server (issue #63).

    python scripts/rehearse_baseline.py

Runs the real baseline config (configs/baseline_v1.0.0.yaml): every task, through
Inspect's ``eval_set``, with the same generate config. The one change is the models:
two scripted stand-ins (Inspect's mockllm with custom outputs) replace the Ollama
models, so this runs anywhere and gives the same result every time. Logs and the
report go to logs/baseline-rehearsal/v1.0.0, never to the real baseline directory.

The stand-ins are not models. Each answers with a wrong answer the task files already
store, where a family stores one, and with the target everywhere else:

    standin-trap-a   B1: the label read forward (naive_label); B2: SSM tables swapped;
                     B3: every stated rate doubled; B4: parity checked, values ignored
    standin-trap-b   B2: the BNR table for both words; B3: the other bus speed;
                     B4: values checked, parity ignored

So the report shows per-task scores that differ by model, and the review list flags the
items both stand-ins score 0 on: every B3 item. B2's traps get partial credit from the
field-level scorer (0.5 and 0.75 on the discriminating items), so they are not flagged. Those items are
correct, which is the point of the review: a flag asks a person to look, it does not
say the item is wrong.

The real run is ``bench baseline configs/baseline_v1.0.0.yaml`` on a machine with
Ollama (docs/baseline.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.model import Model, ModelOutput, ModelUsage, get_model  # noqa: E402

from bench import baseline_report  # noqa: E402
from bench.arinc429.b1 import canonical  # noqa: E402
from bench.loader import load_tasks  # noqa: E402
from runner import baseline  # noqa: E402
from tasks.a1_adsb_decode import TASK_FILE as A1  # noqa: E402
from tasks.a2_cpr_position import TASK_FILE as A2  # noqa: E402
from tasks.b1_arinc429_word_decode import TASK_FILE as B1  # noqa: E402
from tasks.b2_arinc429_ssm_interpretation import TASK_FILE as B2  # noqa: E402
from tasks.b3_arinc429_bus_conformance import TASK_FILE as B3  # noqa: E402
from tasks.b4_arinc429_fault_vs_injection import TASK_FILE as B4  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "baseline_v1.0.0.yaml"
LOG_ROOT = REPO_ROOT / "logs" / "baseline-rehearsal"

# Stand-in -> the stored wrong answer it gives, by metadata key, per subfamily.
TRAPS = {
    "standin-trap-a": {"B2": "swapped_answer", "B3": "contradicting_rate_answer", "B4": "parity_only_answer"},
    "standin-trap-b": {"B2": "bnr_table_answer", "B3": "wrong_speed_answer", "B4": "value_only_answer"},
}


def samples_by_prompt() -> dict[str, Sample]:
    found: dict[str, Sample] = {}
    for path in (A1, A2, B1, B2, B3, B4):
        for sample in load_tasks(path):
            found[str(sample.input)] = sample
    return found


def answer(sample: Sample, name: str) -> str:
    meta = sample.metadata or {}
    sub = meta.get("subfamily")
    if name == "standin-trap-a" and sub == "B1" and isinstance(sample.target, str):
        target = json.loads(sample.target)
        if "label" in target:
            return canonical({**target, "label": meta["naive_label"]})
    key = TRAPS[name].get(str(sub))
    if key and key in meta:
        return str(meta[key])
    return str(sample.target)


def standin(name: str, prompts: dict[str, Sample]) -> Model:
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        sample = prompts[messages[-1].text]
        output = ModelOutput.from_content(f"mockllm/{name}", answer(sample, name))
        # A scripted answer has no real token count. Setting usage also stops mockllm
        # estimating one, which would download a tokenizer.
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output
    return get_model(f"mockllm/{name}", custom_outputs=respond)


def main() -> int:
    cfg = baseline.load_and_check(CONFIG)
    prompts = samples_by_prompt()
    log_dir = baseline.log_dir_for(cfg, LOG_ROOT)
    print(f"Rehearsing {CONFIG.name}: {len(cfg['tasks'])} tasks x {len(TRAPS)} scripted stand-ins "
          f"(in place of {', '.join(cfg['models'])})\nLogs: {log_dir}\n")
    ok = baseline.run(cfg, log_dir, models=[standin(n, prompts) for n in TRAPS], display="none")
    report = baseline_report.write_report(log_dir)
    print(baseline_report.summary_text(report))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
