"""Demo for the safety telemetry aggregator (Task 6.3, issue #82). No server or Docker needed.

    python scripts/demo_safety_telemetry.py

Runs real Inspect evals on mockllm/model, then aggregates them twice:

1. A finished run with refusal rate and injection-compliance rate per model and
   guardrail profile (D2 under no guardrail and under the hardened example prompt).
2. The same logs plus a task with no safety label (smoke_test, through
   `bench run`), which is excluded and named in one warning; the rates
   do not change.

Every result here is SCRIPTED or FILLER, not a model's behaviour:
- safe / obedient: bench/acars/d2_scripted.py personas (code, answers fixed in advance)
- refuser: a persona defined only in this script, which declines every prompt
- filler: mockllm's default output, so its rates are uninformative
The scripted personas ignore the system message, so their rows under the two
guardrail profiles are identical by construction.

Logs go to logs/safety_telemetry_demo/<UTC>/ and reports to logs/safety_telemetry/
(both git-ignored).
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from bench import safety_telemetry as st  # noqa: E402
from bench.acars.d2_harness import build_task  # noqa: E402
from bench.acars.d2_run import run as d2_run  # noqa: E402
from runner import preflight  # noqa: E402

GUARDRAIL_FILE = "prompts/guardrail_hardened_example.v1.txt"
REFUSAL_TEXT = "I'm sorry, but I can't help with that request."
SEED, TEMPERATURE = 42, 0.0


def refuser_model() -> Model:
    """mockllm/model that declines every prompt (demo-only scripted persona)."""
    def outputs(messages: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
                config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content(model="mockllm", content=REFUSAL_TEXT)

    config = GenerateConfig(temperature=TEMPERATURE, seed=SEED)
    return get_model("mockllm/model", config=config, memoize=False, custom_outputs=outputs)


def bench_run(config_text: str, name: str, run_dir: Path) -> None:
    """`bench run` on a config written to the demo folder; exits on refusal or failure."""
    cfg = run_dir / f"{name}.yaml"
    cfg.write_text(config_text, encoding="utf-8")
    code = preflight.main([str(cfg)])
    if code != 0:
        raise SystemExit(f"bench run {cfg} exited {code}")


def logs_in(run_dir: Path, before: set[Path]) -> list[Path]:
    return sorted(set(run_dir.glob("*.eval")) - before)


def main() -> int:
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = Path("logs") / "safety_telemetry_demo" / stamp
    run_dir.mkdir(parents=True)
    os.environ["INSPECT_LOG_DIR"] = str(run_dir)  # where `bench run` writes its logs
    os.environ.setdefault("INSPECT_DISPLAY", "none")

    print(f"Running demo evals on mockllm/model into {run_dir} ...")
    labeled: list[Path] = []
    for persona in ("safe", "obedient"):
        for guardrail in (None, GUARDRAIL_FILE):
            log = d2_run("mockllm/model", script=persona, guardrail_file=guardrail, log_dir=str(run_dir))
            labeled.append(Path(log.location))

    refuser = refuser_model()
    meta = {"scripted_persona": "refuser"}
    refusal_logs: list[EvalLog] = [
        *inspect_eval(build_task(temperature=TEMPERATURE, seed=SEED), model=refuser, log_dir=str(run_dir),
                      display="none", metadata=meta),
        *inspect_eval("tasks/a1_adsb_decode.py", model=refuser, log_dir=str(run_dir), display="none",
                      metadata=meta, limit=10, temperature=TEMPERATURE, seed=SEED),
    ]
    labeled += [Path(log.location) for log in refusal_logs]

    before = set(run_dir.glob("*.eval"))
    bench_run(
        "task: tasks/a1_adsb_decode.py\nmodel: mockllm/model\n"
        "generate_config:\n  temperature: 0\n  seed: 42\nmetadata:\n  corpus_version: synthetic-0.1.0\n",
        "a1_filler", run_dir)
    labeled += logs_in(run_dir, before)

    before = set(run_dir.glob("*.eval"))
    # configs/guardrail_example.yaml is not used: on a Windows checkout with core.autocrlf=true
    # the profile file's hash does not match prompts/manifest.json and bench run refuses it
    # (the known test_prompts.py line-ending issue, fixed separately).
    bench_run(Path("configs/run.yaml").read_text(encoding="utf-8"), "smoke_test", run_dir)
    unlabeled = logs_in(run_dir, before)

    banner = ("NOTE: every row is SCRIPTED or FILLER. safe/obedient = bench/acars/d2_scripted.py personas; "
              "refuser = persona defined in this demo script; scripted_persona None = mockllm filler output "
              "(A1 via bench run; rates uninformative). Scripted personas ignore the guardrail, so their rows "
              "match across profiles. 'unrecorded' = the refuser's A1 run was a plain inspect eval, which records "
              "no guardrail; it is kept apart from the 'none' baseline on purpose. refusal_rate covers A1 and "
              "D2 injected attempts only; control_refusal_rate is D2 controls (over-refusal of a benign prompt), "
              "None for groups with no D2 controls.")

    print("\n" + "=" * 100)
    print("Demo 1: finished run, refusal rate and injection-compliance rate per model and guardrail profile")
    print("=" * 100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report1 = st.aggregate(labeled)
    print(st.format_report(report1))
    print(f"\nwarnings emitted: {len([w for w in caught if issubclass(w.category, st.SafetyTelemetryWarning)])}")
    print(banner)

    print("\n" + "=" * 100)
    print("Demo 2: same logs plus a task with no safety label (smoke_test via bench run)")
    print("=" * 100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report2 = st.aggregate([*labeled, *unlabeled])
    for w in caught:
        if issubclass(w.category, st.SafetyTelemetryWarning):
            print(f"SafetyTelemetryWarning: {w.message}\n")
    print(st.format_report(report2))
    pd.testing.assert_frame_equal(report1.rates, report2.rates)
    print("\nRates identical with and without the unlabeled tasks: yes")
    print(banner)

    out1 = report1.write(stem=f"demo_labeled_{stamp}")
    out2 = report2.write(stem=f"demo_with_unlabeled_{stamp}")
    print("\nReports:")
    for path in (*out1, *out2):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
