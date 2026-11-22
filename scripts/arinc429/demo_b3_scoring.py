"""Demo for task 3.5 (issue #60): B3 verdicts follow the stated bus rules.

Loads B3 task b3-03, whose premises state the bus speed, the gap rule, word timing and
each label's transmission rate, and prints the conformance checker's measurements to
show the ground-truth verdict follows those numbers. Then runs the real B3 task with
``inspect eval`` three times. The model is a scripted stand-in (Inspect's mockllm with
custom outputs) that reads the rules and stream from the prompt, so no model server is
needed and the result is the same every time:

- correct: the stream checked against the rules the prompt states
- contradicting rate: every stated label interval doubled, as if the model brought its
  own idea of the label's rate. Good intervals read as too fast, and a dropped word
  looks normal.
- wrong bus speed: the other bus speed, so every word's duration is wrong

Correct answers should score 1 everywhere, and both wrong ones 0 everywhere.

Usage (from the repo root):
    python scripts/arinc429/demo_b3_scoring.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, ModelUsage, get_model  # noqa: E402

from bench.arinc429 import b3, conformance  # noqa: E402
from bench.arinc429.b1 import canonical  # noqa: E402
from bench.loader import load_tasks  # noqa: E402
from tasks.b3_arinc429_bus_conformance import b3_arinc429_bus_conformance  # noqa: E402

SHOW = "b3-03"
KINDS = ("correct", "contradicting rate", "wrong bus speed")


def answer_for(prompt: str, kind: str) -> str:
    speed, rates, stream = b3.parse_prompt(prompt)
    if kind == "contradicting rate":
        rules = b3.rules_for(speed, rates, b3.CONTRADICTING_RATE_FACTOR)
    elif kind == "wrong bus speed":
        rules = b3.rules_for("low" if speed == "high" else "high", rates)
    else:
        rules = b3.rules_for(speed, rates)
    return canonical(b3.answer(stream, rules))


def scripted_model(kind: str):
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        output = ModelOutput.from_content("mockllm/model", answer_for(messages[-1].text, kind))
        # A scripted answer has no real token count. Setting usage also stops mockllm
        # estimating one, which would download a tokenizer.
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output
    return get_model("mockllm/model", custom_outputs=respond)


def main() -> int:
    print(f"B3 v{b3.B3_VERSION}, targets from bench.arinc429.conformance=="
          f"{conformance.CONFORMANCE_VERSION}\n")
    sample = next(s for s in load_tasks(b3.TASK_FILE) if s.id == SHOW)
    meta = sample.metadata or {}
    print(f"===== {SHOW} ({meta['case']}): the rules this task states")
    print("\n".join(f"- {p}" for p in meta["premises"][:-1]))
    print("\nWhat the checker measures against those rules (times in us):")
    print(b3.explain(sample))
    print(f"\ntarget: {sample.target}\n")

    results: dict[str, dict[str, float]] = {}
    answers: dict[str, str] = {}
    for kind in KINDS:
        with tempfile.TemporaryDirectory() as log_dir:
            [log] = inspect_eval(b3_arinc429_bus_conformance(), model=scripted_model(kind),
                                 log_dir=log_dir, display="none")
            if log.status != "success" or log.samples is None:
                print(f"eval failed: {log.error}")
                return 1
            results[kind] = {str(s.id): float(next(iter((s.scores or {}).values())).value)
                             for s in log.samples}
            answers[kind] = next(s.output.completion for s in log.samples if s.id == SHOW)

    print(f"Answers to {SHOW}, scored by inspect eval:")
    for kind in KINDS:
        print(f"  {kind + ':':<20} {answers[kind]}  -> score {results[kind][SHOW]}")
    print("\nThe dropped label 203 word leaves a 62.8 ms interval: too slow at the stated 31.25 ms,")
    print("but normal to a model that takes label 203 to be sent every 62.5 ms.\n")

    print(f"{'answers':<20} {'mean score, all ' + str(len(results['correct'])) + ' tasks':>26}")
    for kind in KINDS:
        scores = results[kind]
        print(f"{kind:<20} {sum(scores.values()) / len(scores):>26.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
