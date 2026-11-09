"""Demo for task 3.4 (issue #59): B2 SSM answers scored against the reference decoder.

Runs the real B2 task with ``inspect eval``. The model is a scripted stand-in
(Inspect's mockllm with custom outputs), so no model server is needed and the result
is the same every time. Each sample is run twice, once per kind of answer:

- correct: the reference decoder's answer
- swapped: each word's SSM read with the other encoding's table, so 11 on a BNR word
  becomes "minus" and 11 on a BCD word becomes "normal_operation" (catalogue B2)

A third pass reads every SSM with the BNR table only, the "one memorised table"
failure the catalogue describes. Correct answers should pass everywhere. The wrong
answers fail on the discriminating items (SSM 11 and 00) and pass on the controls
(SSM 01 and 10), where both tables agree.

Usage (from the repo root):
    python scripts/arinc429/demo_b2_scoring.py
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, ModelUsage, get_model  # noqa: E402

from bench.arinc429 import b2, decoder  # noqa: E402
from bench.arinc429.b1 import canonical  # noqa: E402
from tasks.b2_arinc429_ssm_interpretation import b2_arinc429_ssm_interpretation  # noqa: E402

SHOW = ("b2-01", "b2-05", "b2-09", "b2-13")  # SSM 11, SSM 00, control 01, parity fault
KINDS = {"correct": None, "swapped": "other", "BNR table only": "BNR"}


def scripted_model(table: str | None):
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        words = [int(w, 16) for w in re.findall(r"Word \d: (0x[0-9A-F]{8})", messages[-1].text)]
        answer = b2.reference_answer(words) if table is None else b2.table_answer(words, table)
        output = ModelOutput.from_content("mockllm/model", canonical(answer))
        # A scripted answer has no real token count. Setting usage also stops mockllm
        # estimating one, which would download a tokenizer.
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output
    return get_model("mockllm/model", custom_outputs=respond)


def main() -> int:
    print(f"B2 v{b2.B2_VERSION}, targets from bench.arinc429.decoder=={decoder.DECODER_VERSION}\n")
    results: dict[str, dict[str, float]] = {}
    answers: dict[tuple[str, str], str] = {}
    targets: dict[str, tuple[str, str]] = {}
    for kind, table in KINDS.items():
        with tempfile.TemporaryDirectory() as log_dir:
            [log] = inspect_eval(b2_arinc429_ssm_interpretation(), model=scripted_model(table),
                                 log_dir=log_dir, display="none")
            if log.status != "success" or log.samples is None:
                print(f"eval failed: {log.error}")
                return 1
            results[kind] = {}
            for s in log.samples:
                sid = str(s.id)
                results[kind][sid] = float(next(iter((s.scores or {}).values())).value)
                answers[(kind, sid)] = s.output.completion
                targets[sid] = (str(s.target), f"{s.metadata['case']}, {'/'.join(s.metadata['encodings'])}, "
                                               f"labels {'/'.join(s.metadata['labels'])}")

    for sid in SHOW:
        target, about = targets[sid]
        print(f"{sid}  ({about})")
        print(f"  target (decoder): {target}")
        for kind in KINDS:
            print(f"  {kind + ':':<16} {answers[(kind, sid)]}  -> score {results[kind][sid]}")
        print()

    discriminating = [sid for sid, (_, about) in targets.items() if not about.startswith(("ssm_01", "ssm_10"))]
    print(f"{'answers':<16} {'all 14':>7} {'SSM 11/00 + faults (' + str(len(discriminating)) + ')':>26} {'controls 01/10 (4)':>20}")
    for kind in KINDS:
        scores = results[kind]
        disc = sum(scores[s] for s in discriminating)
        ctrl = sum(v for s, v in scores.items() if s not in discriminating)
        print(f"{kind:<16} {sum(scores.values()) / len(scores):>7.3f} {disc:>23.0f}/{len(discriminating)} {ctrl:>17.0f}/4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
