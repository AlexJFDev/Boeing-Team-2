"""Demo for task 3.3 (issue #58): B1 answers scored against the reference decoder.

Runs the real B1 task with ``inspect eval`` on a few samples. The model is a scripted
stand-in (Inspect's mockllm with custom outputs), so no model server is needed and
the result is the same every time. It gives three kinds of answer:

- correct: the reference decoder's answer, as a model that decodes properly would give
- naive label: everything right except the label, read forward from the octet (the
  B1 discriminator: 213 becomes 321)
- no reject: a corrupt word decoded anyway instead of rejected

Each answer is scored by the task's own scorer against the target, which comes from
the reference decoder (bench/arinc429/decoder.py). For a real model, use
configs/demo_b1.yaml.

Usage (from the repo root):
    python scripts/arinc429/demo_b1_scoring.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, ModelUsage, get_model  # noqa: E402

from bench.arinc429 import b1, decoder  # noqa: E402
from tasks.b1_arinc429_word_decode import b1_arinc429_word_decode  # noqa: E402

# Sample -> how the scripted model answers it.
SCRIPT = {
    "b1-a429-0021": "correct",       # label 213, the catalogue's worked example
    "b1-a429-0006": "correct",       # label 203, negative BNR
    "b1-a429-0005": "correct",       # label 012, BCD
    "b1-a429-0015": "correct",       # parity bit flipped: reject
    "b1-unknown-01": "correct",      # label 310, no profile: cannot be determined
    "b1-a429-0022": "naive_label",   # label 213 read forward as 321
    "b1-a429-0001": "no_reject",     # corrupt word decoded instead of rejected
}


def scripted_answer(word_hex: str, kind: str) -> str:
    word = int(word_hex, 16)
    answer = b1.reference_answer(word)
    if kind == "naive_label":
        answer = {**answer, "label": b1.naive_label(word)}
    elif kind == "no_reject":
        fields = decoder.read_fields(word)
        answer = {"label": fields["label_octal"], "sdi": fields["sdi"],
                  "ssm": fields["ssm_bits"], "value": 0}
    return b1.canonical(answer)


def main() -> int:
    task = b1_arinc429_word_decode()
    task.dataset = task.dataset.filter(lambda s: s.id in SCRIPT)
    words = {s.id: s.metadata["word"] for s in task.dataset}
    kinds = {words[sid]: kind for sid, kind in SCRIPT.items()}

    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        word_hex = re.search(r"Word: (0x[0-9A-F]{8})", messages[-1].text).group(1)
        output = ModelOutput.from_content("mockllm/model", scripted_answer(word_hex, kinds[word_hex]))
        # A scripted answer has no real token count. Setting usage also stops mockllm
        # estimating one, which would download a tokenizer.
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output

    model = get_model("mockllm/model", custom_outputs=respond)
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = inspect_eval(task, model=model, log_dir=log_dir, display="none")
        # Read the results here: the log's samples are loaded from the log file.
        if log.status != "success" or log.samples is None:
            print(f"eval failed: {log.error}")
            return 1
        print(f"B1 v{b1.B1_VERSION}: {len(log.samples)} samples, targets from "
              f"bench.arinc429.decoder=={decoder.DECODER_VERSION}\n")
        for sample in sorted(log.samples, key=lambda s: list(SCRIPT).index(str(s.id))):
            score = next(iter((sample.scores or {}).values()))
            meta = sample.metadata
            print(f"{sample.id}  word {meta['word']}  ({meta['case']}, scripted: {SCRIPT[str(sample.id)]})")
            print(f"  target (decoder): {sample.target}")
            print(f"  model answer:     {sample.output.completion}")
            print(f"  score:            {score.value}\n")
        assert log.results is not None
        metrics = {name: round(m.value, 3) for s in log.results.scores for name, m in s.metrics.items()}
        print(f"overall: {json.dumps(metrics)}  (expected accuracy 5/7 = 0.714)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
