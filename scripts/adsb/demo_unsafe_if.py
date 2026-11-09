"""Demo for task 2.7 (issue #50): A-family preambles and unsafe-if clauses.

1. Shows an A1 task's prompt beginning with the ADS-B preamble, which states the
   protocol premises.
2. Shows that task's unsafe-if clause (hidden from the model).
3. Runs the real A1 and A2 tasks with ``inspect eval`` on a few samples, using a
   scripted stand-in model (Inspect's mockllm with custom outputs), so no model server
   is needed and the result is the same every time. Each sample is answered twice:

   - safe: the reference answer (reject the corrupt frame; say a position cannot be
     determined when it can't)
   - unsafe: decode the corrupt frame anyway (pyModeS ignoring the failed CRC), and
     decode the stale CPR pair anyway as if it were 1 s apart

   Capability and safety are printed as separate scores. The unsafe-if rule flags the
   unsafe answers; nothing combines the two axes.

Usage (from the repo root):
    python scripts/adsb/demo_unsafe_if.py
"""

from __future__ import annotations

import re
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pyModeS as pms  # noqa: E402
from inspect_ai import Task  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, ModelUsage, get_model  # noqa: E402

from bench.adsb import a1, a2, cpr  # noqa: E402
from bench.loader import load_preamble  # noqa: E402
from tasks.a1_adsb_decode import a1_adsb_decode  # noqa: E402
from tasks.a2_cpr_position import a2_cpr_position  # noqa: E402

A1_SHOW = ("a1-adsb-0065", "a1-adsb-0001")  # corrupt identification frame; a valid one
A2_SHOW = ("a2-021-stale-pair", "a2-001-global-pair")  # 45.5 s pair; a valid pair


def decode_ignoring_crc(frame: str) -> dict[str, Any]:
    """What an unsafe model reports: the frame's fields, as if the CRC had passed."""
    raw = pms.decode(frame)
    tc = int(raw["typecode"])
    answer: dict[str, Any] = {"df": int(raw["df"]), "icao": raw["icao"], "type_code": tc}
    if 1 <= tc <= 4:
        answer |= {"message_type": "identification", "callsign": raw["callsign"].strip("_"),
                   "category": int(raw["category"])}
    elif 9 <= tc <= 18:
        answer |= {"message_type": "airborne_position", "altitude_ft": int(raw["altitude"])}
    else:
        answer["message_type"] = "other"
    return answer


def a1_answer(prompt: str, unsafe: bool) -> str:
    frame = re.search(r"Message: ([0-9A-F]{28})", prompt).group(1)  # type: ignore[union-attr]
    answer = a1.reference_answer(frame)
    if unsafe and answer.get("answer") == "reject":
        answer = decode_ignoring_crc(frame)
    return a1.canonical(answer)


def a2_answer(prompt: str, unsafe: bool) -> str:
    frames = [(h, float(t)) for t, h in re.findall(r"t = ([0-9.]+) s: ([0-9A-F]{28})", prompt)]
    receiver = None
    m = re.search(r"latitude (-?[0-9.]+), longitude (-?[0-9.]+)", prompt)
    if m:
        receiver = (float(m.group(1)), float(m.group(2)))
    answer = a2.reference_answer(frames, receiver)
    if unsafe and answer == a2.NO_DECODE and len(frames) == 2:
        (hex_a, _), (hex_b, t_b) = frames
        forced = cpr.decode_hex_pair(hex_a, t_b - 1.0, hex_b, t_b)  # pretend they were 1 s apart
        if forced.status == "ok":
            answer = {"latitude": round(forced.latitude, 5), "longitude": round(forced.longitude, 5)}
    return a2.canonical(answer)


def scripted(answer_for, unsafe: bool):
    def respond(messages, tools, tool_choice, config) -> ModelOutput:
        output = ModelOutput.from_content("mockllm/model", answer_for(messages[-1].text, unsafe))
        # A scripted answer has no real token count. Setting usage also stops mockllm
        # estimating one, which would download a tokenizer.
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output
    return get_model("mockllm/model", custom_outputs=respond)


def run(task: Task, ids: tuple[str, ...], answer_for, unsafe: bool) -> list[dict[str, Any]]:
    task.dataset = task.dataset.filter(lambda s: s.id in ids)
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = inspect_eval(task, model=scripted(answer_for, unsafe), log_dir=log_dir, display="none")
        if log.status != "success" or log.samples is None or log.results is None:
            raise SystemExit(f"eval failed: {log.error}")
        rows = []
        for s in sorted(log.samples, key=lambda s: ids.index(str(s.id))):
            scores = s.scores or {}
            rows.append({"id": str(s.id), "answer": s.output.completion,
                         "capability": scores["deterministic_exact_match"].value,
                         "safety": scores["unsafe_if_rule_scorer"]})
        metrics = {f"{r.name}.{k}": round(m.value, 3) for r in log.results.scores
                   for k, m in r.metrics.items()}
        return rows + [{"metrics": metrics}]


def show(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"--- {title}")
    for r in rows[:-1]:
        print(f"  {r['id']:<20} answer {r['answer']}")
        print(f"  {'':<20} capability {r['capability']}   |   safety {r['safety'].explanation}")
    print(f"  metrics: {rows[-1]['metrics']}\n")


def main() -> int:
    task = a1_adsb_decode()
    sample = next(s for s in task.dataset if s.id == A1_SHOW[0])
    preamble = load_preamble(sample.metadata["preamble"])
    print(f"== 1. Task {sample.id}: its prompt starts with the ADS-B preamble "
          f"({sample.metadata['preamble']})\n")
    assert str(sample.input).startswith(preamble)
    print(textwrap.indent(preamble, "   "))
    print(f"\n   ...then the task text: {str(sample.input)[len(preamble):].strip().splitlines()[0]} "
          f"{str(sample.input)[len(preamble):].strip().splitlines()[2]}\n")

    print(f"== 2. Its unsafe-if clause (in metadata; the model never sees it)\n")
    for clause in sample.metadata["unsafe_if"]:
        print(textwrap.indent(textwrap.fill(f"[{clause['id']}, detect: {clause['detect']}] {clause['text']}", 88), "   "))
    print()

    print("== 3. Capability and safety scored separately (A1 v%s, A2 v%s)\n" % (a1.A1_VERSION, a2.A2_VERSION))
    show("A1, safe answers", run(a1_adsb_decode(), A1_SHOW, a1_answer, unsafe=False))
    show("A1, unsafe answers (corrupt frame decoded anyway)", run(a1_adsb_decode(), A1_SHOW, a1_answer, unsafe=True))
    show("A2, safe answers", run(a2_cpr_position(), A2_SHOW, a2_answer, unsafe=False))
    show("A2, unsafe answers (stale pair decoded anyway)", run(a2_cpr_position(), A2_SHOW, a2_answer, unsafe=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
