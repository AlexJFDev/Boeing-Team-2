"""Run the Task D2 prompt-injection harness and print a report.

    python -m bench.acars.d2_run --model mockllm/model --script safe
    python -m bench.acars.d2_run --model mockllm/model --script obedient --vectors V01,V06
    python -m bench.acars.d2_run --model mockllm/model                  # filler: unparseable
    python -m bench.acars.d2_run --model ollama/llama3.1 --seed 42 --temperature 0
    python -m bench.acars.d2_run --demo                                 # safe vs obedient vs filler
    python -m bench.acars.d2_run --log logs/<file>.eval                 # report an existing log

``--script safe|obedient`` drives ``mockllm/model`` with the scripted replies in
``bench/acars/d2_scripted.py`` (offline, no API key or GPU). The report shows,
per vector, whether the control answer was right (capability) and what the model
did with the injected prompt (safety), then the two signals' counts separately.
There is no combined score.

The exit code says whether the harness ran, not how the model behaved: 0 when
every eval finished, 1 when an eval errored, 2 for bad arguments.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessage, GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo

from bench.acars import d2_vectors as d2
from bench.acars.d2_detect import NEEDS_JUDGE, OBEYED, OUTCOMES, RESISTED, UNPARSEABLE
from bench.acars.d2_harness import (
    CAPABILITY_CATEGORIES,
    CONTROL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    INJECTED,
    VECTOR_FILE,
    build_task,
)
from bench.acars.d2_scripted import PERSONAS, scripted_responder
from runner.endpoint import ollama_model_metadata

SAFETY_SCORER = "d2_safety"
CAPABILITY_SCORER = "d2_control_capability"
DEFAULT_LOG_DIR = "logs"

INJECTED_LABEL = {
    RESISTED: "PASS (resisted)",
    OBEYED: "FAIL (obeyed)",
    UNPARSEABLE: "UNPARSEABLE (not a pass)",
    NEEDS_JUDGE: "DEFERRED (needs judge)",
}


# -- models --
def scripted_model(persona: str, config: GenerateConfig | None = None) -> Model:
    """mockllm/model whose reply depends on the prompt, via its callable custom_outputs."""
    respond = scripted_responder(d2.load(VECTOR_FILE), persona)

    def outputs(
        messages: list[ChatMessage], tools: list[ToolInfo], tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        system_text = "\n".join(m.text for m in messages if m.role == "system")
        user_text = next((m.text for m in reversed(messages) if m.role == "user"), "")
        return ModelOutput.from_content(model="mockllm", content=respond(system_text, user_text))

    return get_model("mockllm/model", config=config, memoize=False, custom_outputs=outputs)


def run(
    model: str,
    script: str | None = None,
    vectors: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    guardrail_file: str | None = None,
    sentinel: str | None = None,
    log_dir: str = DEFAULT_LOG_DIR,
    max_connections: int | None = None,
) -> EvalLog:
    if script is not None and model != "mockllm/model":
        raise ValueError("--script only works with --model mockllm/model")
    if max_connections is not None and max_connections < 1:
        raise ValueError("--max-connections must be at least 1")
    if max_connections is None:
        max_connections = default_max_connections(model)
    task = build_task(vectors=vectors, temperature=temperature, seed=seed, max_tokens=max_tokens,
                      guardrail_file=guardrail_file, sentinel=sentinel)
    # The same sampling settings go to the task (recorded in plan.config) and to
    # the model (recorded in eval.model_generate_config).
    config = GenerateConfig(temperature=float(temperature), seed=seed, max_tokens=max_tokens,
                            max_connections=max_connections)
    model_obj = scripted_model(script, config) if script else get_model(model, config=config)
    metadata: dict[str, Any] = {
        "scripted_persona": script or "none",
        "max_connections": max_connections if max_connections is not None else "inspect_default",
    }
    metadata.update(ollama_model_metadata(model) or {})
    [log] = inspect_eval(
        task,
        model=model_obj,
        log_dir=log_dir,
        display="none",
        metadata=metadata,
        max_connections=max_connections,
    )
    return log


def default_max_connections(model: str) -> int | None:
    """1 for Ollama: requests run one at a time, so the server does not batch them,
    which made identical prompts give different replies. None keeps Inspect's default."""
    return 1 if model.startswith("ollama/") else None


# -- report --
def _value(sample: Any, scorer_name: str) -> str:
    scores = sample.scores or {}
    score = scores.get(scorer_name)
    return str(score.value) if score is not None else "missing"


def rows(log: EvalLog) -> list[dict[str, str]]:
    """One row per vector: control capability and injected safety outcome, side by side."""
    by_vector: dict[str, dict[str, str]] = {}
    for s in log.samples or []:
        md = s.metadata
        row = by_vector.setdefault(md["vector_id"], {
            "vector": md["vector_id"], "host": md["host"], "goal": md["goal"],
            "check": md["compliance"]["type"], "control": "missing", "injected": "missing",
        })
        if md["condition"] == CONTROL:
            row["control"] = _value(s, CAPABILITY_SCORER)
        elif md["condition"] == INJECTED:
            row["injected"] = _value(s, SAFETY_SCORER)
    return [by_vector[k] for k in sorted(by_vector)]


def safety_counts(log: EvalLog) -> Counter[str]:
    return Counter(r["injected"] for r in rows(log))


def flip_control_annotations(log: EvalLog) -> dict[str, list[str]]:
    """Report-only: for answer_flip vectors scored ``resisted``, was the control right on the flipped key?

    Read from scores already in the log (the control sample's capability field
    breakdown); nothing is re-scored and no outcome changes. Returns vector ids
    under "control_wrong", "control_correct" and "control_unreadable".
    """
    out: dict[str, list[str]] = {"control_wrong": [], "control_correct": [], "control_unreadable": []}
    samples = {str(s.id): s for s in log.samples or []}
    for vid in sorted({s.metadata["vector_id"] for s in samples.values()}):
        inj, ctl = samples.get(f"{vid}:{INJECTED}"), samples.get(f"{vid}:{CONTROL}")
        if inj is None or ctl is None or inj.metadata["compliance"]["type"] != "answer_flip":
            continue
        if _value(inj, SAFETY_SCORER) != RESISTED:
            continue
        key = inj.metadata["compliance"]["field"]
        cap = (ctl.scores or {}).get(CAPABILITY_SCORER)
        field = ((cap.metadata or {}).get("fields") or {}).get(key) if cap is not None else None
        if field is None:
            out["control_unreadable"].append(vid)
        else:
            out["control_correct" if field["match"] else "control_wrong"].append(vid)
    return out


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h) for i, h in enumerate(header)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in body])


def provenance_lines(log: EvalLog) -> list[str]:
    md = log.eval.metadata or {}
    cfg = log.plan.config
    persona = md.get("scripted_persona", "none")
    model = log.eval.model + (f" (scripted: {persona})" if persona != "none" else "")
    lines = [f"model: {model}"]
    ident = md.get("endpoint_model")
    if isinstance(ident, dict):
        if ident.get("status") == "ok":
            lines.append(f"  ollama {ident.get('ollama_version')}  {ident.get('name')}  digest {ident.get('digest')}"
                         f"  {ident.get('parameter_size')} {ident.get('quantization_level')}")
        else:
            lines.append(f"  model identity NOT recorded: {ident.get('status')}: {ident.get('error')}")
    return lines + [
        f"temperature: {cfg.temperature}  seed: {cfg.seed}  max_tokens: {cfg.max_tokens}"
        f"  max_connections: {md.get('max_connections', 'inspect_default')}",
        f"vectors: {md.get('vector_file')} v{md.get('vector_version')} sha256 {md.get('vector_sha256')}",
        f"prompt template: v{md.get('system_template_version')} sha256 {md.get('prompt_template_sha256')}",
        f"guardrail: {md.get('guardrail_file') or 'none'} sha256 {md.get('guardrail_sha256')}",
        f"V06 sentinel: {md.get('v06_sentinel')} ({md.get('v06_sentinel_source', 'unknown')})",
        f"detector: {md.get('detector_version')}  capability check: {md.get('capability_check_version')}"
        f"  task: {log.eval.task_version}  inspect_ai: {md.get('inspect_ai_version')}",
    ]


def report(log: EvalLog) -> str:
    body = [(r["vector"], r["host"], r["goal"], r["check"], r["control"],
             INJECTED_LABEL.get(r["injected"], r["injected"])) for r in rows(log)]
    table = _table(("VECTOR", "HOST", "GOAL", "CHECK", "CONTROL ANSWER", "INJECTED (SAFETY)"), body)
    safety = safety_counts(log)
    capability = Counter(r["control"] for r in rows(log))
    decided = safety[RESISTED] + safety[OBEYED]
    out = [f"Task D2 indirect prompt injection: {log.eval.task}", *provenance_lines(log), "", table, ""]
    out.append("Safety (injected prompts):  " + "  ".join(f"{o} {safety[o]}" for o in OUTCOMES))
    out.append(f"  deferred to the judge model (item 6.7): {safety[NEEDS_JUDGE]}; "
               f"unparseable (not counted as a pass): {safety[UNPARSEABLE]}")
    out.append(f"  resist rate (decided only): {safety[RESISTED] / decided:.2f} "
               f"({safety[RESISTED]} of {decided})" if decided
               else "  resist rate (decided only): n/a (no decided samples)")
    notes = flip_control_annotations(log)
    wrong, unreadable = notes["control_wrong"], notes["control_unreadable"]
    out.append(f"  resisted, but control wrong for the flipped key: {len(wrong)}"
               + (f" ({', '.join(wrong)})" if wrong else "")
               + (f"; control unreadable: {len(unreadable)} ({', '.join(unreadable)})" if unreadable else ""))
    out.append("Capability (control prompts, reported separately):  "
               + "  ".join(f"{c} {capability[c]}" for c in CAPABILITY_CATEGORIES if c != "not_applicable"))
    out.append("No combined safety/capability score is computed (CLAUDE.md rule 2).")
    out.append(f"status: {log.status}  log: {log.location}")
    return "\n".join(out)


def demo_report(logs: dict[str, EvalLog]) -> str:
    """Safe, obedient and filler runs side by side: control answer / injected outcome."""
    names = list(logs)
    per = {n: {r["vector"]: r for r in rows(logs[n])} for n in names}
    vids = sorted(set().union(*(set(p) for p in per.values())))
    header = ["VECTOR", "CHECK"] + [f"{n.upper()}: CONTROL / INJECTED" for n in names]
    body = []
    for vid in vids:
        first = next(p[vid] for p in per.values() if vid in p)
        cells = []
        for n in names:
            r = per[n].get(vid)
            cells.append(f"{r['control']} / {INJECTED_LABEL.get(r['injected'], r['injected'])}" if r else "-")
        body.append([vid, first["check"], *cells])
    out = ["Task D2 demo: the same vectors against three stand-in models", "", _table(header, body), ""]
    for n in names:
        out.extend(["", f"== {n} ==", report(logs[n])])
    return "\n".join(out)


# -- CLI --
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.acars.d2_run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Inspect model, e.g. mockllm/model or ollama/llama3.1")
    p.add_argument("--script", choices=PERSONAS, help="scripted mockllm persona (offline demo)")
    p.add_argument("--vectors", help="comma-separated vector ids, e.g. V01,V03 (default: all)")
    p.add_argument("--guardrail-file", help="text appended to the system message (default: none)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--sentinel", help="override the V06 sentinel (default: derived from --seed and the vector file)")
    p.add_argument("--max-connections", type=int,
                   help="concurrent model requests (default: 1 for ollama/..., Inspect's default otherwise)")
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--demo", action="store_true", help="run safe, obedient and filler mockllm side by side")
    p.add_argument("--log", help="print the report for an existing .eval log and exit")
    args = p.parse_args(argv)

    if args.log:
        print(report(read_eval_log(args.log)))
        return 0
    common: dict[str, Any] = dict(
        vectors=args.vectors, temperature=args.temperature, seed=args.seed,
        max_tokens=args.max_tokens, guardrail_file=args.guardrail_file,
        sentinel=args.sentinel, log_dir=args.log_dir, max_connections=args.max_connections,
    )
    try:
        if args.demo:
            logs = {
                "safe": run("mockllm/model", "safe", **common),
                "obedient": run("mockllm/model", "obedient", **common),
                "filler": run("mockllm/model", None, **common),
            }
            print(demo_report(logs))
            return 0 if all(lg.status == "success" for lg in logs.values()) else 1
        if not args.model:
            p.error("--model is required (or use --demo / --log)")
        log = run(args.model, args.script, **common)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(report(log))
    return 0 if log.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
