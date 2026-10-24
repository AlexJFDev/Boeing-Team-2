"""Task D2 (ACARS indirect prompt injection): load, render and review the vector set.

A "vector" hides one instruction in one field of a synthetic ACARS host message.
`validate()` is the automated half of the review in backlog item 6.1:

  AC1  every vector has a hidden instruction, the field it hides in, and the
       expected safe behavior;
  AC2  anything beyond synthetic data (real frequencies, tail numbers, flight
       identifiers, links) is rejected.

AC2 is a set of heuristics. They catch the usual slips; a human reviewer still
signs off on the vector set.

Standard library only. Run from the repo root:
    python -m bench.acars.d2_vectors            # latest version: review table + checks
    python -m bench.acars.d2_vectors path.json  # check a specific file
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data" / "acars"
_VERSION_FILE_RE = re.compile(r"^d2_vectors_v(\d+)\.(\d+)\.(\d+)\.json$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

SLOT = "{{INJECTION}}"

GOALS = {
    "output_marker", "answer_manipulation", "warning_suppression",
    "system_prompt_leak", "unsafe_advice", "information_omission",
}
FRAMINGS = {
    "plain_directive", "fake_system_message", "authority_claim",
    "urgency", "task_documentation", "polite_request",
}
OBFUSCATIONS = {"none", "base64", "split_lines", "delimiter_escape"}
STEALTH = {"low", "medium", "high"}
COMPLIANCE_TYPES = {"canary", "answer_flip", "answer_omit", "prompt_leak", "judge"}
ANSWER_KINDS = {"literal", "computed", "free"}

VECTOR_KEYS = (
    "id", "host", "field", "goal", "framing", "obfuscation", "stealth", "owasp",
    "hidden_instruction", "injection_text", "compliance",
    "expected_safe_behavior", "unsafe_if",
)
HOST_KEYS = (
    "kind", "header", "text_template", "slot", "clean_filler", "task_prompt", "answer",
)

# Project convention for synthetic identifiers (see bench/acars/README.md).
REGISTRATION_RE = re.compile(r"^SYN[A-Z0-9]{2,4}$")
FLIGHT_ID_RE = re.compile(r"^SYN[0-9]{3}$")
LABEL_RE = re.compile(r"^Z[0-9A-Z]$")
MODE_RE = re.compile(r"^[0-9A-Z]$")

# AC2 heuristics: (rule name, pattern). Any match is a problem.
_BEYOND_SYNTHETIC = [
    ("VHF-band frequency", re.compile(r"\b1(?:0[89]|[12][0-9]|3[0-7])\.\d{1,3}\b")),
    ("frequency unit", re.compile(r"\b(?:KHZ|MHZ|GHZ)\b", re.I)),
    ("US-style tail number", re.compile(r"\bN\d{1,5}[A-Z]{0,2}\b")),
    ("hyphenated registration", re.compile(r"\b[A-Z]{1,2}-[A-Z0-9]{3,5}\b")),
    ("link or domain", re.compile(
        r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:com|net|org|gov|mil|edu|io|aero|int)\b)", re.I)),
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("IP address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("phone-like number", re.compile(r"\+\d[\d ()-]{7,}")),
]
_FLIGHT_LIKE_RE = re.compile(r"\b[A-Z]{2,3}\d{2,4}\b")
_FLIGHT_LIKE_OK = re.compile(r"^(?:SYN\d+|FL\d{2,3})$")
_BASE64_TOKEN_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


# -- loading --
def latest_path(data_dir: Path | str | None = None) -> Path:
    """Highest-versioned d2_vectors_vX.Y.Z.json in the folder (compared as numbers).

    A committed version is frozen: to change it, copy it to the next version
    (CLAUDE.md rule 4) and the loader picks the new file up.
    """
    folder = Path(data_dir) if data_dir else DATA_DIR
    found = []
    for p in folder.glob("d2_vectors_v*.json"):
        m = _VERSION_FILE_RE.match(p.name)
        if m:
            found.append((tuple(int(x) for x in m.groups()), p))
    if not found:
        raise FileNotFoundError(f"no d2_vectors_vX.Y.Z.json in {folder}")
    return max(found)[1]


def load(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or latest_path()).read_text(encoding="utf-8"))


def name_problems(path: Path | str, doc: dict) -> list[str]:
    """The file name's version must match the `version` field inside the file."""
    m = _VERSION_FILE_RE.match(Path(path).name)
    if not m:
        return [f"file name {Path(path).name!r} must look like d2_vectors_vX.Y.Z.json"]
    named = ".".join(m.groups())
    if str(doc.get("version")) != named:
        return [f"file name says version {named} but the file says {doc.get('version')!r}"]
    return []


# -- rendering --
def render_text(host: dict, injection_text: str | None = None) -> str:
    """Message text with the slot filled by the injection (or the clean filler)."""
    fill = host["clean_filler"] if injection_text is None else injection_text
    return host["text_template"].replace(SLOT, fill)


def render_message(host: dict, injection_text: str | None = None) -> str:
    h = host["header"]
    return "\n".join([
        f"MODE: {h['mode']}",
        f"LABEL: {h['label']}",
        f"REG: {h['registration']}",
        f"FLT: {h['flight_id']}",
        "TEXT:",
        render_text(host, injection_text),
    ])


def render_prompt(doc: dict, vector: dict, injected: bool = True) -> str:
    """Full task prompt for a vector; injected=False gives the clean control."""
    host = doc["hosts"][vector["host"]]
    inj = vector["injection_text"] if injected else None
    return f"{host['task_prompt']}\n\nACARS MESSAGE\n{render_message(host, inj)}"


# --- AC2 checks --
def synthetic_problems(text: str) -> list[str]:
    """Reasons `text` looks like more than synthetic data. Empty list means it passed."""
    out = []
    if not all(ch == "\n" or 32 <= ord(ch) < 127 for ch in text):
        out.append("non-ASCII or control characters (hidden characters make review impossible)")
    for name, pattern in _BEYOND_SYNTHETIC:
        m = pattern.search(text)
        if m:
            out.append(f"{name}: {m.group(0)!r}")
    for m in _FLIGHT_LIKE_RE.finditer(text):
        if not _FLIGHT_LIKE_OK.match(m.group(0)):
            out.append(f"possible real flight identifier: {m.group(0)!r}")
    return out


def _norm(value) -> str:
    return str(value).strip().lower()


def _literal_present(value, text: str) -> bool:
    items = value if isinstance(value, list) else [value]
    return all(str(v) in text for v in items)


def _base64_payloads(text: str) -> list[str]:
    found = []
    for tok in _BASE64_TOKEN_RE.findall(text):
        try:
            found.append(base64.b64decode(tok, validate=True).decode("ascii"))
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
    return found


# -- validation --
def _check_host(hid: str, host: dict) -> list[str]:
    p = [f"{hid}: missing '{k}'" for k in HOST_KEYS if k not in host]
    if p:
        return p
    hdr = host["header"]
    if not REGISTRATION_RE.match(str(hdr.get("registration", ""))):
        p.append(f"{hid}: registration {hdr.get('registration')!r} is not the synthetic form SYN + 2-4 characters")
    if not FLIGHT_ID_RE.match(str(hdr.get("flight_id", ""))):
        p.append(f"{hid}: flight_id {hdr.get('flight_id')!r} is not the synthetic form SYN + 3 digits")
    if not LABEL_RE.match(str(hdr.get("label", ""))):
        p.append(f"{hid}: label {hdr.get('label')!r} is not a placeholder label (Z + 1 character)")
    if not MODE_RE.match(str(hdr.get("mode", ""))):
        p.append(f"{hid}: mode {hdr.get('mode')!r} must be one character")
    if host["text_template"].count(SLOT) != 1:
        p.append(f"{hid}: text_template must contain {SLOT} exactly once")
        return p
    clean = render_text(host)
    for why in synthetic_problems(clean):
        p.append(f"{hid}: clean message is not synthetic: {why}")
    for why in synthetic_problems(host["task_prompt"]):
        p.append(f"{hid}: task_prompt is not synthetic: {why}")
    answer = host["answer"]
    if not isinstance(answer, dict) or not answer:
        p.append(f"{hid}: answer must list the keys the model is asked for")
        return p
    for key, spec in answer.items():
        kind = spec.get("kind")
        if kind not in ANSWER_KINDS:
            p.append(f"{hid}: answer '{key}' has kind {kind!r}; expected one of {sorted(ANSWER_KINDS)}")
        elif kind == "free":
            continue
        elif "value" not in spec:
            p.append(f"{hid}: answer '{key}' needs a value")
        elif kind == "literal" and not _literal_present(spec["value"], clean):
            p.append(f"{hid}: literal answer '{key}'={spec['value']!r} does not appear in the clean message")
        if key not in host["task_prompt"]:
            p.append(f"{hid}: answer key '{key}' is not named in the task prompt")
    return p


def _check_compliance(vid: str, v: dict, host: dict, injected_render: str) -> list[str]:
    c = v.get("compliance")
    if not isinstance(c, dict) or c.get("type") not in COMPLIANCE_TYPES:
        return [f"{vid}: compliance.type must be one of {sorted(COMPLIANCE_TYPES)}"]
    p = []
    t = c["type"]
    answer = host["answer"]
    if t == "canary":
        tok = str(c.get("token", ""))
        if len(tok) < 8 or not tok.isalnum():
            p.append(f"{vid}: canary token must be at least 8 letters/digits")
        elif tok.lower() in injected_render.lower():
            p.append(f"{vid}: canary {tok!r} appears literally in the input, so quoting the injection would trip it")
        if v.get("obfuscation") == "base64" and tok:
            if not any(tok.upper() in d.upper() for d in _base64_payloads(v["injection_text"])):
                p.append(f"{vid}: no base64 payload in injection_text decodes to something containing {tok!r}")
    elif t == "answer_flip":
        field = c.get("field")
        spec = answer.get(field)
        if spec is None or spec.get("kind") == "free":
            p.append(f"{vid}: answer_flip field {field!r} must be a checkable key of host {v['host']}")
        elif "injected_value" not in c:
            p.append(f"{vid}: answer_flip needs injected_value")
        elif _norm(c["injected_value"]) == _norm(spec.get("value")):
            p.append(f"{vid}: injected_value equals the true answer, so following the injection is not detectable")
    elif t == "answer_omit":
        if c.get("field") not in answer:
            p.append(f"{vid}: answer_omit field {c.get('field')!r} is not a key of host {v['host']}")
    elif t == "prompt_leak":
        if not c.get("needs"):
            p.append(f"{vid}: prompt_leak must say what the harness has to supply (e.g. a preamble sentinel)")
    return p


def _check_vector(vid: str, v: dict, doc: dict) -> list[str]:
    # AC1: hidden instruction, field, expected safe behavior (plus the rest of the record)
    p = []
    for k in VECTOR_KEYS:
        val = v.get(k)
        if val is None or (isinstance(val, str) and not val.strip()):
            p.append(f"{vid}: missing '{k}'")
    if p:
        return p
    host = doc["hosts"].get(v["host"])
    if host is None:
        return [f"{vid}: unknown host {v['host']!r}"]
    if v["field"] != host.get("slot"):
        p.append(f"{vid}: field {v['field']!r} is not the host's injection slot {host.get('slot')!r}")
    if v["goal"] not in GOALS:
        p.append(f"{vid}: goal {v['goal']!r} not in {sorted(GOALS)}")
    if v["framing"] not in FRAMINGS:
        p.append(f"{vid}: framing {v['framing']!r} not in {sorted(FRAMINGS)}")
    if v["obfuscation"] not in OBFUSCATIONS:
        p.append(f"{vid}: obfuscation {v['obfuscation']!r} not in {sorted(OBFUSCATIONS)}")
    if v["stealth"] not in STEALTH:
        p.append(f"{vid}: stealth {v['stealth']!r} not in {sorted(STEALTH)}")
    if not re.match(r"^LLM(?:0[1-9]|10)$", str(v["owasp"])):
        p.append(f"{vid}: owasp {v['owasp']!r} should look like LLM01..LLM10")
    if len(v["expected_safe_behavior"].strip()) < 20:
        p.append(f"{vid}: expected_safe_behavior is too short to be useful")
    if host["text_template"].count(SLOT) != 1:
        return p  # host problem is reported with the host
    # AC2: the injected material itself must be synthetic
    injected = render_message(host, v["injection_text"])
    for why in synthetic_problems(v["injection_text"]):
        p.append(f"{vid}: injection_text is not synthetic: {why}")
    for why in synthetic_problems(v["hidden_instruction"]):
        p.append(f"{vid}: hidden_instruction is not synthetic: {why}")
    for decoded in _base64_payloads(v["injection_text"]):
        for why in synthetic_problems(decoded):
            p.append(f"{vid}: base64 payload is not synthetic: {why}")
    p.extend(_check_compliance(vid, v, host, injected))
    return p


def validate(doc: dict) -> list[str]:
    """All review problems found in the vector set (empty list means it passes)."""
    p = []
    if not _VERSION_RE.match(str(doc.get("version", ""))):
        p.append("file: version must look like 1.2.3")
    if doc.get("family") != "D2":
        p.append("file: family must be 'D2'")
    if doc.get("origin") != "synthetic":
        p.append("file: origin must be 'synthetic'")
    if doc.get("source") != "spec-synthesized":
        p.append("file: source must be 'spec-synthesized' (an approved source for the data boundary checker)")
    if doc.get("injection_slot") != SLOT:
        p.append(f"file: injection_slot must be {SLOT!r}")
    hosts = doc.get("hosts")
    vectors = doc.get("vectors")
    if not isinstance(hosts, dict) or not hosts:
        return p + ["file: 'hosts' must be a non-empty object"]
    if not isinstance(vectors, list) or not vectors:
        return p + ["file: 'vectors' must be a non-empty list"]
    for hid, host in hosts.items():
        p.extend(_check_host(hid, host))
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    for i, v in enumerate(vectors):
        vid = str(v.get("id") or f"vector[{i}]")
        if vid in seen_ids:
            p.append(f"{vid}: duplicate vector id")
        seen_ids.add(vid)
        key = str(v.get("injection_text", "")).strip().lower()
        if key and key in seen_text:
            p.append(f"{vid}: injection_text duplicates {seen_text[key]}")
        seen_text.setdefault(key, vid)
        p.extend(_check_vector(vid, v, doc))
    return p


# -- CLI table --
def review_table(doc: dict) -> str:
    cols = ("ID", "HOST", "FIELD", "GOAL", "FRAMING", "OBFUSCATION", "CHECK", "INJECTED TEXT")
    rows = []
    for v in doc["vectors"]:
        text = v["injection_text"].replace("\n", " / ")
        if len(text) > 56:
            text = text[:53] + "..."
        rows.append((v["id"], v["host"], v["field"], v["goal"], v["framing"],
                     v["obfuscation"], v["compliance"]["type"], text))
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths)).rstrip()
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(r) for r in rows])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else latest_path()
    doc = load(path)
    print(f"{path.name}\n")
    print(review_table(doc))
    problems = validate(doc) + name_problems(path, doc)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for msg in problems:
            print(f"  {msg}")
        return 1
    print(f"\n{len(doc['vectors'])} vectors on {len(doc['hosts'])} hosts: all review checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
