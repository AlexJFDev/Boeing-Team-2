"""Sanitized, synthetic ACARS dispatch message corpus (task 5.1, issue #76).

This module owns four things:

1. The message schema and the project-defined body grammar per label.
2. The sanitization checklist (S1-S8), implemented as automated checks.
3. ``add_message`` (reject at ingest) and ``load_corpus`` (fail at load).
4. Canonical serialization and the corpus hash recorded in the manifest.

Claim limits (read before citing this corpus):
* The body grammars are PROJECT-DEFINED. Real ACARS "airline defined" labels have
  no public body spec, so ground truth here means "per this project's grammar",
  not "per ARINC 618/620".
* "Sanitized" means "passes checklist S1-S8 below", not "safe". Pattern checks
  cannot prove that a free-text string is free of every real-world identifier.
  The corpus is synthetic-only, which is what actually keeps real data out.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRAMMAR_VERSION = "syn-acars-grammar-1"
CHECKLIST_VERSION = "1.0"
SANITIZATION_METHOD = "synthesized"

IN_SCOPE_LABELS = ("H1", "80", "5Z")
ALLOWED_MODES = ("2",)
CATEGORIES = (
    "valid",
    "valid_lookalike",
    "malformed",
    "label_mismatch",
    "adversarial_free_text",
)

REQUIRED_FIELDS = (
    "id", "mode", "label", "registration", "flight_id", "text", "reference", "sanitization",
)
REQUIRED_REFERENCE_FIELDS = (
    "category", "well_formed", "body_matches_label", "label_consistent",
    "fields", "free_text_fields", "contains_instruction_like_text",
)
REQUIRED_MANIFEST_FIELDS = (
    "name", "version", "sha256", "message_count", "grammar_version",
    "checklist_version", "source", "data_boundary",
)

# ----------------------------------------------------------------- checklist
CHECKLIST: tuple[tuple[str, str], ...] = (
    ("S1", "registration uses the reserved synthetic format SYN-A###"),
    ("S2", "flight_id uses the reserved synthetic format SYN####"),
    ("S3", "text has no real-looking registration (US N-number, hyphenated tail)"),
    ("S4", "text has no real-looking flight designator (IATA or ICAO style)"),
    ("S5", "text has no email address, URL or phone number"),
    ("S6", "text has no coordinates"),
    ("S7", "text has no calendar date (ISO, slash, compact or DDMMMYY form)"),
    ("S8", "text is printable ASCII, single line, at most 200 characters"),
)

# ------------------------------------------------------------ body grammars
_HHMM = r"(?:[01]\d|2[0-3])[0-5]\d"
_STN = r"ZZ\d{2}"
_RMK = r"(?: RMK (?P<rmk>\S.*))?"

GRAMMARS: dict[str, re.Pattern[str]] = {
    # Dispatch release
    "H1": re.compile(
        rf"REL (?P<flight>SYN\d{{4}}) (?P<orig>{_STN})-(?P<dest>{_STN}) "
        rf"FUEL (?P<fuel>\d{{4,5}}){_RMK}"
    ),
    # Out/off report
    "80": re.compile(
        rf"OUT (?P<out>{_HHMM}) OFF (?P<off>{_HHMM}) STN (?P<stn>{_STN}){_RMK}"
    ),
    # ETA / gate downlink
    "5Z": re.compile(rf"ETA (?P<eta>{_HHMM}) GATE (?P<gate>G\d{{2}}){_RMK}"),
}
FREE_TEXT_GROUPS = ("rmk",)

# ------------------------------------------------------ sanitization patterns
_B = r"(?<![A-Z0-9])", r"(?![A-Z0-9])"
REG_SYN = re.compile(r"SYN-A\d{3}")
FLIGHT_SYN = re.compile(r"SYN\d{4}")
# Synthetic tokens that are allowed inside text; removed before scanning.
_SYNTHETIC_ALLOW = (
    re.compile(_B[0] + r"SYN-A\d{3}" + _B[1]),
    re.compile(_B[0] + r"SYN\d{4}" + _B[1]),
    re.compile(_B[0] + r"ZZ\d{2}" + _B[1]),
    re.compile(_B[0] + r"G\d{2}" + _B[1]),
)
_REAL_REG = (
    re.compile(r"N\d{1,5}[A-Z]{0,2}"),        # US N-number
    re.compile(r"[A-Z]{1,2}-[A-Z0-9]{3,4}"),  # G-ABCD, VH-OQA, D-AIMA
)
_REAL_FLIGHT = (
    re.compile(r"(?:[A-Z][A-Z0-9]|[0-9][A-Z])\d{1,4}[A-Z]?"),  # IATA: UA123, 9W402
    re.compile(r"[A-Z]{3}\d{1,4}[A-Z]?"),       # ICAO style: UAL123
)
_TOKEN = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+)*")
_CONTACT = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"(?i)(?:https?://|www\.)\S+"),
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\+?\d{10,15}(?!\d)"),  # unseparated phone number
)
_COORDS = (
    re.compile(r"(?<!\d)-?\d{1,3}\.\d{3,}(?!\d)"),
    re.compile(r"\b[NS]\d{4,6}[EW]\d{5,7}\b", re.I),
    re.compile(r"\b\d{4,6}[NS]\d{5,7}[EW]\b", re.I),
)
_MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
_DATES = (
    re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}-\d{2}(?!\d)"),                 # 2024-05-17
    re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?!\d)"),            # 05/17/2024
    re.compile(r"(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)"),  # 20240517
    re.compile(rf"(?i)(?<!\d)\d{{1,2}}(?:{_MONTHS})(?:\d{{4}}|\d{{2}})?(?![A-Z0-9])"),   # 17MAY24
)
MAX_TEXT_LEN = 200


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


class CorpusRejection(ValueError):
    """Raised when a message or corpus fails validation."""

    def __init__(self, violations: list[Violation], where: str = ""):
        self.violations = violations
        prefix = f"{where}: " if where else ""
        super().__init__(prefix + "; ".join(str(v) for v in violations))


# ------------------------------------------------------------------- parsing
def parse_body(text: str) -> tuple[str | None, dict[str, str]]:
    """Return (label whose grammar the body matches, parsed fields) or (None, {})."""
    for label, pattern in GRAMMARS.items():
        m = pattern.fullmatch(text)
        if m:
            return label, {k: v for k, v in m.groupdict().items() if v is not None}
    return None, {}


def compute_reference(label: str, text: str) -> dict[str, Any]:
    """Reference values derivable from message structure alone."""
    matched, fields = parse_body(text)
    return {
        "well_formed": matched == label,
        "body_matches_label": matched,
        "label_consistent": matched == label,
        "fields": fields,
        "free_text_fields": [g for g in FREE_TEXT_GROUPS if g in fields],
    }


# ------------------------------------------------------------ sanitization
def sanitization_violations(msg: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    reg, flight, text = msg.get("registration"), msg.get("flight_id"), msg.get("text")
    if not (isinstance(reg, str) and REG_SYN.fullmatch(reg)):
        out.append(Violation("S1", f"registration {reg!r} is not a reserved synthetic value"))
    if not (isinstance(flight, str) and FLIGHT_SYN.fullmatch(flight)):
        out.append(Violation("S2", f"flight_id {flight!r} is not a reserved synthetic value"))
    if not isinstance(text, str):
        return out

    if not (text.isascii() and text.isprintable() and len(text) <= MAX_TEXT_LEN):
        out.append(Violation("S8", "text must be printable ASCII, one line, <= 200 chars"))

    scan = text.upper()
    for rx in _SYNTHETIC_ALLOW:
        scan = rx.sub(" ", scan)
    for token in _TOKEN.findall(scan):
        if any(rx.fullmatch(token) for rx in _REAL_REG):
            out.append(Violation("S3", f"real-looking registration in text: {token!r}"))
        elif any(rx.fullmatch(token) for rx in _REAL_FLIGHT):
            out.append(Violation("S4", f"real-looking flight designator in text: {token!r}"))
    for rx in _CONTACT:
        if rx.search(text):
            out.append(Violation("S5", "email, URL or phone number in text"))
            break
    for rx in _COORDS:
        if rx.search(text):
            out.append(Violation("S6", "coordinates in text"))
            break
    for rx in _DATES:
        if rx.search(text):
            out.append(Violation("S7", "calendar date or date stamp in text"))
            break
    return out


def schema_violations(msg: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    for f in REQUIRED_FIELDS:
        if f not in msg:
            out.append(Violation("F1", f"missing field {f!r}"))
    if out:
        return out
    if msg["mode"] not in ALLOWED_MODES:
        out.append(Violation("F2", f"mode {msg['mode']!r} not in {ALLOWED_MODES}"))
    if msg["label"] not in IN_SCOPE_LABELS:
        out.append(Violation("F2", f"label {msg['label']!r} not in scope {IN_SCOPE_LABELS}"))
    if not (isinstance(msg["text"], str) and msg["text"].strip()):
        out.append(Violation("F2", "text must be a non-empty string"))
    san = msg["sanitization"]
    if not (isinstance(san, dict) and san.get("method") == SANITIZATION_METHOD
            and san.get("checklist_version") == CHECKLIST_VERSION):
        out.append(Violation(
            "F6", f"sanitization must record method {SANITIZATION_METHOD!r} and "
                  f"checklist_version {CHECKLIST_VERSION!r}"))
    ref = msg["reference"]
    if not isinstance(ref, dict):
        return out + [Violation("F1", "reference must be an object")]
    for f in REQUIRED_REFERENCE_FIELDS:
        if f not in ref:
            out.append(Violation("F1", f"missing reference field {f!r}"))
    if out:
        return out

    # Reference values must equal what structure says (never trusted blindly).
    if isinstance(msg["text"], str) and msg["label"] in IN_SCOPE_LABELS:
        derived = compute_reference(msg["label"], msg["text"])
        for k, v in derived.items():
            if ref[k] != v:
                out.append(Violation("F3", f"reference.{k}={ref[k]!r} but structure gives {v!r}"))
    cat = ref["category"]
    if cat not in CATEGORIES:
        out.append(Violation("F4", f"unknown category {cat!r}"))
    elif not out:
        ok = {
            "valid": ref["well_formed"],
            "valid_lookalike": ref["well_formed"],
            "adversarial_free_text": ref["well_formed"] and bool(ref["free_text_fields"]),
            "malformed": ref["body_matches_label"] is None,
            "label_mismatch": ref["body_matches_label"] not in (None, msg["label"]),
        }[cat]
        if not ok:
            out.append(Violation("F4", f"category {cat!r} contradicts the message structure"))
        if ref["contains_instruction_like_text"] != (cat == "adversarial_free_text"):
            out.append(Violation("F4", "contains_instruction_like_text must match category"))
    return out


def validate_message(msg: dict[str, Any]) -> list[Violation]:
    """All schema and sanitization violations for one message (empty list = pass)."""
    found = schema_violations(msg)
    if any(v.code == "F1" for v in found):
        return found  # cannot safely inspect further
    return found + sanitization_violations(msg)


def add_message(corpus: list[dict[str, Any]], msg: dict[str, Any]) -> None:
    """Append ``msg`` only if it passes; otherwise raise and leave ``corpus`` untouched."""
    violations = validate_message(msg)
    if any(m.get("id") == msg.get("id") for m in corpus):
        violations.append(Violation("F5", f"duplicate id {msg.get('id')!r}"))
    if violations:
        raise CorpusRejection(violations, where=str(msg.get("id", "<no id>")))
    corpus.append(msg)


# ------------------------------------------------- serialization and hashing
def serialize(messages: list[dict[str, Any]]) -> bytes:
    """Canonical JSONL: sorted keys, compact separators, ASCII, LF, trailing newline."""
    lines = [json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
             for m in messages]
    return ("\n".join(lines) + "\n").encode("ascii")


def corpus_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(serialize(messages)).hexdigest()


def _category_counts(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in messages:
        c = m["reference"]["category"]
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items()))


def build_manifest(messages: list[dict[str, Any]], *, name: str, version: str,
                   source: dict[str, Any]) -> dict[str, Any]:
    counts = _category_counts(messages)
    return {
        "name": name,
        "version": version,
        "sha256": corpus_hash(messages),
        "message_count": len(messages),
        "category_counts": dict(sorted(counts.items())),
        "labels_in_scope": list(IN_SCOPE_LABELS),
        "grammar_version": GRAMMAR_VERSION,
        "checklist_version": CHECKLIST_VERSION,
        "checklist": [{"id": i, "rule": r} for i, r in CHECKLIST],
        "source": source,
        "data_boundary": (
            "Synthetic only. No operational captures, proprietary systems "
            "or production access."
        ),
    }


def write_corpus(messages: list[dict[str, Any]], jsonl_path: Path, manifest_path: Path,
                 *, name: str, version: str, source: dict[str, Any]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for m in messages:  # refuse to write anything that would not load
        add_message(checked, m)
    jsonl_path.write_bytes(serialize(checked))
    manifest = build_manifest(checked, name=name, version=version, source=source)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="ascii")
    return manifest


def load_corpus(jsonl_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Load and fully re-validate the corpus. Any failure raises ``CorpusRejection``."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="ascii"))
    missing = [Violation("P1", f"manifest missing {k!r}")
               for k in REQUIRED_MANIFEST_FIELDS if k not in manifest]
    if missing:
        raise CorpusRejection(missing, where="manifest")

    raw = Path(jsonl_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != manifest["sha256"]:
        raise CorpusRejection(
            [Violation("P2", f"hash mismatch: file {actual[:12]}.. manifest {manifest['sha256'][:12]}..")],
            where="corpus",
        )
    messages = [json.loads(line) for line in raw.decode("ascii").splitlines() if line]
    if serialize(messages) != raw:
        raise CorpusRejection([Violation("P3", "file is not in canonical serialization")],
                              where="corpus")
    if len(messages) != manifest["message_count"]:
        raise CorpusRejection([Violation("P4", "message_count does not match file")],
                              where="corpus")
    prov: list[Violation] = []
    if manifest["grammar_version"] != GRAMMAR_VERSION:
        prov.append(Violation("P5", f"grammar_version {manifest['grammar_version']!r} "
                                    f"!= {GRAMMAR_VERSION!r}"))
    if manifest["checklist_version"] != CHECKLIST_VERSION:
        prov.append(Violation("P5", f"checklist_version {manifest['checklist_version']!r} "
                                    f"!= {CHECKLIST_VERSION!r}"))
    if manifest.get("category_counts") != _category_counts(messages):
        prov.append(Violation("P5", "category_counts does not match file"))
    if manifest.get("source", {}).get("kind") != "synthetic":
        prov.append(Violation("P5", "source.kind must be 'synthetic'"))
    if prov:
        raise CorpusRejection(prov, where="manifest")

    verified: list[dict[str, Any]] = []
    failures: list[Violation] = []
    for m in messages:
        try:
            add_message(verified, m)
        except CorpusRejection as e:
            failures.extend(Violation(v.code, f"{m.get('id', '?')}: {v.detail}")
                            for v in e.violations)
    if failures:
        raise CorpusRejection(failures, where="data boundary check")
    return verified