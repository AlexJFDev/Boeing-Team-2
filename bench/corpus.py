"""Corpus versioning and corpus hash (WBS 1.14, issue #22).

Every corpus file under data/ is listed, with its sha256, in the MANIFEST.json of its
own directory. A file is loaded only if its bytes match that entry, so an edited,
corrupted or unlisted file stops the run instead of silently changing results.

Two manifest formats are accepted:

    {"file": "frames_v1.0.0.jsonl", "sha256": "...", ...}        one file (data/adsb, WBS 2.1)
    {"files": {"adsb_v1.0.0.txt": "...", ...}}                   several files (tasks, preambles)

Versioning rule (CLAUDE.md rule 4): a file name carries its version (``_vX.Y.Z``),
and a manifest entry is never changed once added. A change is a new file with a
bumped version and a new entry. ``add_to_manifest`` refuses to overwrite an entry,
so editing a frozen file in place cannot be made to pass by re-hashing it.

The corpus hash of a run is one sha256 over the sorted list of
``<sha256>  <repo-relative path>`` lines, the same format ``sha256sum`` prints, for
every file the task uses: its task file, the preambles it names, and any data files
passed in. It depends only on file bytes and repo-relative paths, and data/ is
marked ``-text`` in .gitattributes, so it is identical on every machine.

Command line:

    python -m bench.corpus data/adsb/frames_v1.0.0.jsonl     # verify one file, print its sha256
    python -m bench.corpus data/tasks/loader_demo_v0.1.0.jsonl --corpus
                                                            # verify a task file's corpus, print the corpus hash
    python -m bench.corpus --add data/tasks/new_task_v1.0.0.jsonl
                                                            # list a new file in its manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bench.provenance import sha256_file

CORPUS_VERSION = "1.0.0"  # version of this hashing scheme, recorded with every hash

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "MANIFEST.json"
PREAMBLE_DIR = REPO_ROOT / "data" / "preambles"
VERSIONED_NAME = re.compile(r"_v\d+\.\d+\.\d+\.[A-Za-z0-9]+$")


class CorpusError(ValueError):
    """A corpus file is missing from its manifest or does not match it."""


def relpath(path: Path) -> str:
    """The repo-relative POSIX path, the form that goes into the corpus hash."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        raise CorpusError(f"{path} is outside the repository") from None


def manifest_entries(manifest_path: Path) -> dict[str, str]:
    """File name -> sha256 for every file a manifest lists (either format)."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CorpusError(f"no {MANIFEST_NAME} in {Path(manifest_path).parent}") from None
    if isinstance(manifest.get("files"), dict):
        return {str(k): str(v) for k, v in manifest["files"].items()}
    if "file" in manifest and "sha256" in manifest:
        return {str(manifest["file"]): str(manifest["sha256"])}
    raise CorpusError(f"{manifest_path} lists no files")


def verify_file(path: str | Path) -> str:
    """Check a corpus file against its directory's manifest and return its sha256.

    Raises CorpusError if the manifest does not list the file, or if its bytes differ
    from the recorded hash (it was edited in place, corrupted, or not committed as is).
    """
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"{path} does not exist")
    entries = manifest_entries(path.parent / MANIFEST_NAME)
    expected = entries.get(path.name)
    if expected is None:
        raise CorpusError(
            f"{path.name} is not listed in {path.parent.name}/{MANIFEST_NAME}; "
            f"add it with: python -m bench.corpus --add {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise CorpusError(
            f"{path.name} does not match {path.parent.name}/{MANIFEST_NAME} "
            f"(sha256 {actual}, expected {expected}). Corpus files are never edited in "
            f"place: restore it with git, or save the change as a new version.")
    return actual


def corpus_hash(files: dict[str, str]) -> str:
    """One sha256 over ``<sha256>  <path>`` lines, sorted by path (sha256sum format)."""
    listing = "".join(f"{sha}  {path}\n" for path, sha in sorted(files.items()))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def preamble_names(task_file: Path) -> list[str]:
    """The preambles a task file names, in order of first use."""
    names: list[str] = []
    for line in Path(task_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            name = json.loads(line).get("preamble")
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def task_corpus(task_file: str | Path, data_files: Iterable[str | Path] = (),
                preamble_dir: Path = PREAMBLE_DIR) -> dict[str, Any]:
    """Verify every file a task uses and describe its corpus for the EvalLog.

    Covers the task file, every preamble it names, and ``data_files`` (for example
    the frame set a task was built from). Raises CorpusError on the first file that
    is unlisted or does not match.
    """
    paths = [Path(task_file)]
    paths += [preamble_dir / f"{name}.txt" for name in preamble_names(Path(task_file))]
    paths += [Path(p) for p in data_files]
    files = {relpath(p): verify_file(p) for p in paths}
    return {
        "corpus_hash": corpus_hash(files),
        "corpus_files": dict(sorted(files.items())),
        "corpus_hash_scheme": CORPUS_VERSION,
    }


def add_to_manifest(path: str | Path) -> str:
    """List a new corpus file in its directory's multi-file manifest. Never overwrites."""
    path = Path(path)
    if not VERSIONED_NAME.search(path.name):
        raise CorpusError(f"{path.name} has no version in its name (expected ..._vX.Y.Z.ext)")
    manifest_path = path.parent / MANIFEST_NAME
    manifest: dict[str, Any] = {"files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("files"), dict):
            raise CorpusError(f"{manifest_path} is a single-file manifest; it is written by its generator")
    sha = sha256_file(path)
    existing = manifest["files"].get(path.name)
    if existing is not None and existing != sha:
        raise CorpusError(
            f"{path.name} is already listed with a different sha256. Listed files are "
            f"frozen: save the change as a new version instead.")
    manifest["files"][path.name] = sha
    manifest["files"] = dict(sorted(manifest["files"].items()))
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify corpus files against their manifests.")
    parser.add_argument("file", help="a corpus file under data/")
    parser.add_argument("--corpus", action="store_true",
                        help="treat FILE as a task file: verify it and its preambles, print the corpus hash")
    parser.add_argument("--add", action="store_true", help="list FILE in its directory's manifest")
    args = parser.parse_args(argv)
    try:
        if args.add:
            print(f"added {args.file}: sha256 {add_to_manifest(args.file)}")
        elif args.corpus:
            corpus = task_corpus(args.file)
            for path, sha in corpus["corpus_files"].items():
                print(f"ok  {sha}  {path}")
            print(f"corpus hash: {corpus['corpus_hash']}")
        else:
            print(f"ok  {verify_file(args.file)}  {args.file}")
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
