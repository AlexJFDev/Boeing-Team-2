# Frozen corpora

One JSON record per frozen corpus release, written by `python -m bench.freeze freeze`
(WBS 3.11). `protocol_v1.0.0.json` will be protocol corpus v1.0: the A1, A2 and B1-B4
task files and the preambles they name. See the "Freezing v1.0" section of
[docs/corpus.md](../../docs/corpus.md).

A record holds the corpus hash, each family's run-level corpus hash (the one its
EvalLogs record), every file with its sha256, the task, scorer and oracle versions,
and the checks that passed. The commit that adds it carries the annotated tag
`corpus-<name>-v<release>`.

Records are never edited or rewritten. `python -m bench.freeze verify`, run in CI,
fails if any file a record lists has changed or gone. A change after the freeze is a
new corpus release with its own record (CLAUDE.md rule 4).
