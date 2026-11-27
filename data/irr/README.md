# Inter-rater agreement pilots

One directory per pilot (WBS 3.9), written by `python -m bench.irr`. See
[docs/irr_pilot.md](../../docs/irr_pilot.md).

| File | Written by | Contents |
| --- | --- | --- |
| `packet.jsonl` | `sample` | The blind grading packet: rubric, prompt and answer per item |
| `key.json` | `sample` | Item → task, sample and model; seed; packet hash; rubric fingerprint |
| `grades/<rater>.csv` | each rater | `item_id,grade,note`, grade `safe` or `unsafe` |
| `agreement.json`, `agreement.md` | `score` | Krippendorff's alpha, raw agreement, per-clause breakdown, disagreements |
| `REVISION.md` | `score`, below threshold only | The rubric sent back: disputed items by clause, with both raters' notes |

Records are kept, including failed ones. Never edit a pilot after it has been scored;
run a new one instead. `python -m bench.irr gate` reads these records to decide whether
the corpus can be frozen.

The answers in a packet come from models running on the benchmark's synthetic or
approved-source tasks (CLAUDE.md rule 1).
