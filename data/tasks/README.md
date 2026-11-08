# Task files

JSON Lines, one task per line, loaded by `bench/loader.py` (WBS 1.13). Field
reference is in the loader's docstring. `family` is the catalogue letter (A ADS-B,
B ARINC 429, C ARINC 664, D ACARS); the item type such as A1 goes in `metadata`.

| File | Contents |
| --- | --- |
| `loader_demo_v0.1.0.jsonl` | Three ADS-B tasks that exercise the loader (valid decode, parity failure, single CPR frame with a per-task premise). A demo, not part of the scored corpus; the A1/A2 tasks come from WBS 2.5 and 2.6. |

Version rules match `data/adsb/README.md`: a file that a run has used is frozen, and
a change gets a new file name with a bumped version.

Every file here is listed with its sha256 in `MANIFEST.json` (WBS 1.14). The loader
refuses a task file that is unlisted or does not match. List a new file with:

```bash
python -m bench.corpus --add data/tasks/<name>_vX.Y.Z.jsonl
```

That command never changes an existing entry, so an edited file cannot be re-listed;
save the change as a new version instead.
| `a1_adsb_decode_v1.0.0.jsonl` | A1 ADS-B identification and altitude decode (WBS 2.5, 29 tasks): 13 identification frames (10 DF17, 3 DF18), one altitude frame per synthetic track (10), and 6 parity failures to reject. Built by `scripts/adsb/build_a1_tasks.py`; targets come from pyModeS and are re-checked against it every time the task is built (`python -m bench.adsb.a1`). |
| `loader_demo_v0.1.0.jsonl` | Three ADS-B tasks that exercise the loader (valid decode, parity failure, single CPR frame with a per-task premise). A demo, not part of the scored corpus; the A1/A2 tasks come from WBS 2.5 and 2.6. |
| `a2_cpr_position_v1.0.0.jsonl` | A2 ADS-B CPR position decode (WBS 2.6, 28 tasks): 10 global even/odd pairs, 10 local decodes against a stated receiver position, 2 stale pairs (over 10 s apart), 4 same-format pairs and 2 single frames with no reference. The last three kinds have the target `{"answer":"cannot_be_determined"}`. Built by `scripts/adsb/build_a2_tasks.py`; targets come from the CPR verifier (`bench/adsb/cpr.py`) and are re-checked against it every time the task is built (`python -m bench.adsb.a2`). |
| `a1_adsb_decode_v1.1.0.jsonl` | A1 v1.0.0 plus unsafe-if clauses (WBS 2.7): the same 29 tasks, prompts and targets, with `metadata.unsafe_if` added. The 6 parity failures carry a rule-checked clause (`a1.corrupt_frame_trusted`); the decodable frames carry judge clauses (`a1.identity_overtrusted`, and `a1.source_misattributed` for DF 18). The current A1 version. |
| `a2_cpr_position_v1.1.0.jsonl` | A2 v1.0.0 plus unsafe-if clauses (WBS 2.7): the same 28 tasks, prompts and targets, with `metadata.unsafe_if` added. The 8 cannot-be-determined tasks carry a rule-checked clause (`a2.position_without_basis`); the decodable ones carry a judge clause (`a2.position_overtrusted`). The current A2 version. |
| `b1_arinc429_word_decode_v1.0.0.jsonl` | B1 ARINC 429 word decode (WBS 3.3, 27 tasks): 21 good words, 3 per label profile with different SSMs (15 BNR, 6 BCD); 4 parity failures to reject (2 with the parity bit flipped, 2 with another bit flipped); and 2 words on label 310, which has no profile, so their value is `cannot_be_determined`. Words come from the seeded generator (`bench/arinc429/generator.py`, seed 4290). Built by `scripts/arinc429/build_b1_tasks.py`; targets come from the reference decoder (`bench/arinc429/decoder.py`) and are re-checked against it every time the task is built (`python -m bench.arinc429.b1`). Uses the `arinc429_v1.0.0` preamble. |
| `b2_arinc429_ssm_interpretation_v1.0.0.jsonl` | B2 ARINC 429 SSM interpretation (WBS 3.4, 14 tasks). Each task shows two words with their labels, one BNR and one BCD with the same SSM bits, and asks for each word's SSM bits and meaning: 4 pairs with SSM 11 and 4 with 00 (where the BNR and BCD tables differ), 2 each with 01 and 10 (controls, where they agree), and 2 pairs in which one word fails parity and must be rejected. Words come from the seeded generator (seed 5290). Built by `scripts/arinc429/build_b2_tasks.py`; targets come from the reference decoder and are re-checked against it every time the task is built (`python -m bench.arinc429.b2`). Uses the `arinc429_v1.0.0` preamble. |

Version rules match `data/adsb/README.md`: a file that a run has used is frozen, and
a change gets a new file name with a bumped version.

## Unsafe-if clauses

Since WBS 2.7, every A-family task carries its own unsafe-if clauses in
`metadata.unsafe_if` (format and rules in `bench/adsb/unsafe_if.py`). Each clause names
the task's frames and values, and says what response would be unsafe for that task.
The model never sees the clauses. `detect: "rule"` clauses are graded now by the
safety scorer `bench/safety.py`, which runs beside the capability scorer and is never
combined with it. `detect: "judge"` clauses are worded for the judge model (WBS 6.7)
and are listed as pending until it exists. Clauses are rebuilt from the frames
whenever the task validates, so an edited clause stops the run.
