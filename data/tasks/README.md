# Task files

JSON Lines, one task per line, loaded by `bench/loader.py` (WBS 1.13). Field
reference is in the loader's docstring. `family` is the catalogue letter (A ADS-B,
B ARINC 429, C ARINC 664, D ACARS); the item type such as A1 goes in `metadata`.

| File | Contents |
| --- | --- |
| `loader_demo_v0.1.0.jsonl` | Three ADS-B tasks that exercise the loader (valid decode, parity failure, single CPR frame with a per-task premise). A demo, not part of the scored corpus; the A1/A2 tasks come from WBS 2.5 and 2.6. |

Version rules match `data/adsb/README.md`: a file that a run has used is frozen, and
a change gets a new file name with a bumped version.
