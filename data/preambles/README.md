# Operational preambles

Each file is the preamble for one task family at one version. The loader
(`bench/loader.py`, WBS 1.13) puts the preamble at the start of every task prompt
that names it, so the premises that make an answer determinate (protocol version,
bus rates, resync rules, timing limits) are stated in the task instead of assumed.

The preamble goes in the user message, not the system message. It is part of the
task and must be identical for every configuration we compare. The system message
is a configuration variable (prompt templates, guardrail profiles) and must not be
able to drop it.

Premises that change from task to task do not go here. The catalogue's C1 items
must state which resync rule the receiver uses, and the same frames give different
answers under the two rules; B3 items state a label's configured transmission rate.
Those go in the task's own `premises` list, which the loader prints after the family
preamble under "Premises for this task:".

Tasks refer to a preamble by name, for example `"preamble": "adsb_v1.0.0"`, which
resolves to `adsb_v1.0.0.txt` here. A name with no file fails the load and reports
the task ID.

## Rules

- The file text is sent to the model exactly as written, so it holds premises only.
  Sources and notes go in this README.
- Never edit a preamble in place once a task uses it. Copy it to a new version
  (`adsb_v1.1.0.txt`), make the change, and point the tasks at the new name.
- Files live under `data/`, which `.gitattributes` marks `-text`, so line endings
  (and therefore hashes) are the same on every machine.
- Every preamble is listed with its sha256 in `MANIFEST.json` (WBS 1.14), and a task
  that uses an edited or unlisted preamble fails to load. Add a new one with
  `python -m bench.corpus --add data/preambles/<family>_vX.Y.Z.txt`.

## adsb_v1.0.0 sources

| Premise | Source | Confidence |
| --- | --- | --- |
| 1. DF17/DF18, 112 bits, 28 hex characters | ICAO Annex 10 Vol IV (Mode S extended squitter); matches `bench/adsb/oracle.py` | Primary, section not cited |
| 2. ADS-B Version 2 / DO-260B layouts | `data/adsb/README.md`: frames are encoded to the DO-260B layout | Project decision |
| 3. 24-bit parity; reject on failure | ICAO Annex 10 Vol IV (Mode S parity); project rule that CRC failures are rejected, not decoded | Primary, section not cited |
| 4. Even + odd within 10 s for a global position | ICAO Annex 10 Vol III, Part I, Appendix to Ch. 5, §2.6.7 "Globally unambiguous airborne position decoding": "The time between the 'even' and 'odd' format encoded position reports shall be no longer than 10 seconds." | Primary (2002 edition) |
| 5. Cannot-be-determined is allowed | Benchmark design: some items have "cannot be determined" as the correct answer | Project decision |
| 6–7. Synthetic times and identifiers | `data/adsb/README.md` data boundary | Project data |

DO-260B itself is paywalled; nothing here is quoted from it.
